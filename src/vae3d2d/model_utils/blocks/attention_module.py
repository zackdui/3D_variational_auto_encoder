from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionND(nn.Module):
    """
    N-D (2D / 3D) self-attention over feature maps.

    Supports:
      - 2D: x shape (B, C, H, W)
      - 3D: x shape (B, C, D, H, W)
      - Global attention (window_size=None)
      - Windowed attention (window_size=(..., ...))
      - Optional relative positional bias (for windowed mode)

    Output has the same shape as the input.
    """

    def __init__(
        self,
        in_channels: int,
        num_heads: int = 4,
        dim_head: Optional[int] = None,
        dropout: float = 0.0,
        spatial_dims: int = 3,
        window_size: Optional[Sequence[int]] = None,
        use_rel_pos_bias: bool = False,
    ):
        super().__init__()
        assert spatial_dims in (2, 3), "spatial_dims must be 2 or 3."
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.use_rel_pos_bias = use_rel_pos_bias

        # Determine head dimension / inner dimension
        if dim_head is None:
            # default: split channels evenly across heads
            assert (
                in_channels % num_heads == 0
            ), "If dim_head is None, in_channels must be divisible by num_heads."
            dim_head = in_channels // num_heads
        self.dim_head = dim_head
        inner_dim = num_heads * dim_head

        self.scale = dim_head ** -0.5

        # Project to QKV jointly, then chunk
        self.to_qkv = nn.Linear(in_channels, inner_dim * 3, bias=False)

        # Project back to in_channels
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, in_channels),
            nn.Dropout(dropout),
        )

        # Normalize over channels in token space
        self.norm = nn.LayerNorm(in_channels)

        # Window configuration
        if window_size is not None:
            if isinstance(window_size, int):
                window_size = (window_size,) * spatial_dims
            else:
                window_size = tuple(window_size)
            assert len(window_size) == spatial_dims, "window_size must match spatial_dims."
        self.window_size: Optional[Tuple[int, ...]] = window_size

        # Relative positional bias (only for windowed attention)
        self.relative_position_bias_table = None
        self.register_buffer("relative_position_index", None, persistent=False)

        if self.use_rel_pos_bias:
            if self.window_size is None:
                raise ValueError("use_rel_pos_bias=True requires window_size to be set.")
            self._init_relative_position_bias()


    # Relative positional bias helpers
    def _init_relative_position_bias(self):
        """
        Build relative_position_index and bias table for a window of size window_size.
        Works for 2D or 3D. Follows the same spirit as your example / Swin.
        """
        ws = self.window_size
        assert ws is not None

        if self.spatial_dims == 3:
            Wd, Wh, Ww = ws
            num_tokens = Wd * Wh * Ww

            coords_d = torch.arange(Wd)
            coords_h = torch.arange(Wh)
            coords_w = torch.arange(Ww)
            # 3, Wd, Wh, Ww
            coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing="ij"))
            # 3, Wd*Wh*Ww
            coords_flatten = coords.view(3, -1)
            # 3, N, N
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            # N, N, 3
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()

            # shift to start from 0
            relative_coords[:, :, 0] += Wd - 1
            relative_coords[:, :, 1] += Wh - 1
            relative_coords[:, :, 2] += Ww - 1

            # flatten indices for table
            # like in your example: 2*Wd-1, 2*Wh-1, 2*Ww-1
            dd = (2 * Wh - 1) * (2 * Ww - 1)
            dh = (2 * Ww - 1)
            relative_coords[:, :, 0] *= dd
            relative_coords[:, :, 1] *= dh
            relative_position_index = relative_coords.sum(-1)  # N, N

            num_rel_positions = (2 * Wd - 1) * (2 * Wh - 1) * (2 * Ww - 1)

        else:  # 2D
            Wh, Ww = ws
            num_tokens = Wh * Ww

            coords_h = torch.arange(Wh)
            coords_w = torch.arange(Ww)
            coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # 2, Wh, Ww
            coords_flatten = coords.view(2, -1)  # 2, N
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, N, N
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # N, N, 2

            relative_coords[:, :, 0] += Wh - 1
            relative_coords[:, :, 1] += Ww - 1

            relative_coords[:, :, 0] *= (2 * Ww - 1)
            relative_position_index = relative_coords.sum(-1)  # N, N

            num_rel_positions = (2 * Wh - 1) * (2 * Ww - 1)

        # Register index buffer
        self.register_buffer("relative_position_index", relative_position_index, persistent=False)

        # Learnable bias table: (num_rel_positions, num_heads)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(num_rel_positions, self.num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        self.num_window_tokens = num_tokens

    def _get_rel_pos_bias(self) -> torch.Tensor:
        """
        Returns a tensor of shape (num_heads, Nw, Nw) where Nw is tokens per window.
        """
        assert self.use_rel_pos_bias
        assert self.relative_position_index is not None
        table = self.relative_position_bias_table  # (num_rel_positions, num_heads)
        index = self.relative_position_index  # (Nw, Nw)

        Nw = index.shape[0]
        # (Nw*Nw, num_heads)
        bias = table[index.view(-1)]
        # (Nw, Nw, num_heads)
        bias = bias.view(Nw, Nw, self.num_heads)
        # (num_heads, Nw, Nw)
        bias = bias.permute(2, 0, 1).contiguous()
        return bias

    # Core attention on (B', N, C)
    def _attention_tokens(self, x_tokens: torch.Tensor) -> torch.Tensor:
        """
        x_tokens: (B', N, C)
        returns:  (B', N, C)
        """
        Bp, N, C = x_tokens.shape
        x_norm = self.norm(x_tokens)  # (B', N, C)

        qkv = self.to_qkv(x_norm)  # (B', N, 3*inner_dim)
        q, k, v = qkv.chunk(3, dim=-1)  # each (B', N, inner_dim)

        H = self.num_heads
        Dh = self.dim_head
        inner_dim = H * Dh

        # (B', N, H*Dh) -> (B', H, N, Dh)
        q = q.view(Bp, N, H, Dh).transpose(1, 2)  # (B', H, N, Dh)
        k = k.view(Bp, N, H, Dh).transpose(1, 2)
        v = v.view(Bp, N, H, Dh).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B', H, N, N)

        # Add relative positional bias if available (for windowed attention)
        if self.use_rel_pos_bias:
            bias = self._get_rel_pos_bias()  # (H, N, N)
            attn_scores = attn_scores + bias.unsqueeze(0)  # broadcast over batch

        attn = F.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn, v)  # (B', H, N, Dh)

        # Merge heads: (B', N, H*Dh)
        out = out.transpose(1, 2).contiguous().view(Bp, N, inner_dim)

        # Project back to C
        out = self.to_out(out)  # (B', N, C)

        return out

    # Window partition / reverse helpers (2D and 3D)
    def _compute_padding(self, size: int, window: int):
        """
        Compute how much padding is needed at the end of a dimension so it becomes divisible.
        Returns (pad_before, pad_after) — but we only pad at the end here.
        """
        remainder = size % window
        if remainder == 0:
            return (0, 0)
        pad_after = window - remainder
        return (0, pad_after)

    def _pad_input_2d(self, x: torch.Tensor):
        """
        x: (B, C, H, W)
        Returns:
            padded_x: (B, C, H_pad, W_pad)
            pads: (pad_h, pad_w)
        """
        _, _, H, W = x.shape
        Wh, Ww = self.window_size

        pad_h = self._compute_padding(H, Wh)[1]
        pad_w = self._compute_padding(W, Ww)[1]

        # F.pad pads in reverse order: (W_left, W_right, H_left, H_right)
        x_padded = F.pad(x, (0, pad_w, 0, pad_h))

        return x_padded, (pad_h, pad_w)

    def _pad_input_3d(self, x: torch.Tensor):
        """
        x: (B, C, D, H, W)
        Returns:
            padded_x: (B, C, D_pad, H_pad, W_pad)
            pads: (pad_d, pad_h, pad_w)
        """
        _, _, D, H, W = x.shape
        Wd, Wh, Ww = self.window_size

        pad_d = self._compute_padding(D, Wd)[1]
        pad_h = self._compute_padding(H, Wh)[1]
        pad_w = self._compute_padding(W, Ww)[1]

        # F.pad order: (W_left, W_right, H_left, H_right, D_left, D_right)
        x_padded = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))

        return x_padded, (pad_d, pad_h, pad_w)

    def _unpad_output(self, x: torch.Tensor, pads):
        """
        Remove trailing padding from output.
        pads: 2-tuple for 2D, 3-tuple for 3D.
        """
        if x.dim() == 4:  # (B, C, H, W)
            pad_h, pad_w = pads
            if pad_h > 0:
                x = x[:, :, :-pad_h, :]
            if pad_w > 0:
                x = x[:, :, :, :-pad_w]
        else:  # (B, C, D, H, W)
            pad_d, pad_h, pad_w = pads
            if pad_d > 0:
                x = x[:, :, :-pad_d, :, :]
            if pad_h > 0:
                x = x[:, :, :, :-pad_h, :]
            if pad_w > 0:
                x = x[:, :, :, :, :-pad_w]
        return x

    def _partition_windows_2d(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
        """
        x: (B, C, H, W)
        returns:
          x_windows: (B * num_windows, Nw, C)
          sizes: (B, C, H, W)
        """
        B, C, H, W = x.shape
        Wh, Ww = self.window_size
        assert H % Wh == 0 and W % Ww == 0, "H, W must be divisible by window_size."

        Gh = H // Wh
        Gw = W // Ww
        # (B, C, Gh, Wh, Gw, Ww)
        x = x.view(B, C, Gh, Wh, Gw, Ww)
        # (B, Gh, Gw, Wh, Ww, C)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        # (B * Gh * Gw, Wh * Ww, C)
        x_windows = x.view(B * Gh * Gw, Wh * Ww, C)
        return x_windows, (B, C, H, W)

    def _merge_windows_2d(self, x_windows: torch.Tensor, sizes: Tuple[int, int, int, int]) -> torch.Tensor:
        """
        x_windows: (B * Gh * Gw, Wh * Ww, C)
        sizes: (B, C, H, W)
        returns:
          x: (B, C, H, W)
        """
        B, C, H, W = sizes
        Wh, Ww = self.window_size
        Gh = H // Wh
        Gw = W // Ww
        # (B, Gh, Gw, Wh, Ww, C)
        x = x_windows.view(B, Gh, Gw, Wh, Ww, C)
        # (B, C, H, W)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, C, H, W)
        return x

    def _partition_windows_3d(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int, int, int]]:
        """
        x: (B, C, D, H, W)
        returns:
          x_windows: (B * Gd * Gh * Gw, Nw, C) where Nw = Wd*Wh*Ww
          sizes: (B, C, D, H, W)
        """
        B, C, D, H, W = x.shape
        Wd, Wh, Ww = self.window_size
        assert D % Wd == 0 and H % Wh == 0 and W % Ww == 0, "D,H,W must be divisible by window_size."

        Gd = D // Wd
        Gh = H // Wh
        Gw = W // Ww

        # (B, C, Gd, Wd, Gh, Wh, Gw, Ww)
        x = x.view(B, C, Gd, Wd, Gh, Wh, Gw, Ww)
        # (B, Gd, Gh, Gw, Wd, Wh, Ww, C)
        x = x.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous()
        # (B * Gd * Gh * Gw, Wd * Wh * Ww, C)
        x_windows = x.view(B * Gd * Gh * Gw, Wd * Wh * Ww, C)
        return x_windows, (B, C, D, H, W)

    def _merge_windows_3d(self, x_windows: torch.Tensor, sizes: Tuple[int, int, int, int, int]) -> torch.Tensor:
        """
        x_windows: (B * Gd * Gh * Gw, Nw, C)
        sizes: (B, C, D, H, W)
        returns:
          x: (B, C, D, H, W)
        """
        B, C, D, H, W = sizes
        Wd, Wh, Ww = self.window_size
        Gd = D // Wd
        Gh = H // Wh
        Gw = W // Ww

        # (B, Gd, Gh, Gw, Wd * Wh * Ww, C)
        x = x_windows.view(B, Gd, Gh, Gw, Wd * Wh * Ww, C)
        # (B, Gd, Gh, Gw, Wd, Wh, Ww, C)
        x = x.view(B, Gd, Gh, Gw, Wd, Wh, Ww, C)
        # (B, C, D, H, W)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous().view(B, C, D, H, W)
        return x

    # Forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W) or (B, C, D, H, W)
        returns: same shape as x
        """
        orig_x = x
        if self.spatial_dims == 2:
            assert x.dim() == 4, f"Expected 4D input for spatial_dims=2, got {x.dim()}D."
            B, C, H, W = x.shape
        else:
            assert x.dim() == 5, f"Expected 5D input for spatial_dims=3, got {x.dim()}D."
            B, C, D, H, W = x.shape

        if self.window_size is None:
            # Global attention over all spatial positions
            # (B, C, *spatial) -> (B, N, C)
            if self.spatial_dims == 2:
                N = H * W
                x_tokens = x.view(B, C, N).transpose(1, 2)  # (B, N, C)
                out_tokens = self._attention_tokens(x_tokens)  # (B, N, C)
                out = out_tokens.transpose(1, 2).contiguous().view(B, C, H, W)
            else:
                N = D * H * W
                x_tokens = x.view(B, C, N).transpose(1, 2)  # (B, N, C)
                out_tokens = self._attention_tokens(x_tokens)  # (B, N, C)
                out = out_tokens.transpose(1, 2).contiguous().view(B, C, D, H, W)
        else:
            # Windowed attention
            if self.spatial_dims == 2:
                x, pads = self._pad_input_2d(x)
                x_windows, sizes = self._partition_windows_2d(x)  # (B*num_windows, Nw, C)
                out_windows = self._attention_tokens(x_windows)  # same shape
                out = self._merge_windows_2d(out_windows, sizes)
            else:
                x, pads = self._pad_input_3d(x)
                x_windows, sizes = self._partition_windows_3d(x)  # (B*num_windows, Nw, C)
                out_windows = self._attention_tokens(x_windows)
                out = self._merge_windows_3d(out_windows, sizes)
            
            out = self._unpad_output(out, pads)

        # Residual connection
        return orig_x + out


if __name__ == "__main__":
    # Example usage of SelfAttentionND

    # 3D global
    attn3d = SelfAttentionND(
        in_channels=16,
        num_heads=4,
        dim_head=None,       # will use C_enc / num_heads
        dropout=0.0,
        spatial_dims=3,
        window_size=None,    # global attention
        use_rel_pos_bias=False,
    )

    # 3D patch
    attn3d_window = SelfAttentionND(
        in_channels=64,
        num_heads=4,
        dim_head=None,
        dropout=0.0,
        spatial_dims=3,
        window_size=(3, 8, 8),  # must divide D,H,W
        use_rel_pos_bias=True,
    )

    attn2d = SelfAttentionND(
        in_channels=64,
        num_heads=4,
        spatial_dims=2,
        window_size=(8, 8),      # or None for global
        use_rel_pos_bias=True,
    )


