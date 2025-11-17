# This will check the feasibility of a batch size on your GPU

import torch
from torch.cuda import OutOfMemoryError
import torch.nn as nn


def find_max_batch_size_power2(
    model,
    spatial_size,              # e.g. (C, D, H, W)
    device="cuda",
    max_power=8,               # tests up to 2**8 = 256
    use_amp=False,
    amp_dtype=torch.float16,
    optimizer=None,
):
    """
    Return the largest power-of-two batch size (1,2,4,...)
    that fits in GPU memory.
    Args:
        model: nn.Module. In train mode, forward should return (out, loss).
        spatial_size: tuple like (C, D, H, W) or (C, H, W) or (C, L).
                      The batch dimension will be added as the first dim.
        device: "cuda", "cuda:0", or "cpu".
        max_power: test powers up to 2**max_power
        use_amp: whether to run under autocast (mixed precision).
        amp_dtype: dtype for autocast (e.g., torch.float16 or torch.bfloat16).
        optimizer: optional torch.optim.Optimizer. If given, we zero_grad each try.

    Returns:
        best: largest batch size that runs without OOM. 0 if even batch_size=1 fails.
    """

    device_type = "cuda" if "cuda" in device else "cpu"
    model = model.to(device)
    model.train()

    C, *spatial = spatial_size

    scaler = torch.amp.GradScaler(enabled=(use_amp and device_type == "cuda"))

    # --- make random batch ---
    def make_random_batch(batch_size):
        shape = (batch_size, C, *spatial)
        return torch.randn(*shape, device=device)

    # --- test if batch_size fits ---
    def can_run(batch_size):
        try:
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

            x = make_random_batch(batch_size)

            with torch.amp.autocast(
                device_type=device_type,
                dtype=amp_dtype,
                enabled=use_amp,
            ):
                out, loss = model(x)

            if use_amp and device_type == "cuda":
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if device_type == "cuda":
                torch.cuda.synchronize()
            return True

        except OutOfMemoryError:
            pass
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise

        finally:
            if "x" in locals():
                del x
            if "out" in locals():
                del out
            if "loss" in locals():
                del loss
            if device_type == "cuda":
                torch.cuda.empty_cache()

        return False

    # --- search only powers of two ---
    best = 0
    for p in range(max_power + 1):     # 0 → max_power
        bs = 2 ** p
        if can_run(bs):
            best = bs
        else:
            break

    return best

def test_memory(model, use_amp=False):
        device = "cuda"

        model.to(device)
        model.train()

        # Dummy input
        x = torch.randn(1, 1, 208, 512, 512, device=device)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        print(f"\n=== Testing use_amp={use_amp} ===")
        print("Before forward: ", torch.cuda.memory_allocated(device) / 1024**2, "MB")

        if use_amp:
            autocast = torch.amp.autocast("cuda", dtype=torch.bfloat16)
        else:
            autocast = torch.no_grad()  # no-op context, replaced below
            autocast.__enter__ = lambda *a, **k: None
            autocast.__exit__ = lambda *a, **k: None

        with torch.amp.autocast("cuda", dtype=torch.float16) if use_amp else torch.autocast("cuda", enabled=False):
            out, loss = model(x)

        print("After forward:  ", torch.cuda.memory_allocated(device) / 1024**2, "MB")
        print("Peak forward:   ", torch.cuda.max_memory_allocated(device) / 1024**2, "MB")

        loss.backward()
        print("After backward: ", torch.cuda.memory_allocated(device) / 1024**2, "MB")
        print("Peak total:     ", torch.cuda.max_memory_allocated(device) / 1024**2, "MB")
        print("Peak reserved: ", torch.cuda.max_memory_reserved() / 1024**2, "MB")



