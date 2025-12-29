from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Eagle_Loss(nn.Module):
    """
    Eagle_Loss is a custom loss function designed for image reconstruction tasks, with an emphasis on preserving
    textures and edges in the reconstructed images. It operates by analyzing the variance of image gradients within
    patches of the image, and by computing loss in the frequency domain using a high-pass filter.

    Parameters:
        patch_size (int): Defines the size of the patches used to calculate the variance in gradients.
        cutoff (float): The cutoff frequency for the high-pass filter used in gaussian high-pass filtering.
        cpu (bool): Determines whether the kernels are stored on the CPU (if True) or on CUDA (if False).

    Methods:
        forward(output, target): Calculates the Eagle loss between the output and target images.
        calculate_gradient(img): Computes the x and y gradients of an image using the Scharr filters.
        calculate_patch_loss(output_gradient, target_gradient): Measures the loss based on the variance of
            gradients within the image patches.
        gaussian_highpass_weights2d(size): Generates the weights for high-pass filtering in the frequency domain.
        fft_loss(pred, gt): Calculates the loss in the frequency domain using the high-pass filter.

    Example Usage:
        eagle_loss = Eagle_Loss(patch_size=3)
        loss = eagle_loss(output_image, target_image)
    """
    def __init__(self, patch_size, cpu=False, cutoff=0.5):
        super(Eagle_Loss, self).__init__()
        self.patch_size = patch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() and not cpu else 'cpu')
        self.cutoff = cutoff

        # Scharr kernel for the gradient map calculation
        kernel_values = [[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]]
        # self.kernel_x = nn.Parameter(
        #     torch.tensor(kernel_values, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device),
        #     requires_grad=False)
        # self.kernel_y = nn.Parameter(
        #     torch.tensor(kernel_values, dtype=torch.float32).t().unsqueeze(0).unsqueeze(0).to(self.device),
        #     requires_grad=False)
        
        kx = torch.tensor(kernel_values, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        ky = torch.tensor(kernel_values, dtype=torch.float32).t().unsqueeze(0).unsqueeze(0)

        
        # register as buffers, no device specified; model.to(device) will move them
        self.register_buffer("kernel_x", kx, persistent=False)
        self.register_buffer("kernel_y", ky, persistent=False)

        # Operation for unfolding image into non-overlapping patches
        self.unfold = nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size).to(self.device)

    def rgb_to_grayscale(self, x):
        """Convert RGB image to grayscale using standard weights"""
        if x.size(1) == 3:
            # Use standard RGB to grayscale conversion weights
            weights = torch.tensor([0.2989, 0.5870, 0.1140]).to(x.device)
            weights = weights.view(1, 3, 1, 1)
            return (x * weights).sum(dim=1, keepdim=True)
        return x

    def forward(self, output, target):
        # force FP32, disable autocast inside
        # with torch.amp.autocast(device_type="cuda", enabled=False):
        output32 = output.float()
        target32 = target.float()
        # output, target = output.to(self.device), target.to(self.device)
        
        # Convert to grayscale if RGB
        output = self.rgb_to_grayscale(output32)
        target = self.rgb_to_grayscale(target32)

        if output.size(1) != 1 or target.size(1) != 1:
            raise ValueError("Input 'output' and 'target' should be either RGB (3 channels) or grayscale (1 channel)")

        # Gradient maps calculation
        gx_output, gy_output = self.calculate_gradient(output)
        gx_target, gy_target = self.calculate_gradient(target)

        # Unfolding and variance calculation
        eagle_loss = self.calculate_patch_loss(gx_output, gx_target) + \
                    self.calculate_patch_loss(gy_output, gy_target)

        return eagle_loss

    def calculate_gradient(self, img):
        # img = img.to(self.device)
        gx = F.conv2d(img, self.kernel_x, padding=1, groups=img.shape[1])
        gy = F.conv2d(img, self.kernel_y, padding=1, groups=img.shape[1])
        return gx, gy

    def calculate_patch_loss(self, output_gradient, target_gradient):
        # output_gradient, target_gradient = output_gradient.to(self.device), target_gradient.to(self.device)
        batch_size = output_gradient.size(0)
        num_channels = output_gradient.size(1)
        patch_size_squared = self.patch_size * self.patch_size
        output_patches = self.unfold(output_gradient).view(batch_size, num_channels, patch_size_squared, -1)
        target_patches = self.unfold(target_gradient).view(batch_size, num_channels, patch_size_squared, -1)
        var_output = torch.var(output_patches, dim=2, keepdim=True)
        var_target = torch.var(target_patches, dim=2, keepdim=True)

        shape0, shape1 = output_gradient.shape[-2] // self.patch_size, output_gradient.shape[-1] // self.patch_size
        return self.fft_loss(var_target.view(batch_size, num_channels, shape0, shape1), var_output.view(batch_size, num_channels, shape0, shape1))

    # def gaussian_highpass_weights2d(self, size):
    #     freq_x = torch.fft.fftfreq(size[0]).reshape(-1, 1).repeat(1, size[1]).to(self.device)
    #     freq_y = torch.fft.fftfreq(size[1]).reshape(1, -1).repeat(size[0], 1).to(self.device)

    #     freq_mag = torch.sqrt(freq_x ** 2 + freq_y ** 2)
    #     weights = torch.exp(-0.5 * ((freq_mag - self.cutoff) ** 2))
    #     return 1 - weights  # Inverted for high pass
    
    def gaussian_highpass_weights2d(self, size, device):
        h, w = size
        freq_x = torch.fft.fftfreq(h, device=device).reshape(-1, 1).repeat(1, w)
        freq_y = torch.fft.fftfreq(w, device=device).reshape(1, -1).repeat(h, 1)

        freq_mag = torch.sqrt(freq_x ** 2 + freq_y ** 2)
        weights = torch.exp(-0.5 * ((freq_mag - self.cutoff) ** 2))
        return 1.0 - weights  # high-pass

    def fft_loss(self, pred, gt):
        # pred, gt = pred.to(self.device), gt.to(self.device)
        # Ensure we are in float32 for FFT, even under AMP/bfloat16
        pred = pred.to(dtype=torch.float32)
        gt   = gt.to(dtype=torch.float32)
        device = pred.device
        # pred = pred.to(self.device, dtype=torch.float32)
        # gt   = gt.to(self.device, dtype=torch.float32)

        pred_fft = torch.fft.fft2(pred)
        gt_fft = torch.fft.fft2(gt)
        # Clamp FFT components to avoid overflow
        limit = 1e6
        pred_real = torch.clamp(pred_fft.real, min=-limit, max=limit)
        pred_imag = torch.clamp(pred_fft.imag, min=-limit, max=limit)
        gt_real   = torch.clamp(gt_fft.real,   min=-limit, max=limit)
        gt_imag   = torch.clamp(gt_fft.imag,   min=-limit, max=limit)
        # Magnitude with strong stabilization
        eps = 1e-12

        pred_mag = torch.sqrt(
            torch.clamp(pred_fft.real ** 2 + pred_fft.imag ** 2, min=eps)
        )
        gt_mag = torch.sqrt(
            torch.clamp(gt_fft.real ** 2 + gt_fft.imag ** 2, min=eps)
        )
        pred_mag = torch.sqrt(pred_fft.real ** 2 + pred_fft.imag ** 2)
        gt_mag = torch.sqrt(gt_fft.real ** 2 + gt_fft.imag ** 2)

        # weights = self.gaussian_highpass_weights2d(pred.size()[2:]).to(pred.device)
        weights = self.gaussian_highpass_weights2d(pred.size()[2:], device)
        weights = weights.to(device).unsqueeze(0).unsqueeze(0)
        weighted_pred_mag = weights * pred_mag
        weighted_gt_mag = weights * gt_mag

        return F.l1_loss(weighted_pred_mag, weighted_gt_mag)
    
class Eagle_Loss_3D(nn.Module):
    """
    Slice-wise 3D Eagle loss for volumes of shape (B, C, D, H, W).
    Uses the 2D Eagle_Loss on each axial slice and averages the result.
    """
    def __init__(self, patch_size, cpu=False, cutoff=0.5):
        super().__init__()
        self.eagle2d = Eagle_Loss(patch_size=patch_size, cpu=cpu, cutoff=cutoff)

    def forward(self, output, target):
        """
        output, target: (B, C, D, H, W)
        """
        if output.dim() != 5 or target.dim() != 5:
            raise ValueError("Eagle_Loss_3D expects (B, C, D, H, W) tensors")

        if output.shape != target.shape:
            raise ValueError("output and target must have the same shape")

        B, C, D, H, W = output.shape

        # (B, C, D, H, W) -> (B*D, C, H, W)
        output_2d = output.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W).float()
        target_2d = target.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W).float()

        loss = self.eagle2d(output_2d, target_2d)
        # your Eagle_Loss already returns a scalar mean over batch,
        # so this is effectively mean over all slices.
        return loss

def gradient_loss_3d(pred, target):
    """
    3D gradient loss for edge preservation
    pred, target: [B, C, D, H, W]
    """
    # Gradients in all 3 dimensions
    grad_d_pred = pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :]
    grad_h_pred = pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :]
    grad_w_pred = pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1]
    
    grad_d_target = target[:, :, 1:, :, :] - target[:, :, :-1, :, :]
    grad_h_target = target[:, :, :, 1:, :] - target[:, :, :, :-1, :]
    grad_w_target = target[:, :, :, :, 1:] - target[:, :, :, :, :-1]
    
    loss_d = F.l1_loss(grad_d_pred, grad_d_target)
    loss_h = F.l1_loss(grad_h_pred, grad_h_target)
    loss_w = F.l1_loss(grad_w_pred, grad_w_target)
    
    return loss_d + loss_h + loss_w

def focal_frequency_loss_3d(pred, target):
    """
    3D FFT-based loss for high-frequency details
    pred, target: [B, C, D, H, W]
    """
    # 3D FFT
    pred_fft = torch.fft.rfftn(pred, dim=(-3, -2, -1))
    target_fft = torch.fft.rfftn(target, dim=(-3, -2, -1))
    
    # Use magnitude
    pred_mag = torch.abs(pred_fft)
    target_mag = torch.abs(target_fft)
    
    return F.l1_loss(pred_mag, target_mag)