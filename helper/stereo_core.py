"""
Stereo 3D Core Functions
========================

Shared core functions for stereo image generation.
Used by both sbs_generator.py and sbs_tester.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from kornia.filters import gaussian_blur2d


__all__ = [
    'load_image_pair',
    'normalize_depth',
    'apply_depth_gamma',
    'forward_warp_stereo',
    'StereoParams',
    'StereoGenerator'
]


def load_image_pair(rgb_path: Path, depth_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load RGB and depth images from file paths.

    Parameters
    ----------
    rgb_path : Path
        Path to the RGB image file.
    depth_path : Path
        Path to the depth map file.

    Returns
    -------
    tuple of np.ndarray
        RGB image and depth map.

    Raises
    ------
    ValueError
        If images cannot be loaded or dimensions are invalid.
    """
    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)

    if rgb is None:
        raise ValueError(f'Could not load RGB: {rgb_path}')
    if depth is None:
        raise ValueError(f'Could not load depth: {depth_path}')

    if len(depth.shape) == 3:
        depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)

    if rgb.shape[:2] != depth.shape[:2]:
        depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LANCZOS4)

    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    return rgb, depth


def normalize_depth(depth: torch.Tensor) -> torch.Tensor:
    """
    Normalize depth to [0, 1] range.

    Parameters
    ----------
    depth : torch.Tensor
        Depth tensor of any shape.

    Returns
    -------
    torch.Tensor
        Normalized depth tensor in range [0, 1].
    """
    d_min, d_max = depth.min(), depth.max()
    if d_max - d_min < 1e-6:
        return torch.zeros_like(depth)
    return (depth - d_min) / (d_max - d_min)


def apply_depth_gamma(depth: torch.Tensor, gamma: float) -> torch.Tensor:
    """
    Apply gamma correction to depth.

    Parameters
    ----------
    depth : torch.Tensor
        Normalized depth tensor in range [0, 1].
    gamma : float
        Gamma value for correction.

    Returns
    -------
    torch.Tensor
        Gamma-corrected depth tensor.
    """
    return torch.pow(depth.clamp(0.001, 1.0), gamma)


def forward_warp_stereo(image: torch.Tensor, depth: torch.Tensor, max_disparity: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate both left and right warped views in a single pass.

    Parameters
    ----------
    image : torch.Tensor
        Input image tensor of shape [B, C, H, W].
    depth : torch.Tensor
        Normalized depth tensor of shape [B, 1, H, W].
    max_disparity : float
        Maximum disparity in pixels.

    Returns
    -------
    tuple of torch.Tensor
        Left warped, left mask, right warped, and right mask tensors.
    """
    B, C, H, W = image.shape
    device = image.device

    src_y, src_x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')

    disp = depth.squeeze() * max_disparity

    src_y_flat = src_y.flatten()
    src_x_flat = src_x.float().flatten()
    disp_flat = disp.flatten()
    depth_flat = depth.squeeze().flatten()

    sort_order = torch.argsort(depth_flat)
    src_y_sorted = src_y_flat[sort_order]
    src_x_sorted = src_x_flat[sort_order]
    disp_sorted = disp_flat[sort_order]
    src_idx_sorted = sort_order

    image_flat = image.view(C, -1)

    def _warp_single_direction(disp_signed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Warp image in one direction with bilinear splatting."""
        tgt_x_float = src_x_sorted + disp_signed
        tgt_x_floor = tgt_x_float.floor().long()
        frac_x = tgt_x_float - tgt_x_floor.float()

        warped = torch.zeros_like(image)
        warped_flat = warped.view(C, -1)
        valid_mask = torch.zeros(H * W, device=device)

        valid_floor = (tgt_x_floor >= 0) & (tgt_x_floor < W)
        tgt_idx_floor = src_y_sorted * W + tgt_x_floor

        tgt_idx_floor_valid = tgt_idx_floor[valid_floor]
        src_idx_floor_valid = src_idx_sorted[valid_floor]
        weight_floor_valid = (1.0 - frac_x)[valid_floor]

        for c in range(C):
            warped_flat[c].scatter_(0, tgt_idx_floor_valid, image_flat[c, src_idx_floor_valid])
        valid_mask.scatter_(0, tgt_idx_floor_valid, weight_floor_valid)

        tgt_x_ceil = tgt_x_floor + 1
        valid_ceil = (tgt_x_ceil >= 0) & (tgt_x_ceil < W)
        tgt_idx_ceil = src_y_sorted * W + tgt_x_ceil

        tgt_idx_ceil_valid = tgt_idx_ceil[valid_ceil]
        src_idx_ceil_valid = src_idx_sorted[valid_ceil]
        weight_ceil_valid = frac_x[valid_ceil]

        significant = weight_ceil_valid > 0.3
        tgt_idx_sig = tgt_idx_ceil_valid[significant]
        src_idx_sig = src_idx_ceil_valid[significant]

        for c in range(C):
            warped_flat[c].scatter_(0, tgt_idx_sig, image_flat[c, src_idx_sig])
        valid_mask.scatter_(0, tgt_idx_sig, weight_ceil_valid[significant])

        return warped_flat.view(B, C, H, W), (valid_mask > 0.1).float().view(B, 1, H, W)

    left_warped, left_mask = _warp_single_direction(disp_sorted)
    right_warped, right_mask = _warp_single_direction(-disp_sorted)

    return left_warped, left_mask, right_warped, right_mask


@dataclass
class StereoParams:
    """Parameters for stereo image generation."""
    max_disparity: float = 50.0
    convergence: float = -10.0
    super_sampling: float = 3.0
    edge_softness: float = 20.0
    artifact_smoothing: float = 1.0
    depth_gamma: float = 0.2
    sharpen: float = 14.0


class StereoGenerator:
    """
    Efficient batch processor for stereo SBS image generation.

    Supports both batch mode (parameters set at init) and interactive mode
    (parameters passed per frame).
    """
    _DEFAULT_PARAMS = StereoParams()

    def __init__(self, device: str) -> None:
        """
        Initialize stereo generator.

        Parameters
        ----------
        device : str
            PyTorch device string ('cuda' or 'cpu').
        """
        self.device = device

    def process_frame(self, rgb: np.ndarray, depth: np.ndarray, params: StereoParams | None = None) -> np.ndarray:
        """
        Process single frame to generate SBS stereo image.

        Parameters
        ----------
        rgb : np.ndarray
            RGB image as numpy array.
        depth : np.ndarray
            Depth map as numpy array.
        params : StereoParams, optional
            Stereo generation parameters (overrides instance defaults).

        Returns
        -------
        np.ndarray
            SBS image as numpy array (left | right).
        """
        p = params or self._DEFAULT_PARAMS
        original_h, original_w = rgb.shape[:2]

        # Pre-stretch image to account for:
        # 1. Disparity shift: Both views need max_disparity buffer on each side
        # 2. Convergence shift: Additional buffer to maintain output width after convergence crop
        total_buffer = 2.0 * p.max_disparity + abs(p.convergence)
        stretch_factor = 1.0 + (total_buffer / original_w)
        stretched_w = int(original_w * stretch_factor)

        rgb_stretched = cv2.resize(rgb, (stretched_w, original_h), interpolation=cv2.INTER_LANCZOS4)
        depth_stretched = cv2.resize(depth, (stretched_w, original_h), interpolation=cv2.INTER_LANCZOS4)

        rgb_t = self._to_torch(rgb_stretched)
        depth_t = self._to_torch(depth_stretched)
        depth_norm = normalize_depth(depth_t)

        if p.super_sampling > 1.0:
            depth_norm = self._depth_upsampling(depth_norm, p.super_sampling)
            rgb_t = F.interpolate(rgb_t, size=depth_norm.shape[2:], mode='bilinear', align_corners=False)

        if p.edge_softness > 0:
            depth_norm = self._soft_depth_edges(depth_norm, p.edge_softness)

        if p.depth_gamma != 1.0:
            depth_norm = apply_depth_gamma(depth_norm, p.depth_gamma)

        left_warped, left_mask, right_warped, right_mask = forward_warp_stereo(rgb_t, depth_norm, p.max_disparity)

        left = self._postprocess_view(left_warped, left_mask, p.artifact_smoothing)
        right = self._postprocess_view(right_warped, right_mask, p.artifact_smoothing)

        base_crop_offset = (stretched_w - original_w) // 2
        convergence_shift = int(round(p.convergence))

        # Convergence shifts crop windows in opposite directions:
        # Positive convergence: left crops more from right, right crops more from left (objects pop out)
        # Negative convergence: left crops more from left, right crops more from right (objects recede)
        left_crop_offset = base_crop_offset + convergence_shift
        right_crop_offset = base_crop_offset - convergence_shift

        if p.super_sampling > 1.0:
            upscaled_w = left.shape[3]
            scale_ratio = upscaled_w / stretched_w
            left_crop_upscaled = int(left_crop_offset * scale_ratio)
            right_crop_upscaled = int(right_crop_offset * scale_ratio)
            original_w_upscaled = int(original_w * scale_ratio)

            left = left[:, :, :, left_crop_upscaled : left_crop_upscaled + original_w_upscaled]
            right = right[:, :, :, right_crop_upscaled : right_crop_upscaled + original_w_upscaled]

            if p.sharpen > 0:
                left = self._sharpen_image(left, p.sharpen)
                right = self._sharpen_image(right, p.sharpen)

            left = F.interpolate(left, size=(original_h, original_w), mode='area')
            right = F.interpolate(right, size=(original_h, original_w), mode='area')
        else:
            left = left[:, :, :, left_crop_offset : left_crop_offset + original_w]
            right = right[:, :, :, right_crop_offset : right_crop_offset + original_w]

            if p.sharpen > 0:
                left = self._sharpen_image(left, p.sharpen)
                right = self._sharpen_image(right, p.sharpen)

        left_np = self._to_numpy_uint8(left)
        right_np = self._to_numpy_uint8(right)

        return np.hstack([left_np, right_np])

    def _to_torch(self, image: np.ndarray) -> torch.Tensor:
        """
        Convert numpy array to torch tensor [B, C, H, W].

        Parameters
        ----------
        image : np.ndarray
            Input image as numpy array.

        Returns
        -------
        torch.Tensor
            Image tensor in [B, C, H, W] format.
        """
        if len(image.shape) == 2:
            return torch.from_numpy(image.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(self.device, non_blocking=True)

        return torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).to(self.device, non_blocking=True)

    def _to_numpy_uint8(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Convert torch tensor to uint8 numpy array.

        Parameters
        ----------
        tensor : torch.Tensor
            Input tensor.

        Returns
        -------
        np.ndarray
            Image as uint8 numpy array.
        """
        return tensor.squeeze(0).permute(1, 2, 0).clamp(0, 255).cpu().numpy().astype(np.uint8)

    def _depth_upsampling(self, depth: torch.Tensor, scale_factor: float) -> torch.Tensor:
        """
        Upsample depth map using GPU bilinear interpolation.

        Parameters
        ----------
        depth : torch.Tensor
            Input depth tensor.
        scale_factor : float
            Upsampling scale factor.

        Returns
        -------
        torch.Tensor
            Upsampled depth tensor.
        """
        new_H = int(depth.shape[2] * scale_factor)
        new_W = int(depth.shape[3] * scale_factor)
        return F.interpolate(depth, size=(new_H, new_W), mode='bilinear', align_corners=False)

    def _soft_depth_edges(self, depth: torch.Tensor, edge_softness: float) -> torch.Tensor:
        """
        Apply Gaussian blur to soften depth edges.

        Parameters
        ----------
        depth : torch.Tensor
            Input depth tensor.
        edge_softness : float
            Gaussian blur sigma value.

        Returns
        -------
        torch.Tensor
            Blurred depth tensor.
        """
        blur_kernel = max(5, min(int(edge_softness * 6) | 1, 31))
        return gaussian_blur2d(depth, (blur_kernel, blur_kernel), (edge_softness, edge_softness))

    def _smooth_warping_artifacts(self, image: torch.Tensor, artifact_smoothing: float) -> torch.Tensor:
        """
        Smooth warping artifacts using fast OpenCV bilateral filter.

        Parameters
        ----------
        image : torch.Tensor
            Input image tensor.
        artifact_smoothing : float
            Strength of smoothing.

        Returns
        -------
        torch.Tensor
            Smoothed image tensor.
        """
        image_np = image.squeeze().permute(1, 2, 0).cpu().numpy()
        if image_np.max() > 1.0:
            image_np = image_np.astype(np.uint8)
        else:
            image_np = (image_np * 255).astype(np.uint8)

        d = max(5, min(int(artifact_smoothing * 4), 15))
        filtered_np = cv2.bilateralFilter(image_np, d=d, sigmaColor=30, sigmaSpace=artifact_smoothing * 25)
        filtered = torch.from_numpy(filtered_np).permute(2, 0, 1).unsqueeze(0).float()
        return filtered.to(image.device)

    def _sharpen_image(self, image: torch.Tensor, strength: float) -> torch.Tensor:
        """
        GPU-accelerated unsharp mask sharpening.

        Parameters
        ----------
        image : torch.Tensor
            Input image tensor.
        strength : float
            Sharpening strength.

        Returns
        -------
        torch.Tensor
            Sharpened image tensor.
        """
        kernel_size = 5
        sigma = 1.0
        blurred = gaussian_blur2d(image, (kernel_size, kernel_size), (sigma, sigma))
        sharpened = image + strength * (image - blurred)
        return sharpened.clamp(0, 255)

    def _inpaint_missing_regions(self, image_np: np.ndarray, inpaint_mask_np: np.ndarray) -> np.ndarray:
        """
        Inpaint missing regions in an image using OpenCV.

        Parameters
        ----------
        image_np : np.ndarray
            Input image.
        inpaint_mask_np : np.ndarray
            Binary mask indicating regions to inpaint.

        Returns
        -------
        np.ndarray
            Inpainted image.
        """
        if not inpaint_mask_np.any():
            return image_np

        kernel = np.ones((3, 3), np.uint8)
        inpaint_mask_np = cv2.dilate(inpaint_mask_np, kernel, iterations=1)
        return cv2.inpaint(image_np, inpaint_mask_np, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    def _postprocess_view(self, warped: torch.Tensor, valid_mask: torch.Tensor, artifact_smoothing: float) -> torch.Tensor:
        """
        Apply smoothing and inpainting to a warped view.

        Parameters
        ----------
        warped : torch.Tensor
            Warped image tensor.
        valid_mask : torch.Tensor
            Mask of valid pixels.
        artifact_smoothing : float
            Strength of smoothing.

        Returns
        -------
        torch.Tensor
            Post-processed image tensor.
        """
        inpaint_mask_np = ((1 - valid_mask.squeeze(0)) * 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)

        if artifact_smoothing > 0:
            warped = self._smooth_warping_artifacts(warped, artifact_smoothing)

        result_np = self._to_numpy_uint8(warped)
        result_np = self._inpaint_missing_regions(result_np, inpaint_mask_np)

        return self._to_torch(result_np)
    