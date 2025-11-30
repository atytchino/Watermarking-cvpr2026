"""
Universal AE (Y-core) with external watermark injection surfaces.

This module implements a universal autoencoder designed for joint pretraining
on both RGB and grayscale imagery. The core architecture is Y-channel centric
and exposes well-defined intermediate "surfaces" (latent and skip features)
for externally controlled watermark injection.

Key characteristics:
- Pretraining on up to three datasets: AFHQ (RGB), MRI (Gray), ORNL (Gray)
- Balanced batches 1:1 (color vs gray) via a custom batch sampler
- External watermark API with full-channel access to:
    * latent:  [B,1024,32,32]
    * skip64:  [B, 512,64,64]
- Two convenience entrypoints:
    * embed_external_wm_rgb  – for RGB inputs
    * embed_external_wm_gray – for grayscale inputs
- Epoch-wise checkpoints plus "best" model tracking
- Split PSNR/SSIM logs for:
    * Y (all images)
    * Y (color subset)
    * Y (grayscale subset)
    * RGB (color subset)

"""

from __future__ import annotations
import os, sys, time, math, csv, argparse, random
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

# =========================
# -------- PATHS FOR DATASETS (UP TO 3) ----------
# =========================

# Example commented-out dataset roots:
# TLD_TRAIN = r"E:\TLD\Train"
# TLD_VAL   = r"E:\TLD\Val"
#
# AFHQ_TRAIN= r"E:\AFHQ_LOWMEM\train"
# AFHQ_VAL  = r"E:\AFHQ_LOWMEM\val"

ORNL_TRAIN = r"E:\ORNL_GRAYSCALE\Train"
ORNL_VAL   = r"E:\ORNL_GRAYSCALE\Val"

# TRAIN_ROOT_PAIRS is a fixed-size container for up to three (train,val) pairs.
# Non-existing pairs can be represented as (None, None) placeholders.
TRAIN_ROOT_PAIRS = [(ORNL_TRAIN, ORNL_VAL), (None, None), (None, None)]

SAVE_ROOT = Path(r"E:\Universal_ae_ver2_ORNL")
SAVE_ROOT.mkdir(parents=True, exist_ok=True)
CKPT_DIR = SAVE_ROOT
LOG_DIR  = SAVE_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# -------- UTILS ----------
# =========================

def seed_all(seed: int = 123456789) -> None:
    """
    Seed all relevant random number generators to make experiments
    as reproducible as reasonably possible.

    The function touches:
    - Python's built-in `random`
    - NumPy RNG
    - PyTorch CPU RNG
    - PyTorch CUDA RNG (all devices)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_luma_Y(x: torch.Tensor) -> torch.Tensor:
    """
    Convert an RGB or single-channel input to luminance Y.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape [B, C, H, W] with values in [0, 1].
        If C == 3, the tensor is interpreted as RGB and converted to Y
        via the standard ITU-R BT.601 coefficients:
            Y = 0.299 R + 0.587 G + 0.114 B
        If C == 1, the tensor is already a luma/gray channel and is
        returned as-is.

    Returns
    -------
    torch.Tensor
        Tensor of shape [B, 1, H, W] containing luma Y in [0, 1].
    """
    if x.size(1) == 1:
        return x
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return y


def rgb_to_ycbcr(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert RGB image(s) in [0, 1] to YCbCr color space.

    Parameters
    ----------
    x : torch.Tensor
        Tensor of shape [B, 3, H, W] with values in [0, 1].

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        (Y, Cb, Cr) each of shape [B, 1, H, W], clamped to [0, 1].
    """
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    y  = 0.299*r + 0.587*g + 0.114*b
    cb = (-0.168736)*r - 0.331264*g + 0.5*b + 0.5
    cr = 0.5*r - 0.418688*g - 0.081312*b + 0.5
    return y.clamp(0, 1), cb.clamp(0, 1), cr.clamp(0, 1)


def ycbcr_to_rgb(y: torch.Tensor, cb: torch.Tensor, cr: torch.Tensor) -> torch.Tensor:
    """
    Convert YCbCr channels back to RGB in [0, 1].

    Parameters
    ----------
    y, cb, cr : torch.Tensor
        Tensors of shape [B, 1, H, W], representing luminance and
        chroma channels in the normalized YCbCr space.

    Returns
    -------
    torch.Tensor
        Reconstructed RGB tensor of shape [B, 3, H, W], clamped to [0, 1].
    """
    cb_c = cb - 0.5
    cr_c = cr - 0.5
    r = y + 1.402*cr_c
    g = y - 0.344136*cb_c - 0.714136*cr_c
    b = y + 1.772*cb_c
    return torch.cat([r, g, b], dim=1).clamp(0, 1)


# ---- PSNR/SSIM per-image ----

def psnr_tensor(x: torch.Tensor, y: torch.Tensor, data_range: float = 1.0, eps: float = 1e-12) -> torch.Tensor:
    """
    Compute the per-image Peak Signal-to-Noise Ratio (PSNR).

    Parameters
    ----------
    x, y : torch.Tensor
        Tensors of shape [B, C, H, W]. All channels are treated equally.
    data_range : float
        Dynamic range of the input data. For images in [0, 1] this is 1.0.
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    torch.Tensor
        PSNR values for each image, shape [B].
    """
    mse = (x - y).pow(2).flatten(1).mean(dim=1).clamp_min(eps)
    return 10.0 * torch.log10((data_range ** 2) / mse)


def _gaussian_window(window_size: int, sigma: float, device, dtype):
    """
    Construct a 2D Gaussian window used as a convolutional kernel
    for SSIM computation.

    The kernel is separable and is formed as the outer product of
    two 1D Gaussians.
    """
    coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).unsqueeze(0)  # [1,W]
    window_2d = (g.t() @ g)         # [W,W]
    return window_2d


def ssim_tensor(
    x: torch.Tensor,
    y: torch.Tensor,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
    eps: float = 1e-12
) -> torch.Tensor:
    """
    Compute the per-image Structural Similarity Index (SSIM).

    The implementation follows the standard window-based SSIM formulation
    with a Gaussian window applied independently to each channel.

    Parameters
    ----------
    x, y : torch.Tensor
        Input tensors of shape [B, C, H, W] with the same shape.
        C can be 1 or 3; SSIM is averaged over channels.
    data_range : float
        Dynamic range of the input data.
    window_size : int
        Spatial size of the Gaussian window (odd).
    sigma : float
        Standard deviation of the Gaussian window.
    eps : float
        Numerical stability constant.

    Returns
    -------
    torch.Tensor
        SSIM values for each image, shape [B].
    """
    assert x.shape == y.shape and x.dim() == 4
    B, C, H, W = x.shape
    device, dtype = x.device, x.dtype

    w = _gaussian_window(window_size, sigma, device, dtype)
    w = w.expand(C, 1, window_size, window_size).contiguous()
    padding = window_size // 2

    mu_x = F.conv2d(x, w, padding=padding, groups=C)
    mu_y = F.conv2d(y, w, padding=padding, groups=C)

    mu_x2 = mu_x.pow(2)
    mu_y2 = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, w, padding=padding, groups=C) - mu_x2
    sigma_y2 = F.conv2d(y * y, w, padding=padding, groups=C) - mu_y2
    sigma_xy = F.conv2d(x * y, w, padding=padding, groups=C) - mu_xy

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / (
        (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2) + eps
    )
    ssim_per_img = ssim_map.clamp(0, 1).flatten(2).mean(dim=2).mean(dim=1)
    return ssim_per_img


# ---- Meters & split logging ----

from dataclasses import dataclass


@dataclass
class AvgMeter:
    """
    Simple streaming average meter for scalar quantities.

    Used to accumulate PSNR, SSIM and loss values over an epoch without
    storing the entire history in memory.
    """
    sum: float = 0.0
    n: int = 0

    def update(self, val, k: int = 1):
        """
        Update the running sum and count.

        Parameters
        ----------
        val : float or scalar tensor
            Value to accumulate.
        k : int
            Effective batch size (number of samples represented by `val`).
        """
        if val is None:
            return
        self.sum += float(val) * k
        self.n += k

    @property
    def avg(self) -> float:
        """Return the average value; returns 0 if no samples were accumulated."""
        return self.sum / max(1, self.n)


class SplitMeters:
    """
    Bundle of average meters that tracks reconstruction quality
    across several splits:

    - psnr_y_all,  ssim_y_all:   Y-channel metrics over all samples
    - psnr_y_color, ssim_y_color: Y-channel metrics over color-only samples
    - psnr_y_gray,  ssim_y_gray:  Y-channel metrics over grayscale-only samples
    - psnr_rgb_color, ssim_rgb_color: RGB metrics over color-only samples

    This decomposition is useful when pretraining on a mixed
    RGB/grayscale dataset and we want to monitor performance
    per modality.
    """
    def __init__(self):
        self.psnr_y_all = AvgMeter()
        self.ssim_y_all = AvgMeter()
        self.psnr_y_color = AvgMeter()
        self.ssim_y_color = AvgMeter()
        self.psnr_y_gray = AvgMeter()
        self.ssim_y_gray = AvgMeter()
        self.psnr_rgb_color = AvgMeter()
        self.ssim_rgb_color = AvgMeter()

    def log_line(self, ep: int, it: int, loss_val: float) -> str:
        """
        Format a human-readable summary line for logging during training.
        """
        return (
            f"[E{ep:02d} it {it:05d}] loss={loss_val:.4f} | "
            f"Y: psnr={self.psnr_y_all.avg:.2f} ssim={self.ssim_y_all.avg:.3f} | "
            f"Ycolor={self.psnr_y_color.avg:.2f}/{self.ssim_y_color.avg:.3f} "
            f"Ygray={self.psnr_y_gray.avg:.2f}/{self.ssim_y_gray.avg:.3f} | "
            f"RGBcolor={self.psnr_rgb_color.avg:.2f}/{self.ssim_rgb_color.avg:.3f}"
        )


@torch.no_grad()
def update_split_metrics(
    meters: SplitMeters,
    x: torch.Tensor,
    y_hat: torch.Tensor,
    is_color_mask: torch.Tensor
) -> None:
    """
    Update split PSNR/SSIM metrics given a batch of reconstructions.

    Parameters
    ----------
    meters : SplitMeters
        Container with running averages for all metric splits.
    x : torch.Tensor
        Ground-truth images, shape [B, C, H, W], C ∈ {1, 3}, in [0, 1].
    y_hat : torch.Tensor
        Reconstructed images, same shape as `x`.
    is_color_mask : torch.Tensor
        Boolean tensor of shape [B]; True for color samples, False for gray.

    Notes
    -----
    Metrics are always computed in float32 with AMP disabled, to avoid
    accidental type conflicts across mixed-precision code paths.
    """
    # For safety, explicitly disable autocast and convert to float32.
    with torch.amp.autocast(device_type="cuda", enabled=False):
        x_f = x.float()
        y_f = y_hat.float()

        B = x_f.size(0)
        y_x, yhat_x = to_luma_Y(x_f), to_luma_Y(y_f)   # [B,1,H,W]
        psnr_y = psnr_tensor(y_x, yhat_x)              # [B]
        ssim_y = ssim_tensor(y_x, yhat_x)              # [B]

        meters.psnr_y_all.update(psnr_y.mean().item(), k=B)
        meters.ssim_y_all.update(ssim_y.mean().item(), k=B)

        if is_color_mask.any():
            idx = is_color_mask
            meters.psnr_y_color.update(psnr_y[idx].mean().item(), k=int(idx.sum()))
            meters.ssim_y_color.update(ssim_y[idx].mean().item(), k=int(idx.sum()))
            # For color images, we additionally compute PSNR/SSIM in RGB space.
            psnr_rgb = psnr_tensor(x_f[idx], y_f[idx])
            ssim_rgb = ssim_tensor(x_f[idx], y_f[idx])
            meters.psnr_rgb_color.update(psnr_rgb.mean().item(), k=int(idx.sum()))
            meters.ssim_rgb_color.update(ssim_rgb.mean().item(), k=int(idx.sum()))

        if (~is_color_mask).any():
            idx = ~is_color_mask
            meters.psnr_y_gray.update(psnr_y[idx].mean().item(), k=int(idx.sum()))
            meters.ssim_y_gray.update(ssim_y[idx].mean().item(), k=int(idx.sum()))


# =========================
# -------- DATA -----------
# =========================

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(roots: List[str]) -> List[str]:
    """
    Recursively list all image files under the given root directories.

    Parameters
    ----------
    roots : list of str
        Root directories to traverse.

    Returns
    -------
    List[str]
        Sorted list of absolute file paths whose extensions are in IMG_EXTS.
    """
    files = []
    for root in roots:
        if not root or not Path(root).exists():
            print(f"[WARN] dataset path not found: {root}")
            continue
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if os.path.splitext(fn)[1].lower() in IMG_EXTS:
                    files.append(os.path.join(dirpath, fn))
    return files


def detect_dataset_mode(
    root: str,
    max_sample: int = 100,
    name: Optional[str] = None
) -> str:
    """
    Determine the predominant dataset mode ("rgb" or "gray") by sampling
    a subset of images from the given root.

    The procedure:
    - randomly sample up to `max_sample` images from `root`;
    - classify each image as color or grayscale using `detect_is_color_pil`;
    - if the minority type exceeds a small threshold, treat the dataset
      as inconsistent and raise an error.

    Parameters
    ----------
    root : str
        Root directory of the dataset.
    max_sample : int
        Maximum number of images to sample for inspection.
    name : str, optional
        Human-readable dataset name for logging.

    Returns
    -------
    str
        Either "rgb" or "gray".

    Raises
    ------
    RuntimeError
        If the dataset appears to be strongly mixed (color and grayscale).
    """
    files = list_images([root])
    if not files:
        raise RuntimeError(f"No images found under dataset root: {root}")

    k = min(max_sample, len(files))
    sample_paths = random.sample(files, k)

    n_color = 0
    n_gray  = 0
    for p in sample_paths:
        try:
            with Image.open(p) as im:
                is_color = detect_is_color_pil(im)
        except Exception:
            # If the image is corrupted, treat it as color to avoid biasing
            # the statistics toward grayscale.
            is_color = True
        if is_color:
            n_color += 1
        else:
            n_gray += 1

    mismatches = min(n_color, n_gray)
    mode = "rgb" if n_color >= n_gray else "gray"

    ds_name = name or os.path.basename(os.path.normpath(root)) or root
    print(
        f"[DATASET CHECK] {ds_name}: mode={mode.upper()}, "
        f"n_color={n_color}, n_gray={n_gray}, sampled={k}, mismatches={mismatches}"
    )

    if mismatches > 5:
        raise RuntimeError(
            f"Dataset '{ds_name}' under {root} is not homogeneous enough: "
            f"{n_color} color, {n_gray} gray in {k} samples (mismatches={mismatches}). "
            f"Please clean up the dataset (e.g., separate RGB and GRAY into different roots)."
        )
    if mismatches >= 1:
        print(
            f"[WARN] Dataset '{ds_name}' has some mixed images "
            f"(mismatches={mismatches} <= 5). Continuing, but this looks suspicious."
        )

    return mode


def detect_is_color_pil(img: Image.Image, nearly_gray_thresh: int = 2) -> bool:
    """
    Heuristic classification of a PIL image as "color" vs "grayscale".

    The logic is:
    - If PIL mode is single-channel (L, I;16, I, F, 1) -> grayscale.
    - Otherwise, convert to RGB and check the maximum absolute difference
      between channels; if it does not exceed `nearly_gray_thresh`, we
      treat the image as nearly grayscale, otherwise as color.
    """
    if img.mode in ("L", "I;16", "I", "F", "1"):
        return False
    rgb = img.convert("RGB")
    arr = np.asarray(rgb, dtype=np.int16)
    diff_rg = np.abs(arr[..., 0] - arr[..., 1])
    diff_rb = np.abs(arr[..., 0] - arr[..., 2])
    diff_gb = np.abs(arr[..., 1] - arr[..., 2])
    maxdiff = int(max(diff_rg.max(initial=0), diff_rb.max(initial=0), diff_gb.max(initial=0)))
    return maxdiff > nearly_gray_thresh


class MixedImageDataset(Dataset):
    """
    Unified dataset wrapper operating over multiple root directories.

    Behaviour
    ---------
    - For each image we optionally precompute and cache a boolean flag
      `is_color` (True for color-like images, False for nearly grayscale),
      using `detect_is_color_pil`.
    - In `__getitem__`:
        * if is_color == True  -> return a [3, H, W] RGB tensor
        * if is_color == False -> return a [1, H, W] single-channel tensor
    - Mixing RGB and grayscale images under the same roots is allowed;
      the effective AE mode is decided later by the trainer.

    This design allows us to retain maximal flexibility in how we construct
    balanced RGB/gray batches while keeping the dataset itself simple.
    """
    def __init__(self, roots: List[str], size: int = 512, scan_flags: bool = True):
        super().__init__()
        self.paths = list_images(roots)
        if not self.paths:
            raise RuntimeError("No images found under given roots.")
        self.size = int(size)
        self.scan_flags = bool(scan_flags)

        self._flags: Optional[List[bool]] = None   # True = color, False = gray

        if self.scan_flags:
            flags: List[bool] = []
            for p in self.paths:
                try:
                    with Image.open(p) as im:
                        flags.append(detect_is_color_pil(im))
                except Exception:
                    # If the image cannot be opened, classify it as color to avoid
                    # overestimating the grayscale portion of the dataset.
                    flags.append(True)
            self._flags = flags

    def __len__(self) -> int:
        return len(self.paths)

    def is_color_flags(self) -> List[bool]:
        """
        Return the list of precomputed color/grayscale flags.

        If the flags have not been scanned yet and `scan_flags=True`,
        the dataset will lazily inspect all images on the first call.
        """
        if self._flags is not None:
            return self._flags

        flags: List[bool] = []
        for p in self.paths:
            try:
                with Image.open(p) as im:
                    flags.append(detect_is_color_pil(im))
            except Exception:
                flags.append(True)

        self._flags = flags
        return flags

    def __getitem__(self, idx: int):
        p = self.paths[idx]

        # Determine whether the current image is treated as color or grayscale.
        if self._flags is not None:
            is_color = self._flags[idx]
        else:
            with Image.open(p) as im_tmp:
                is_color = detect_is_color_pil(im_tmp) if self.scan_flags else True

        with Image.open(p) as im:
            if is_color:
                # 3-channel RGB input
                img = im.convert("RGB").resize((self.size, self.size), Image.BICUBIC)
                np_img = np.array(img, dtype=np.uint8, copy=True)      # [H,W,3]
                x = (
                    torch.from_numpy(np_img)
                    .permute(2, 0, 1)
                    .to(torch.float32)
                    .div_(255.0)
                )  # [3,H,W]
            else:
                # 1-channel grayscale input
                img = im.convert("L").resize((self.size, self.size), Image.BICUBIC)
                np_img = np.array(img, dtype=np.uint8, copy=True)      # [H,W]
                x = (
                    torch.from_numpy(np_img)
                    .unsqueeze(0)
                    .to(torch.float32)
                    .div_(255.0)
                )  # [1,H,W]

        return x, is_color, p


class CollateWithFlags:
    """
    Pickle-friendly collate function wrapper.

    The collate function:
    - accepts a batch which is a list of triples (x, is_color, path),
      where `x` has shape [C, H, W] with C ∈ {1, 3};
    - converts all `x` tensors to `expected_c` channels (1 or 3), emitting
      warnings when conversion is performed;
    - returns:
        xs    : [B, expected_c, H, W]
        flags : [B] bool (original `is_color` flags)
        paths : list of strings (image paths)
    """
    def __init__(self, expected_c: int, ae_mode: str):
        assert expected_c in (1, 3)
        self.expected_c = expected_c
        self.ae_mode = ae_mode

    def __call__(self, batch):
        xs = []
        flags = []
        paths = []
        ec = self.expected_c
        mode = self.ae_mode

        for x, is_color, p in batch:
            C, H, W = x.shape
            if C == ec:
                x_conv = x
            else:
                # Channel conversion is required.
                if ec == 3 and C == 1:
                    # grayscale -> RGB
                    x_conv = x.repeat(3, 1, 1)
                    print(f"[WARN] converting GRAY -> RGB for {p} (AE mode={mode})")
                elif ec == 1 and C == 3:
                    # RGB -> grayscale (Y) using the same formula as in to_luma_Y
                    r, g, b = x[0:1], x[1:2], x[2:3]
                    y = 0.299 * r + 0.587 * g + 0.114 * b
                    x_conv = y
                    print(f"[WARN] converting RGB -> GRAY for {p} (AE mode={mode})")
                else:
                    raise RuntimeError(f"Unexpected channels {C} for sample {p}, expected {ec}.")

            xs.append(x_conv)
            flags.append(bool(is_color))
            paths.append(p)

        xs = torch.stack(xs, dim=0)                     # [B, expected_c, H, W]
        flags_t = torch.tensor(flags, dtype=torch.bool) # [B]
        return xs, flags_t, paths


def make_collate_fn(expected_c: int, ae_mode: str):
    """
    Thin wrapper kept for compatibility with earlier versions of the code.

    Returns
    -------
    CollateWithFlags
        A pickle-friendly collate function object.
    """
    return CollateWithFlags(expected_c, ae_mode)


class AlternatingBatchSampler(Sampler):
    """
    Custom batch sampler that attempts to produce 1:1 batches of color vs
    grayscale samples, falling back to whatever is available when one side
    runs out.

    This is used to enforce balanced exposure of RGB and grayscale images
    during pretraining, which empirically stabilizes optimization.
    """
    def __init__(self, indices_color, indices_gray, batch_size,
                 drop_last=True, shuffle=True):
        assert batch_size >= 2 and batch_size % 2 == 0, "batch_size must be even >=2"
        self.ic = list(indices_color)
        self.ig = list(indices_gray)
        self.bs = int(batch_size)
        self.half = self.bs // 2
        self.drop_last = bool(drop_last)
        self.shuffle = bool(shuffle)

    def __iter__(self):
        ic = self.ic.copy()
        ig = self.ig.copy()
        if self.shuffle:
            random.shuffle(ic)
            random.shuffle(ig)
        pc = pg = 0
        total = len(ic) + len(ig)
        n_batches = total // self.bs if self.drop_last else math.ceil(total / self.bs)
        for _ in range(n_batches):
            batch = []
            # First half of the batch: preferentially color indices.
            for _ in range(self.half):
                if pc < len(ic):
                    batch.append(ic[pc]); pc += 1
                elif pg < len(ig):
                    batch.append(ig[pg]); pg += 1
            # Second half of the batch: preferentially gray indices.
            for _ in range(self.half):
                if pg < len(ig):
                    batch.append(ig[pg]); pg += 1
                elif pc < len(ic):
                    batch.append(ic[pc]); pc += 1
            if len(batch) < self.bs and self.drop_last:
                break
            yield batch

    def __len__(self):
        total = len(self.ic) + len(self.ig)
        return total // self.bs if self.drop_last else math.ceil(total / self.bs)


def decide_ae_mode_from_dataset(
    train_ds: MixedImageDataset,
    max_sample: int = 100
) -> Tuple[str, int]:
    """
    Infer the global AE operating mode ("rgb" vs "gray") from the dataset.

    The decision is based on the binary `is_color` flags returned by
    `train_ds.is_color_flags()`.

    Returns
    -------
    Tuple[str, int]
        ae_mode    : "rgb" or "gray"
        expected_c : 3 or 1 (number of channels expected by the AE)

    Logic
    -----
    - sample up to `max_sample` random indices from the dataset;
    - if both color and gray examples are present:
        * define mismatches = min(count_color, count_gray) as the minority;
        * if mismatches >= 1 -> emit a warning;
        * if mismatches > 5  -> raise RuntimeError;
    - choose the operating mode by simple majority:
        if color >= gray -> "rgb", otherwise "gray".
    """
    flags = train_ds.is_color_flags()
    n_total = len(flags)
    if n_total == 0:
        raise RuntimeError("Empty training dataset.")

    k = min(max_sample, n_total)
    indices = random.sample(range(n_total), k)
    sample = [bool(flags[i]) for i in indices]
    n_color = sum(sample)
    n_gray = k - n_color

    # Determine the mode by majority vote.
    if n_color >= n_gray:
        ae_mode = "rgb"
        mismatches = n_gray  # gray samples in an RGB-dominated regime
    else:
        ae_mode = "gray"
        mismatches = n_color # color samples in a GRAY-dominated regime

    if mismatches >= 1:
        print(
            f"[WARN] dataset not homogeneous in 100-sample check: "
            f"{n_color} color, {n_gray} gray (mode={ae_mode}, mismatches={mismatches})."
        )
    if mismatches > 5:
        raise RuntimeError(
            f"Too many mismatching images ({mismatches}) in 100-sample check. "
            f"Please clean up the dataset (or split it across distinct train roots)."
        )

    expected_c = 3 if ae_mode == "rgb" else 1
    print(
        f"[MODE] AE training mode: {ae_mode.upper()} "
        f"(expected channels={expected_c}) based on {k} samples: "
        f"{n_color} color, {n_gray} gray."
    )

    return ae_mode, expected_c


# =========================
# -------- MODEL ----------
# =========================

class ConvGNAct(nn.Module):
    """
    Convenience block: convolution followed by GroupNorm and GELU activation.

    This pattern is used repeatedly in the encoder and decoder to keep
    the code compact while maintaining good optimization behaviour.
    """
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.gn   = nn.GroupNorm(num_groups=min(32, out_ch), num_channels=out_ch)
        self.act  = nn.GELU()

    def forward(self, x):
        return self.act(self.gn(self.conv(x)))


class EncoderY(nn.Module):
    """
    Generic encoder used both for the luminance branch (Y, in_ch=1)
    and the chroma branch (CbCr, in_ch=2).

    The encoder performs a sequence of strided convolutions to reduce
    the spatial resolution from 512×512 down to 32×32 while increasing
    channel depth. It exposes two intermediate representations:

    - s['s64']    : [B, d_skip, 64, 64]  (skip-level features)
    - s['latent'] : [B, d_lat,  32, 32]  (deep latent representation)

    These tensors define the internal "surfaces" to which external
    watermark signals can be attached.
    """
    def __init__(self, ch_stem: int = 64, d_skip: int = 512, d_lat: int = 1024, in_ch: int = 1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, ch_stem, 7, 2, 3, bias=False),            # 512 -> 256
            nn.GroupNorm(min(32, ch_stem), ch_stem), nn.GELU(),
            nn.Conv2d(ch_stem, ch_stem, 3, 2, 1, bias=False),          # 256 -> 128
            nn.GroupNorm(min(32, ch_stem), ch_stem), nn.GELU(),
        )
        self.to64 = nn.Sequential(
            nn.Conv2d(ch_stem, d_skip, 3, 2, 1, bias=False),           # 128 -> 64
            nn.GroupNorm(min(32, d_skip), d_skip), nn.GELU(),
            nn.Conv2d(d_skip, d_skip, 3, 1, 1, bias=False),
            nn.GroupNorm(min(32, d_skip), d_skip), nn.GELU(),          # -> [B,d_skip,64,64]
        )
        self.to32 = nn.Sequential(
            nn.Conv2d(d_skip, d_lat, 3, 2, 1, bias=False),             # 64 -> 32
            nn.GroupNorm(min(32, d_lat), d_lat), nn.GELU(),
            nn.Conv2d(d_lat, d_lat, 3, 1, 1, bias=False),
            nn.GroupNorm(min(32, d_lat), d_lat), nn.GELU(),            # -> [B,d_lat,32,32]
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x: [B,in_ch,512,512]
        x = self.stem(x)                      # [B, ch_stem,128,128]
        x = self.to64(x)                      # [B, d_skip, 64, 64]
        s64 = x
        x = self.to32(x)                      # [B, d_lat, 32, 32]
        latent = x
        assert s64.shape[-2:] == (64, 64),   f"s64 wrong spatial shape: {s64.shape}"
        assert latent.shape[-2:] == (32, 32), f"latent wrong spatial shape: {latent.shape}"
        return {"s64": s64, "latent": latent}


class DecoderY(nn.Module):
    """
    Generic decoder which reconstructs an image from encoder states.

    Typical usage:
    - out_ch = 1 for the Y branch
    - out_ch = 2 for the CbCr branch

    The decoder performs:
    - 32×32  -> 64×64   (d_lat -> d_skip), then fuses skip64,
    - 64×64  -> 128×128,
    - 128×128 -> 256×256,
    - 256×256 -> 512×512,
    finally projecting to `out_ch` channels via a 1×1 convolution.
    """
    def __init__(self, d_skip: int = 512, d_lat: int = 1024, d1: int = 256, d0: int = 128, out_ch: int = 1):
        super().__init__()
        self.up32to64 = nn.ModuleDict({
            "conv1": nn.ConvTranspose2d(d_lat, d_skip, 2, 2, bias=False),
            "nr":    nn.Sequential(nn.GroupNorm(min(32, d_skip), d_skip), nn.GELU()),
            "conv2": nn.Conv2d(d_skip, d_skip, 3, 1, 1, bias=False),
            "nr2":   nn.Sequential(nn.GroupNorm(min(32, d_skip), d_skip), nn.GELU()),
        })
        self.fuse64 = nn.Sequential(
            nn.Conv2d(d_skip + d_skip, d_skip, 3, 1, 1, bias=False),
            nn.GroupNorm(min(32, d_skip), d_skip), nn.GELU()
        )
        self.up64to128 = nn.ModuleDict({
            "conv1": nn.ConvTranspose2d(d_skip, d1, 2, 2, bias=False),
            "nr":    nn.Sequential(nn.GroupNorm(min(32, d1), d1), nn.GELU()),
            "conv2": nn.Conv2d(d1, d1, 3, 1, 1, bias=False),
            "nr2":   nn.Sequential(nn.GroupNorm(min(32, d1), d1), nn.GELU()),
        })
        self.up128to256 = nn.ModuleDict({
            "conv1": nn.ConvTranspose2d(d1, d0, 2, 2, bias=False),
            "nr":    nn.Sequential(nn.GroupNorm(min(32, d0), d0), nn.GELU()),
            "conv2": nn.Conv2d(d0, d0, 3, 1, 1, bias=False),
            "nr2":   nn.Sequential(nn.GroupNorm(min(32, d0), d0), nn.GELU()),
        })
        self.up256to512 = nn.ModuleDict({
            "conv1": nn.ConvTranspose2d(d0, d0//2, 2, 2, bias=False),
            "nr":    nn.Sequential(nn.GroupNorm(min(32, d0//2), d0//2), nn.GELU()),
            "conv2": nn.Conv2d(d0//2, d0//2, 3, 1, 1, bias=False),
            "nr2":   nn.Sequential(nn.GroupNorm(min(32, d0//2), d0//2), nn.GELU()),
        })
        self.out_conv = nn.Conv2d(d0//2, out_ch, 1, 1, 0)

    def forward(self, s: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = s["latent"]                              # [B,d_lat,32,32]
        x = self.up32to64["conv1"](x); x = self.up32to64["nr"](x)
        x = self.up32to64["conv2"](x); x = self.up32to64["nr2"](x)      # -> [B,d_skip,64,64]
        assert x.shape[-2:] == s["s64"].shape[-2:] == (64, 64), \
            f"skip64 mismatch: {x.shape} vs {s['s64'].shape}"
        x = torch.cat([x, s["s64"]], dim=1)                             # [B,2*d_skip,64,64]
        x = self.fuse64(x)                                              # [B, d_skip,64,64]
        x = self.up64to128["conv1"](x); x = self.up64to128["nr"](x)     # [B,d1,128,128]
        x = self.up64to128["conv2"](x); x = self.up64to128["nr2"](x)
        x = self.up128to256["conv1"](x); x = self.up128to256["nr"](x)   # [B,d0,256,256]
        x = self.up128to256["conv2"](x); x = self.up128to256["nr2"](x)
        x = self.up256to512["conv1"](x); x = self.up256to512["nr"](x)   # [B,d0//2,512,512]
        x = self.up256to512["conv2"](x); x = self.up256to512["nr2"](x)
        return torch.sigmoid(self.out_conv(x))                          # [B,out_ch,512,512]


class UniversalAutoEncoder(nn.Module):
    """
    Universal autoencoder with separate Y and CbCr branches.

    The network accepts either:
    - [B, 3, H, W] RGB inputs; or
    - [B, 1, H, W] grayscale inputs,

    and internally operates as follows:
    - RGB inputs are converted to YCbCr and processed by:
        * Y branch (1 channel) via `enc_y` / `dec_y`
        * CbCr branch (2 channels) via `enc_c` / `dec_c`
    - Grayscale inputs use only the Y branch.

    In addition, the model exposes external watermark injection surfaces:
    - Y-branch latent      : [B,1024,32,32]
    - Y-branch skip64      : [B, 512,64,64]
    - CbCr residuals @64×64: [B,  2,64,64] (for RGB only)

    This design enables controlled experiments in which the watermark
    is injected at different representational depths.
    """
    def __init__(self):
        super().__init__()
        # Y (luma) branch
        self.enc_y = EncoderY(in_ch=1)
        self.dec_y = DecoderY(out_ch=1)
        # CbCr (chroma) branch - only used when input is RGB
        self.enc_c = EncoderY(in_ch=2)
        self.dec_c = DecoderY(out_ch=2)
        # Backward-compatible aliases (old code may look at .enc/.dec)
        self.enc = self.enc_y
        self.dec = self.dec_y

    # ===== Base forward (pretraining / plain reconstruction) =====
    def forward_plain(self, x01: torch.Tensor) -> torch.Tensor:
        """
        Plain forward pass without watermark injection, used for
        pretraining and basic reconstruction.

        Parameters
        ----------
        x01 : torch.Tensor
            Input image tensor of shape [B, C, H, W] in [0, 1],
            where C is either 1 (gray) or 3 (RGB).

        Returns
        -------
        torch.Tensor
            Reconstructed image tensor with the same number of channels
            as the input.
        """
        if x01.size(1) == 3:
            # RGB -> YCbCr, run both branches
            y, cb, cr = rgb_to_ycbcr(x01)
            c = torch.cat([cb, cr], dim=1)                     # [B,2,H,W]
            s_y = self.enc_y(y)
            s_c = self.enc_c(c)
            y_hat = self.dec_y(s_y)                            # [B,1,H,W]
            c_hat = self.dec_c(s_c)                            # [B,2,H,W]
            cb_hat, cr_hat = c_hat[:, 0:1], c_hat[:, 1:2]
            return ycbcr_to_rgb(y_hat, cb_hat.clamp(0, 1), cr_hat.clamp(0, 1))
        else:
            # Pure grayscale: only Y branch
            s_y = self.enc_y(x01)
            y_hat = self.dec_y(s_y)
            return y_hat

    def forward(self, x01: torch.Tensor) -> torch.Tensor:
        # Default forward is the plain reconstruction path.
        return self.forward_plain(x01)

    # ===== External WM surfaces (kept API-compatible) =====
    @staticmethod
    def _l2_normalize_per_sample(t: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Normalize the tensor per sample to have L2 norm 1.

        This is useful when watermark tensors need to be constrained
        to a fixed energy budget independent of their raw magnitude.
        """
        B = t.size(0)
        n = t.flatten(1).norm(p=2, dim=1).clamp_min(eps)
        return t / n.view(B, 1, 1, 1)

    @staticmethod
    def _ensure_size(t: torch.Tensor, size_hw: Tuple[int, int]) -> torch.Tensor:
        """
        Ensure that tensor `t` has the desired spatial size, resizing
        via bilinear interpolation if necessary.
        """
        if t.shape[-2:] == size_hw:
            return t
        return F.interpolate(t, size_hw, mode="bilinear", align_corners=False)

    def embed_external_wm(
        self,
        x01: torch.Tensor,
        wm_lat: Optional[torch.Tensor] = None,     # [B,1024,32,32] for Y-branch
        wm_skip: Optional[torch.Tensor] = None,    # [B,512,64,64]  for Y-branch
        wm_cbcr_64: Optional[torch.Tensor] = None, # [B,2,64,64], RGB only
        alpha_lat: float = 0.10,
        alpha_skip: float = 0.10,
        alpha_cbcr: float = 0.05,
        roi_lat_32: Optional[torch.Tensor] = None, # [B,1,32,32] 0/1 mask
        roi_skip_64: Optional[torch.Tensor] = None # [B,1,64,64] 0/1 mask
    ) -> torch.Tensor:
        """
        Forward pass with external watermark injection.

        Parameters
        ----------
        x01 : torch.Tensor
            Input tensor of shape [B, C, H, W] in [0, 1], C ∈ {1, 3}.
        wm_lat : torch.Tensor, optional
            Watermark tensor injected into the Y-branch latent,
            expected shape [B, 1024, 32, 32]. Spatial size is
            auto-resized if needed.
        wm_skip : torch.Tensor, optional
            Watermark tensor injected into the Y-branch skip64,
            expected shape [B, 512, 64, 64].
        wm_cbcr_64 : torch.Tensor, optional
            Chroma watermark residuals ΔCb, ΔCr at 64×64 resolution
            for RGB inputs only. Shape [B, 2, 64, 64].
        alpha_lat, alpha_skip, alpha_cbcr : float
            Scalar mixing coefficients controlling the strength of
            the watermark in each injection surface.
        roi_lat_32 : torch.Tensor, optional
            Spatial region-of-interest mask for the latent surface,
            shape [B, 1, 32, 32] with values in {0,1}. If provided,
            watermark energy is restricted to the masked region.
        roi_skip_64 : torch.Tensor, optional
            Spatial ROI mask for the skip64 surface,
            shape [B, 1, 64, 64].

        Notes
        -----
        All watermark tensors are L2-normalized per sample before
        being added, so the effective perturbation is controlled
        by the α-coefficients and ROI masks.
        """
        is_rgb = (x01.size(1) == 3)
        if is_rgb:
            y, cb, cr = rgb_to_ycbcr(x01)
            c = torch.cat([cb, cr], dim=1)                        # [B,2,H,W]
            s_y = self.enc_y(y)
            s_c = self.enc_c(c)
        else:
            s_y = self.enc_y(x01)
            s_c = None

        # --- latent on Y-branch ---
        if wm_lat is not None:
            wm_lat = self._ensure_size(wm_lat, (32, 32))
            assert wm_lat.size(1) == s_y["latent"].size(1) == 1024, \
                f"wm_lat channels must be 1024 (got {wm_lat.size(1)})"
            wm_lat = self._l2_normalize_per_sample(wm_lat)
            if roi_lat_32 is not None:
                wm_lat = wm_lat * self._ensure_size(roi_lat_32, (32, 32))
            s_y["latent"] = s_y["latent"] + alpha_lat * wm_lat

        # --- skip64 on Y-branch ---
        if wm_skip is not None:
            wm_skip = self._ensure_size(wm_skip, (64, 64))
            assert wm_skip.size(1) == s_y["s64"].size(1) == 512, \
                f"wm_skip channels must be 512 (got {wm_skip.size(1)})"
            wm_skip = self._l2_normalize_per_sample(wm_skip)
            if roi_skip_64 is not None:
                wm_skip = wm_skip * self._ensure_size(roi_skip_64, (64, 64))
            s_y["s64"] = s_y["s64"] + alpha_skip * wm_skip

        # --- decode branches ---
        y_hat = self.dec_y(s_y)                                   # [B,1,H,W]

        if is_rgb:
            # Decode chroma and optionally add chroma WM residuals.
            assert s_c is not None
            c_hat = self.dec_c(s_c)                               # [B,2,H,W]
            cb_hat, cr_hat = c_hat[:, 0:1], c_hat[:, 1:2]

            if wm_cbcr_64 is not None:
                # External chroma residuals (ΔCb, ΔCr @64x64) -> upscale to full resolution
                wm_cbcr_64 = self._ensure_size(wm_cbcr_64, (64, 64))
                assert wm_cbcr_64.size(1) == 2, \
                    f"wm_cbcr_64 must have 2 channels (got {wm_cbcr_64.size(1)})"
                wm_cbcr_64 = self._l2_normalize_per_sample(wm_cbcr_64)
                d_cbcr = F.interpolate(
                    wm_cbcr_64, scale_factor=8, mode="bilinear", align_corners=False
                )
                d_cb, d_cr = d_cbcr[:, 0:1], d_cbcr[:, 1:2]
                cb_hat = torch.clamp(cb_hat + alpha_cbcr * d_cb, 0.0, 1.0)
                cr_hat = torch.clamp(cr_hat + alpha_cbcr * d_cr, 0.0, 1.0)

            return ycbcr_to_rgb(y_hat, cb_hat, cr_hat)
        else:
            # Grayscale: only Y branch is active
            return y_hat

    # Convenience wrappers with explicit input types
    def embed_external_wm_rgb(self, x_rgb: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Convenience wrapper for `embed_external_wm` for RGB inputs.

        Parameters
        ----------
        x_rgb : torch.Tensor
            Tensor of shape [B, 3, H, W] in [0, 1].
        """
        assert x_rgb.size(1) == 3, "x_rgb must be [B,3,H,W]"
        return self.embed_external_wm(x_rgb, **kwargs)

    def embed_external_wm_gray(self, x_gray: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Convenience wrapper for `embed_external_wm` for grayscale inputs.

        Parameters
        ----------
        x_gray : torch.Tensor
            Tensor of shape [B, 1, H, W] in [0, 1].
        """
        assert x_gray.size(1) == 1, "x_gray must be [B,1,H,W]"
        return self.embed_external_wm(x_gray, **kwargs)


# =========================
# -------- TRAIN ----------
# =========================

def parse_args():
    """
    Parse command-line arguments controlling the pretraining run.

    Key arguments:
    - epochs : number of full passes over the training data (default: 10)
    - bs     : batch size
    - size   : spatial resolution to which all images are resized
    - workers: number of data-loader workers
    - lr     : learning rate for AdamW
    - amp    : enable automatic mixed precision (AMP)
    - device / multi-GPU related flags
    - seed   : RNG seed
    - log-every: interval (in iterations) between progress logs
    """
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10)                # <- requested default
    p.add_argument("--bs", type=int, default=4)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--amp", action="store_true", default=True, help="use AMP")
    # device
    p.add_argument("--cuda1-only", action="store_true", help="use CUDA:1 if available")
    p.add_argument("--all-gpus",   action="store_true", help="use DataParallel across all GPUs")
    p.add_argument("--cuda-index", type=int, default=1, help="fallback single GPU index")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--log-every", type=int, default=100)

    # dataset options
    p.add_argument("--scan-flags", action="store_true", default=True,
                   help="scan image color/gray flags at dataset init")
    return p.parse_args()


def tv_loss(z: torch.Tensor) -> torch.Tensor:
    """
    Isotropic total variation (TV) regularizer.

    This loss is not used in the default pretraining procedure, but is
    kept for potential future experiments where smoothness of the
    latent or decoded representations is explicitly encouraged.
    """
    return (z[:, :, :-1, :] - z[:, :, 1:, :]).abs().mean() + \
           (z[:, :, :, :-1] - z[:, :, :, 1:]).abs().mean()


def train_loop(args):
    """
    Main training loop for the universal autoencoder.

    Responsibilities:
    - set global RNG seed and select device(s);
    - build MixedImageDataset instances for training and validation;
    - infer the AE operating mode (RGB vs GRAY) and configure collate;
    - construct a balanced 1:1 batch sampler when both modalities are present;
    - perform forward/backward passes with optional AMP;
    - accumulate split PSNR/SSIM metrics;
    - run validation at the end of each epoch;
    - implement early stopping and checkpoint saving.

    The actual CSV logging of metrics is left as a straightforward extension
    and can be customized depending on the target experiment.
    """
    seed_all(args.seed)

    # device
    if torch.cuda.is_available():
        if args.cuda1_only and torch.cuda.device_count() > 1:
            device = torch.device("cuda:0")
        else:
            device = torch.device(f"cuda:{args.cuda_index}")
    else:
        device = torch.device("cpu")
    print(f"[Device] {device} | all_gpus={args.all_gpus}")

    # model
    ae = UniversalAutoEncoder().to(device)
    if args.all_gpus and torch.cuda.device_count() > 1:
        ae = nn.DataParallel(ae)

    opt = torch.optim.AdamW(ae.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    # --- dataset roots: up to 3 datasets, use only existing ones ---

        # A third dataset (train, val) can be added here later, e.g.:
        # (ORNL_TRAIN, ORNL_VAL),


    train_roots: List[str] = []
    val_roots:   List[str] = []

    for tr, vr in TRAIN_ROOT_PAIRS:
        if tr and Path(tr).exists():
            train_roots.append(tr)
            if vr and Path(vr).exists():
                val_roots.append(vr)
            else:
                print(f"[WARN] val root not found for train root: {tr} (val={vr})")

    if not train_roots:
        raise RuntimeError("No valid train dataset roots provided.")

    # ---- sanity-check each train dataset on up to 100 random images ----
    train_modes: List[str] = []
    for tr in train_roots:
        ds_name = os.path.basename(os.path.normpath(tr)) or tr
        mode = detect_dataset_mode(tr, max_sample=100, name=ds_name)
        train_modes.append(mode)

    unique_train_modes = set(train_modes)
    if len(unique_train_modes) > 1:
        # Mixing RGB and GRAY datasets within a single run is not allowed.
        raise RuntimeError(
            f"Mixed dataset types in one run: {train_modes}. "
            f"All train-datasets in this run must be either RGB or GRAYSCALE only."
        )

    ae_mode = train_modes[0]                   # "rgb" or "gray"
    expected_c = 3 if ae_mode == "rgb" else 1
    print(
        f"[AE MODE] Global AE mode: {ae_mode.upper()} (expected channels={expected_c}) "
        f"for train_roots={train_roots}"
    )

    # --- MixedImageDataset construction and collate configuration ---
    train_ds = MixedImageDataset(train_roots, size=args.size, scan_flags=args.scan_flags)
    val_ds = MixedImageDataset(val_roots, size=args.size, scan_flags=args.scan_flags)
    collate_fn = make_collate_fn(expected_c, ae_mode)

    # Build a 1:1 balanced batch sampler for color vs gray, as in previous versions.
    flags = train_ds.is_color_flags()
    idx_color = [i for i, f in enumerate(flags) if f]
    idx_gray  = [i for i, f in enumerate(flags) if not f]

    if not idx_color or not idx_gray:
        print("Processing mode finalized from dataset color-ness inspection.")
        train_loader = DataLoader(
            train_ds,
            batch_size=args.bs,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=(device.type == "cuda"),
            drop_last=True,
            collate_fn=collate_fn,
        )
    else:
        batch_sampler = AlternatingBatchSampler(
            idx_color,
            idx_gray,
            batch_size=args.bs,
            drop_last=True,
            shuffle=True
        )
        train_loader = DataLoader(
            train_ds,
            batch_sampler=batch_sampler,
            num_workers=args.workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )

    # --- remaining train_loop structure (logging, validation, checkpointing) ---
    # The original script performs per-iteration logging and per-epoch validation;
    # here we keep the same structure and additionally introduce early stopping.
    csv_path = LOG_DIR / "train_log_split.csv"
    ...
    best_psnrY = -1.0

    for epoch in range(1, args.epochs + 1):
        ae.train()
        it = 0
        meters = SplitMeters()
        t0 = time.time()

        for x, is_color_flags, paths in train_loader:
            it += 1
            x = x.to(device, non_blocking=True)        # [B,expected_c,H,W]
            is_color_flags = is_color_flags.to(device) # [B] bool

            with torch.amp.autocast(device_type="cuda", enabled=args.amp):
                y_hat = ae(x)                          # AE internally handles both C=1 and C=3
                loss  = F.l1_loss(y_hat, x)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            with torch.no_grad():
                update_split_metrics(meters, x.detach(), y_hat.detach(), is_color_flags)

            if it % args.log_every == 0:
                line = meters.log_line(epoch, it, loss.item())
                print(line)
                # CSV logging (if needed) can be implemented here, following
                # the same pattern as in the original script.

        # ----- VAL -----
        ae.eval()
        val_m = SplitMeters()
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=args.amp):
            for xv, is_color_v, _ in val_loader:
                xv = xv.to(device, non_blocking=True)
                is_color_v = is_color_v.to(device)
                yv = ae(xv)
                update_split_metrics(val_m, xv, yv, is_color_v)

        print(
            f"[VAL E{epoch:02d}] "
            f"Y: psnr={val_m.psnr_y_all.avg:.2f} ssim={val_m.ssim_y_all.avg:.3f} | "
            f"Ycolor={val_m.psnr_y_color.avg:.2f}/{val_m.ssim_y_color.avg:.3f} "
            f"Ygray={val_m.psnr_y_gray.avg:.2f}/{val_m.ssim_y_gray.avg:.3f} | "
            f"RGBcolor={val_m.psnr_rgb_color.avg:.2f}/{val_m.ssim_rgb_color.avg:.3f} | "
            f"time={time.time()-t0:.1f}s"
        )

        # --- early stopping based on SSIM_Y ---
        # SSIM is in [0,1]; 98.9% corresponds to 0.989.
        if val_m.ssim_y_all.avg >= 0.989:
            state = ae.module.state_dict() if isinstance(ae, nn.DataParallel) else ae.state_dict()
            early_path = CKPT_DIR / "universal_ae_early_stop.pth"
            torch.save(state, early_path)
            print(
                f"[EARLY STOP] SSIM_Y={val_m.ssim_y_all.avg:.4f} >= 0.989 at epoch {epoch}. "
                f"Saved early-stop checkpoint to {early_path}"
            )
            break

        # --- save checkpoints every epoch and track the best model by PSNR_Y, as before ---
        state = ae.module.state_dict() if isinstance(ae, nn.DataParallel) else ae.state_dict()
        ckpt_path = CKPT_DIR / f"universal_ae_epoch{epoch:02d}.pth"
        torch.save(state, ckpt_path)

        if val_m.psnr_y_all.avg > best_psnrY:
            best_psnrY = val_m.psnr_y_all.avg
            best_path = CKPT_DIR / "universal_ae_best.pth"
            torch.save(state, best_path)
            print(f"[SAVE] best -> {best_path} (PSNR_Y {best_psnrY:.2f})")

    print("[DONE] training complete.")


# =========================
# --------  MAIN  ---------
# =========================

def parse_cli():
    """
    Compatibility wrapper preserved for older launch scripts that
    expect a `parse_cli()` entrypoint.
    """
    return parse_args()


if __name__ == "__main__":
    args = parse_cli()
    train_loop(args)
