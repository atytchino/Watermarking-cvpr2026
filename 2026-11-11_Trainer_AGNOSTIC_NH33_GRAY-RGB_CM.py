# -*- coding: utf-8 -*-
"""
AFHQ / ORNL Watermark Trainer (Gray, C1‑driven) — Dual‑Step + EMA

This script implements a two–stage training pipeline for a *detector‑driven* image
watermarking scheme under a grayscale (single–channel) autoencoder:

  * G (generator) operates in AE latent space and a 64×64 skip connection:
      - latent branch:   Δz ∈ R^{1024×32×32}
      - skip‑64 branch:  Δs ∈ R^{512×64×64} (frequency–biased via DCT)
  * C1 is a frozen "guard" classifier with Grad‑CAM, defining *where* the watermark
    is allowed to live (ROI). C1 is treated as immutable infrastructure.
  * C2 is a trainable watermark‑aware classifier operating in grayscale space.

Key design choices:
  * Δ(C2) stabilization:
      - C2 has an explicit watermark head (scalar wm_logit per image).
      - Clean vs watermarked logits are gated by wm_logit through an affine head
        so that class separation explicitly depends on the watermark presence.
      - G is trained so that C1 remains stable, while C2 learns to rely on Δ induced
        by the watermark.

  * Spatial control of watermark support:
      - Hard ROI via top‑k selection in latent / skip‑64, combined with a border ring.
      - All watermark patterns are explicitly zeroed outside ROI.
      - Leak loss penalizes any residual perturbation outside ROI in pixel space.

  * Channel / frequency control:
      - Skip‑64 branch is projected from 1 channel to 512 via a frozen 1×1 conv with
        uniform weights ("anti‑grain"): avoids spiky channel‑wise artifacts while
        giving full‑channel access.
      - DCT‑based shaping (low/mid‑frequency mask) and small low‑pass filtering on
        the latent branch reduce speckle and high‑frequency noise.

  * Controller:
      - Maintains a global perturbation budget ε and a ratio r_skip controlling
        how much of ε goes into latent vs skip‑64.
      - Enforces latent to carry at least 1/3 of the total energy; skip‑64 is capped.
      - Uses PSNR, SSIM, Δ(C2) and C1‑drop to adapt ε and r_skip each epoch.

External AE with official API:
    - forward_plain(x01) → recon
    - embed_external_wm_gray(
          x01, wm_lat, wm_skip, alpha_lat, alpha_skip,
          roi_lat_32, roi_skip_64
      )
    - enc(y) → dict with:
          * "latent" : [B, 1024, 32, 32]
          * "s64"    : [B,  512, 64, 64]

All tensors denoted x01 are assumed to be in [0, 1]; inputs/outputs to classifiers
are normalized to [-1, 1] when needed.
"""

import os, sys, time, math, random
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Dict, Any, List

from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.datasets import ImageFolder

# =========================
# -------- CONFIG ---------
# =========================

# --- User paths (adapt to your environment) ---
C1_CKPT = r"E:\ORNL_GRAYSCALE\C1_RS34\best_raw.pth"
TRAIN_DIR = r"E:\ORNL_GRAYSCALE\Train"
VAL_DIR   = r"E:\ORNL_GRAYSCALE\Val"    # subfolders: class names (e.g. cat, dog, wild)
OUT_ROOT  = r"E:\ORNL_GRAYSCALE\ORNL_watermarked_20251124"

# --- Fixed parameters related to the external autoencoder ---
AE_PTH    = r"E:\Universal_ae_ver2_ORNL\universal_ae_best.pth"
AE_IMPORT  = ("AE_universal_Nov2025", "UniversalAutoEncoder")  # (module, class)
AE_PY_PATH = None  # e.g. r"E:\my_models"

# --- Core training hyperparameters ---
IMAGE_SIZE  = 512
BATCH_SIZE  = 4
NUM_WORKERS = 8
EPOCHS      = 30
ACC_STEPS_C2 = 2
STRICT_NAMES = True

# --- Device selection (robust to single‑GPU setups) ---
if torch.cuda.is_available():
    try:
        torch.cuda.set_device(1)
        DEVICE = torch.device("cuda:1")
        print(f"[WM TRAIN] Using device: {DEVICE}")
    except Exception:
        DEVICE = torch.device("cuda:0")
else:
    DEVICE = torch.device("cpu")

MIXED_PRECISION = True  # enable AMP for C2 and part of G‑loss

# --- Controller / budgets ---
EPS_WARMUP_FRAC = 0.25   # fraction of epochs used to linearly warm up ε to EPS_MAX
EPS_MAX         = 0.08   # global perturbation magnitude ceiling in AE space
RSKIP_INIT      = 0.65   # initial fraction of ε assigned to skip‑64 (latent gets the rest)
CTRL_UPDATE_EVERY = 2    # controller update frequency in iterations
C1_DROP_BLOCK_INCREASE = 0.01  # if C1 accuracy drop exceeds this, block ε increase
PSNR_MIN        = 30.0   # minimal PSNR to consider perturbations "acceptable"
EPS_MIN         = 0.003  # lower bound on ε to avoid trivial solutions
TARGET_DELTA    = 0.65   # target Δ(C2) = Acc_wm − Acc_clean
TARGET_HYST     = 0.02   # hysteresis around TARGET_DELTA for controller decisions
EMAD_BETA       = 0.6    # EMA coefficient for Δ(C2) smoothing

# --- DCT config (skip‑64 shaping) ---
DCT_KEEP_RATIO = 0.10    # effective radius of low/mid‑frequency band in DCT
DC_SCALE       = 0.2     # explicit scaling of DC component in the DCT mask

# --- Optimizer hyperparameters ---
LR_G = 1e-4
LR_C2 = 5e-4
WEIGHT_DECAY = 1e-4

# --- Loss weights ---
LAMBDA_SPARSE    = 1e-3   # sparsity / area penalty for ROI masks
LAMBDA_OVERLAP   = 2.0    # (unused in this version, reserved for future overlap losses)
LAMBDA_TV        = 2e-3   # total variation regularizer on ROI masks
LAMBDA_C1_GUARD  = 0.1    # weight of C1‑consistency sentinel term
LAMBDA_SEP_IN_G  = 0.25   # reserved for additional separation term inside G (not used)
ALPHA_MARGIN     = 1.0    # reserved margin parameter
BETA_WM_BCE      = 0.75   # reserved BCE scaling for wm head
BETA_BAL         = 0.35   # reserved balancing coefficient
MARGIN_M         = 1.5    # margin for C2 logits separation
LAMBDA_LEAK      = 1.0    # leak loss coefficient (outside‑ROI energy)

# --- AE taps shape (dimensionality of generator heads) ---
LAT_CH = 1024
S64_CH = 512

# --- Watermark embed frequency ---
P_ON = 0.90  # watermark is turned on with probability 0.9 (for C2 training)


# =========================
# -------- UTILS ----------
# =========================

def read_ckpt_classes(ckpt_path: str):
    """
    Read the list of class names stored inside a C1 checkpoint.

    The function is intentionally strict: it expects either a top‑level key
    ``'classes'`` or ``meta['classes']``, and validates that each entry is a string.
    """
    import torch
    ckpt = torch.load(ckpt_path, map_location="cpu")
    classes = None
    if isinstance(ckpt, dict):
        classes = ckpt.get("classes")
        if classes is None:
            meta = ckpt.get("meta")
            if isinstance(meta, dict):
                classes = meta.get("classes")
    if classes is None:
        raise ValueError(
            f"[C1 CHECK] '{ckpt_path}' does not contain a class list "
            f"(expected 'classes' or meta['classes'])."
        )
    if not isinstance(classes, (list, tuple)) or not all(isinstance(c, str) for c in classes):
        raise ValueError(f"[C1 CHECK] Invalid 'classes' format in checkpoint: {type(classes)}")
    return list(classes)


def seed_everything(seed=1337):
    """Seed Python, NumPy, and PyTorch RNGs for reproducible training."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_everything(1337)


def ensure_dir(p: str | Path):
    """Create a directory (recursively) if it does not already exist."""
    Path(p).mkdir(parents=True, exist_ok=True)


def psnr_torch(x, x_hat, data_range=1.0):
    """
    Compute scalar PSNR between two batches of images in [0, 1].

    The metric is averaged over the entire tensor; this is not per‑image PSNR
    but a single scalar capturing the global distortion level.
    """
    mse = F.mse_loss(x_hat, x, reduction='mean').clamp_min(1e-12)
    return 10.0 * torch.log10((data_range ** 2) / mse)


def ssim_torch(x, y, C1=0.01**2, C2=0.03**2):
    """
    Compute a simple SSIM estimate using a fixed 11×11 averaging kernel.

    Inputs:
        x, y : tensors of shape [B, C, H, W] in [0, 1], typically 1 or 3 channels.

    The result is a single scalar, the batch‑average of the spatial SSIM map.
    """
    mu_x = F.avg_pool2d(x, 11, 1, 5)
    mu_y = F.avg_pool2d(y, 11, 1, 5)
    sigma_x = F.avg_pool2d(x * x, 11, 1, 5) - mu_x ** 2
    sigma_y = F.avg_pool2d(y * y, 11, 1, 5) - mu_y ** 2
    sigma_xy = F.avg_pool2d(x * y, 11, 1, 5) - mu_x * mu_y
    ssim_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    ssim_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    ssim = ssim_n / (ssim_d + 1e-8)
    return ssim.mean()


def js_divergence(p, q, eps: float = 1e-8):
    """
    Jensen–Shannon divergence between two categorical distributions.

    Inputs:
        p, q : tensors of shape [B, K] representing probabilities.
    """
    p = p.float().clamp_min(eps)
    q = q.float().clamp_min(eps)
    m = 0.5 * (p + q)
    js = 0.5 * ((p * (p.log() - m.log())).sum(dim=1) +
                (q * (q.log() - m.log())).sum(dim=1))
    return js.mean()


def cosine_logits(a, b):
    """Mean cosine similarity between two batches of logits (used as a stability proxy)."""
    return F.cosine_similarity(a, b, dim=1).mean()


# ---------- Orthonormal DCT-II / DCT-III (matrix-based, cached) ----------

def _dct_matrix(N: int, device, dtype):
    """
    Construct an orthonormal DCT‑II matrix C ∈ R^{N×N} such that:

        X = C * x

    corresponds to a 1D DCT‑II transform.
    """
    n = torch.arange(N, device=device, dtype=dtype).reshape(1, N)
    k = torch.arange(N, device=device, dtype=dtype).reshape(N, 1)
    C = torch.cos(math.pi * (n + 0.5) * k / N)
    C[0, :] *= 1.0 / math.sqrt(N)
    C[1:, :] *= math.sqrt(2.0 / N)
    return C  # [N,N]


class DCTCache(nn.Module):
    """
    Cached 2D DCT/IDCT implementation for fixed spatial size (H, W).

    This module keeps DCT matrices on the correct device/dtype and provides
    dct2 / idct2 operations for inputs of shape [B, H, W].
    """
    def __init__(self, h=64, w=64):
        super().__init__()
        self.h = h
        self.w = w
        self.register_buffer("C_h", torch.empty(0))
        self.register_buffer("C_w", torch.empty(0))

    def _ensure(self, device, dtype):
        need_init = (self.C_h.numel() == 0) or (self.C_h.device != device) or (self.C_h.dtype != dtype)
        if need_init:
            C_h = _dct_matrix(self.h, device, dtype)  # [H,H]
            C_w = _dct_matrix(self.w, device, dtype)  # [W,W]
            self.C_h = C_h
            self.C_w = C_w

    def dct2(self, x):  # x: [B,H,W]
        """2D orthonormal DCT‑II."""
        self._ensure(x.device, x.dtype)
        X = torch.matmul(x, self.C_w.t())
        X = X.permute(0, 2, 1)
        X = torch.matmul(X, self.C_h.t())
        X = X.permute(0, 2, 1)
        return X

    def idct2(self, X):  # inverse (orthonormal DCT‑III)
        """2D orthonormal DCT‑III (inverse transform)."""
        self._ensure(X.device, X.dtype)
        Y = torch.matmul(X, self.C_w)
        Y = Y.permute(0, 2, 1)
        Y = torch.matmul(Y, self.C_h)
        Y = Y.permute(0, 2, 1)
        return Y


class LocalVariance(nn.Module):
    """
    Local variance estimator used to identify low‑texture (visually smooth) regions.

    This is used as a prior suggesting that watermark energy should preferably be
    placed in regions with higher structural support (i.e. not entirely flat).
    """
    def __init__(self, k=7):
        super().__init__()
        self.k = k
        self.pool = nn.AvgPool2d(k, 1, k // 2)

    def forward(self, y):  # y: [B,1,H,W] in [0,1]
        mu = self.pool(y)
        sigma2 = self.pool(y * y) - mu * mu
        # Normalize variance to [0,1] within each batch
        sigma2_min = sigma2.amin(dim=(2, 3), keepdim=True)
        sigma2_max = sigma2.amax(dim=(2, 3), keepdim=True)
        var = (sigma2 - sigma2_min) / (sigma2_max - sigma2_min + 1e-8)
        # "Low‑texture" map is the inverse of normalized variance
        lowtex = 1.0 - var
        return lowtex.clamp(0, 1)


# =========================
# -------- DATA -----------
# =========================

class ImageFolderWithPaths(ImageFolder):
    """
    ImageFolder subclass that additionally returns the absolute path
    of each sample, allowing per‑image logging and collages.
    """
    def __getitem__(self, index):
        sample, target = super().__getitem__(index)
        path, _ = self.samples[index]
        return sample, target, path


def make_loaders():
    """
    Build training and validation DataLoaders for a *pure grayscale* C2/C1 pipeline.

    On disk the datasets may contain RGB or L images; here we enforce a single‑channel
    pipeline end‑to‑end:

      - all input images are resized to (IMAGE_SIZE, IMAGE_SIZE),
      - converted to 1‑channel grayscale,
      - normalized to [-1, 1] as expected by the classifiers.

    Returns:
        train_loader, val_loader, classes_list
    """
    tfm = transforms.Compose([
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.Grayscale(num_output_channels=1),   # enforce single‑channel input
        transforms.ToTensor(),                        # [1,H,W] in [0,1]
        transforms.Normalize([0.5], [0.5]),           # → [-1,1] (1 channel)
    ])

    train_ds = ImageFolderWithPaths(TRAIN_DIR, transform=tfm)
    val_ds   = ImageFolderWithPaths(VAL_DIR,   transform=tfm)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    if train_ds.classes != val_ds.classes:
        raise ValueError(
            f"[classes] Train/Val mismatch:\n"
            f"Train={train_ds.classes}\nVal  ={val_ds.classes}"
        )

    return train_loader, val_loader, train_ds.classes


# =========================
# ---- IMPORT MODELS ------
# =========================

def import_ae_class():
    """
    Dynamically import the external AE class referenced by AE_IMPORT.

    AE_IMPORT is a pair (module_name, class_name).
    Optionally, AE_PY_PATH can extend sys.path to point to a custom location.
    """
    if AE_PY_PATH:
        sys.path.append(AE_PY_PATH)
    mod_name, cls_name = AE_IMPORT
    mod = __import__(mod_name, fromlist=[cls_name])
    return getattr(mod, cls_name)


# Guard‑rail C1 (ResNet34 with flexible number of input channels)
class ResNet34FlexIn(nn.Module):
    """
    Thin wrapper around torchvision.models.resnet34 that allows
    the first convolution to operate on either 1 or 3 channels.

    The classifier head is replaced with a linear layer of size num_classes.
    """
    def __init__(self, num_classes=3, in_ch=3):
        super().__init__()
        m = models.resnet34(weights=None)
        if in_ch != 3:
            m.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        self.m = m

    def forward(self, x):
        return self.m(x)


class GuardC1:
    """
    Frozen C1 classifier with Grad‑CAM support and robust checkpoint loading.

    Functionality:
      * Extracts a state_dict from several typical checkpoint formats.
      * Normalizes keys to the canonical 'm.*' ResNet naming scheme.
      * Locates the classification head and remaps it to 'm.fc.weight/bias'
        (expects [num_classes, 512] features).
      * Infers the number of input channels from 'm.conv1.weight' (1 or 3)
        and instantiates a matching ResNet34FlexIn.
      * Filters out non‑ResNet parameters (e.g. 'rgb2y.*') before loading.
      * Internally adapts inputs:
          - RGB→Y when the checkpoint is 1‑channel,
          - 1→3 replication when the checkpoint is 3‑channel.

    The model is always kept in eval mode and treated as a frozen "guard" classifier.
    """
    def __init__(self, ckpt_path: str, device, expected_num_classes: int = None,
                 dataset_classes: list[str] | None = None, strict_names: bool = True):
        import torch, re

        ckpt = torch.load(ckpt_path, map_location=device)

        # --- 0) strict consistency check between checkpoint classes and dataset classes ---
        if dataset_classes is not None:
            ckpt_classes = None
            if isinstance(ckpt, dict):
                ckpt_classes = ckpt.get("classes") or (
                    ckpt.get("meta", {}) if isinstance(ckpt.get("meta"), dict) else {}
                ).get("classes")
            if ckpt_classes is None:
                raise ValueError("[C1 CHECK] Checkpoint does not contain 'classes' or meta['classes'].")
            if len(ckpt_classes) != len(dataset_classes):
                raise ValueError(
                    f"[C1 CHECK] Number of classes mismatch: "
                    f"ckpt={len(ckpt_classes)} vs data={len(dataset_classes)}"
                )
            if strict_names and list(ckpt_classes) != list(dataset_classes):
                raise ValueError(
                    "[C1 CHECK] Class names / ordering mismatch:\n"
                    f"ckpt={list(ckpt_classes)}\n"
                    f"data={list(dataset_classes)}"
                )
            if expected_num_classes is None:
                expected_num_classes = len(dataset_classes)

        # --- 1) extract a plausible state_dict from various checkpoint formats ---
        sd = None
        if isinstance(ckpt, dict):
            for k in ["model_state", "state_dict", "model", "net", "weights", "params", "model_state_dict"]:
                if k in ckpt and isinstance(ckpt[k], dict):
                    sd = ckpt[k]
                    break
            # bare state_dict form: top‑level dict of tensors
            if sd is None and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
                sd = ckpt
        if sd is None:
            raise ValueError(f"[C1 LOAD] Unsupported checkpoint format: {type(ckpt)}")

        # --- 2) strip 'module.' prefixes introduced by DataParallel ---
        def strip_module(d):
            return {(k[7:] if isinstance(k, str) and k.startswith("module.") else k): v
                    for k, v in d.items()}

        sd = strip_module(sd)

        # Normalize prefixes to 'm.*' while avoiding accidental collisions on layerX.* names
        def to_m_safe(key: str) -> str:
            if key.startswith("m."):
                return key
            m = re.search(r'(layer[1-4]\..*)', key)
            if m:
                return "m." + m.group(1)
            # Stem layers (conv1/bn1/fc) optionally preceded by a short backbone prefix
            stem_prefix = r'(?:(?:backbone|encoder|model|net|resnet|features|feature_extractor|body|trunk|wrapper|rgbwrap|stem)\.)?'
            for sub in ("conv1.", "bn1.", "fc."):
                if re.match(rf'^{stem_prefix}{re.escape(sub)}', key):
                    pos = key.find(sub)
                    return "m." + key[pos:]
            # Otherwise keep as is; may be filtered later
            return key

        sd_norm: Dict[str, torch.Tensor] = {}
        for k, v in sd.items():
            nk = to_m_safe(k)
            if nk.startswith("m.m."):  # guard against double 'm.'
                nk = nk[2:]
            sd_norm[nk] = v
        sd = sd_norm

        # --- 2a) explicit resolution of the stem conv1 (avoid picking intra‑block conv1) ---
        def _is_stem_conv_key(k: str) -> bool:
            return k == "m.conv1.weight"

        if "m.conv1.weight" not in sd or sd["m.conv1.weight"].ndim != 4:
            # candidate 4D kernels with shape [64, *, 7, 7] that are NOT inside 'layerX'
            stem_cands = []
            for k, v in sd.items():
                if not (isinstance(k, str) and k.endswith(".weight")):
                    continue
                if getattr(v, "ndim", 0) == 4 and v.shape[2:] == (7, 7) and v.shape[0] == 64 and ("layer" not in k):
                    stem_cands.append((k, v))
            # prefer explicit 'm.conv1.weight' if already present
            pref = [kv for kv in stem_cands if kv[0] == "m.conv1.weight"]
            if pref:
                sd["m.conv1.weight"] = pref[0][1]
            elif stem_cands:
                k_best, v_best = stem_cands[0]
                sd["m.conv1.weight"] = v_best
            else:
                raise ValueError(
                    "[C1 LOAD] Could not locate stem conv1 kernel (7×7, out=64). "
                    "Checkpoint appears incompatible with ResNet‑34."
                )

            self.in_ch = int(sd["m.conv1.weight"].shape[1])
            if self.in_ch not in (1, 3):
                raise ValueError(
                    f"[C1 LOAD] Invalid stem conv1 in_channels={self.in_ch}; expected 1 or 3."
                )

            # --- 2b) ensure that stem BatchNorm (bn1) is valid and present ---
            bn_keys = {
                "weight": None,
                "bias": None,
                "running_mean": None,
                "running_var": None,
            }
            # candidate BN1 keys outside residual layers, with 64 features
            for k, v in sd.items():
                if not isinstance(k, str):
                    continue
                if "layer" in k:
                    continue
                if k.endswith("bn1.weight") and getattr(v, "ndim", 0) == 1 and v.numel() == 64:
                    bn_keys["weight"] = k
                elif k.endswith("bn1.bias") and getattr(v, "ndim", 0) == 1 and v.numel() == 64:
                    bn_keys["bias"] = k
                elif k.endswith("bn1.running_mean") and getattr(v, "ndim", 0) == 1 and v.numel() == 64:
                    bn_keys["running_mean"] = k
                elif k.endswith("bn1.running_var") and getattr(v, "ndim", 0) == 1 and v.numel() == 64:
                    bn_keys["running_var"] = k

            for name, src_k in bn_keys.items():
                if src_k is not None and f"m.bn1.{name}" not in sd:
                    sd[f"m.bn1.{name}"] = sd[src_k]

        # --- 3) locate the classification head and normalize it to m.fc.* ---
        def locate_head(d, expected_nc: int | None):
            cand = []
            for k, v in d.items():
                if k.endswith(".weight") and getattr(v, "ndim", 0) == 2:
                    base = k[:-7]
                    b = base + ".bias"
                    if b in d:
                        of, inf = v.shape
                        score = (inf == 512) * 2 + (expected_nc is not None and of == expected_nc) * 3
                        cand.append((score, k, b, of, inf))
            if not cand:
                return None
            cand.sort(key=lambda t: t[0], reverse=True)
            return cand[0]

        head = locate_head(sd, expected_num_classes)
        if head is None:
            raise ValueError("[C1 LOAD] Could not find a suitable classifier head (fc.*).")
        _, wkey, bkey, of, inf = head
        if expected_num_classes is not None and of != expected_num_classes:
            raise ValueError(
                f"[C1 LOAD] Number of output classes in checkpoint={of} "
                f"does not match expected={expected_num_classes}."
            )
        if inf != 512:
            raise ValueError(
                f"[C1 LOAD] Expected fc in_features=512 (ResNet‑34), got {inf}."
            )
        if (wkey, bkey) != ("m.fc.weight", "m.fc.bias"):
            sd["m.fc.weight"] = sd[wkey]
            sd["m.fc.bias"] = sd[bkey]
            sd.pop(wkey, None)
            sd.pop(bkey, None)

        # --- 4) input channels inferred from conv1 ---
        if "m.conv1.weight" not in sd:
            raise ValueError("[C1 LOAD] 'm.conv1.weight' not found in checkpoint.")
        self.in_ch = int(sd["m.conv1.weight"].shape[1])
        if self.in_ch not in (1, 3):
            raise ValueError(
                f"[C1 LOAD] Unsupported number of input channels: {self.in_ch} (expected 1 or 3)."
            )

        # --- 5) instantiate model and load a filtered state_dict ---
        if expected_num_classes is None:
            expected_num_classes = int(sd["m.fc.weight"].shape[0])
        self.model = ResNet34FlexIn(num_classes=expected_num_classes, in_ch=self.in_ch).to(device)

        model_keys = set(self.model.state_dict().keys())
        sd_filtered = {k: v for k, v in sd.items() if k in model_keys}

        # Fill missing BN1 entries (if any) with defaults from the freshly created model
        mstate = self.model.state_dict()
        for k in [
            "m.bn1.weight",
            "m.bn1.bias",
            "m.bn1.running_mean",
            "m.bn1.running_var",
            "m.bn1.num_batches_tracked",
        ]:
            if k in mstate and k not in sd_filtered:
                sd_filtered[k] = mstate[k]

        missing, unexpected = self.model.load_state_dict(sd_filtered, strict=True)
        if missing or unexpected:
            raise ValueError(f"[C1 LOAD] missing={missing} | unexpected={unexpected}")

        # store class names if available
        if dataset_classes is not None:
            self.classes = list(dataset_classes)
        else:
            self.classes = ckpt.get("classes", [f"class{i}" for i in range(expected_num_classes)])

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Grad‑CAM hooks (last conv block)
        self._feat = None
        self._grad = None
        last_conv = self.model.m.layer4[-1].conv2
        last_conv.register_forward_hook(self._fhook)
        last_conv.register_full_backward_hook(self._bhook)

    # --- CAM utils ---

    def _fhook(self, _, __, out):
        self._feat = out

    def _bhook(self, _, grad_in, grad_out):
        self._grad = grad_out[0]

    # --- input preprocessing to match checkpoint channels ---

    def _prep_in(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adapt the input tensor to the number of channels expected by C1.

        If the model expects 1 channel and receives 3, RGB is converted to luma (Y).
        If the model expects 3 channels and receives 1, the channel is replicated.
        """
        if self.in_ch == x.size(1):
            return x
        if self.in_ch == 1 and x.size(1) == 3:
            r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
            return 0.2126 * r + 0.7152 * g + 0.0722 * b  # RGB→Y
        if self.in_ch == 3 and x.size(1) == 1:
            return x.repeat(1, 3, 1, 1)
        raise ValueError(
            f"[C1 PREP] Incompatible channels: model expects {self.in_ch}, got {x.size(1)}"
        )

    def logits(self, x):
        """Compute raw logits of C1 on input x, with channel adaptation if needed."""
        x_in = self._prep_in(x)
        return self.model(x_in)

    def cam(self, x_normed):
        """
        Compute a Grad‑CAM saliency map over the input.

        Input:
            x_normed : [B, C, H, W] normalized to [-1, 1].

        Output:
            saliency map of shape [B, 1, IMAGE_SIZE, IMAGE_SIZE] in [0, 1].
        """
        import torch.nn.functional as F
        with torch.enable_grad():
            x = x_normed.detach().clone().requires_grad_(True)
            x_in = self._prep_in(x)

            logits = self.model(x_in)
            target = logits.argmax(dim=1)
            score = logits[torch.arange(logits.size(0)), target].sum()

            self.model.zero_grad(set_to_none=True)
            score.backward(retain_graph=False)

            A = self._feat
            G = self._grad
            w = G.mean(dim=(2, 3), keepdim=True)
            cam = (A * w).sum(dim=1, keepdim=True)
            cam = F.relu(cam)
            cam = F.interpolate(
                cam,
                size=(IMAGE_SIZE, IMAGE_SIZE),
                mode="bilinear",
                align_corners=False,
            )
            cam = (cam - cam.amin(dim=(2, 3), keepdim=True)) / (
                cam.amax(dim=(2, 3), keepdim=True) - cam.amin(dim=(2, 3), keepdim=True) + 1e-8
            )
            return cam.detach()


# =========================
# --- mini-U-Net & heads ---
# =========================

class ConvGNAct(nn.Module):
    """Conv → GroupNorm → GELU block used as a basic building unit."""
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.gn   = nn.GroupNorm(num_groups=min(32, out_ch), num_channels=out_ch)
        self.act  = nn.GELU()

    def forward(self, x):
        return self.act(self.gn(self.conv(x)))


class MiniUNetMask(nn.Module):
    """
    A small U‑Net‑style network that produces a soft mask over spatial features.

    It is used to estimate per‑pixel importance over AE latent and skip‑64
    maps. Output is constrained to [0, 1] via a final sigmoid.
    """
    def __init__(self, in_ch, base=32):
        super().__init__()
        c1, c2, c3 = base, base * 2, base * 4
        self.d1 = nn.Sequential(ConvGNAct(in_ch, c1), ConvGNAct(c1, c1))
        self.p1 = nn.MaxPool2d(2)
        self.d2 = nn.Sequential(ConvGNAct(c1, c2), ConvGNAct(c2, c2))
        self.p2 = nn.MaxPool2d(2)
        self.bott = nn.Sequential(ConvGNAct(c2, c3), ConvGNAct(c3, c3))
        self.u2 = nn.ConvTranspose2d(c3, c2, 2, 2)
        self.u1 = nn.ConvTranspose2d(c2, c1, 2, 2)
        self.up2 = nn.Sequential(ConvGNAct(c3, c2), ConvGNAct(c2, c2))
        self.up1 = nn.Sequential(ConvGNAct(c2, c1), ConvGNAct(c1, c1))
        self.out = nn.Conv2d(c1, 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, feat):
        x1 = self.d1(feat)
        x2 = self.d2(self.p1(x1))
        x3 = self.bott(self.p2(x2))
        x  = self.u2(x3)
        x  = torch.cat([x, x2], dim=1)
        x  = self.up2(x)
        x  = self.u1(x)
        x  = torch.cat([x, x1], dim=1)
        x  = self.up1(x)
        return torch.sigmoid(self.out(x))


class GLat(nn.Module):
    """
    Latent‑space generator head for the AE latent [B, LAT_CH, 32, 32].

    The head predicts a signed perturbation pattern and applies a small spatial
    low‑pass filter. A Laplacian of the AE latent is subtracted to discourage
    degenerate high‑frequency artifacts.
    """
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            ConvGNAct(ch, ch),
            ConvGNAct(ch, ch),
            nn.Conv2d(ch, ch, 3, 1, 1),
            nn.Tanh()
        )
        nn.init.zeros_(self.net[-2].weight)
        nn.init.zeros_(self.net[-2].bias)
        # 3×3 average smoothing to reduce speckle in the generated pattern
        self.avg = nn.AvgPool2d(3, 1, 1)

    @staticmethod
    def lap_kernel(device, dtype=torch.float32):
        """3×3 Laplacian kernel used to penalize high‑frequency components."""
        k = torch.tensor([[0, -1, 0],
                          [-1, 4, -1],
                          [0, -1, 0]], dtype=dtype, device=device).view(1, 1, 3, 3)
        return k

    def forward(self, z):
        # Base perturbation in [-1, 1]
        p = self.net(z)
        p = self.avg(p)
        k = self.lap_kernel(z.device, dtype=z.dtype).repeat(z.shape[1], 1, 1, 1)
        lap = F.conv2d(z, weight=k, padding=1, groups=z.shape[1])
        # Return a smoothed pattern with a small Laplacian correction
        return self.avg(p) - 0.02 * lap


class GDCT64(nn.Module):
    """
    Skip‑64 generator head operating in the DCT domain.

    Pipeline:
        1) Project the skip feature map to a single channel.
        2) Apply a 2D DCT.
        3) Mask out high frequencies, keep only low/mid‑frequency band.
        4) Inverse DCT + mild 3×3 spatial smoothing.

    This encourages smooth, low‑frequency watermark patterns in the skip‑64 space.
    """
    def __init__(self, ch_in, h=64, w=64, keep_ratio=DCT_KEEP_RATIO, dc_scale=DC_SCALE):
        super().__init__()
        self.proj = nn.Conv2d(ch_in, 1, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.h, self.w = h, w
        self.register_buffer("B", self._make_lowmid_mask(h, w, keep_ratio, dc_scale))
        self.dct = DCTCache(h, w)

    @staticmethod
    def _make_lowmid_mask(h: int, w: int, keep_ratio: float, dc_scale: float):
        """
        Construct a low/mid‑frequency mask with elliptical cut‑off and explicit
        scaling for the DC component.
        """
        yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        cy, cx = (h - 1) / 2, (w - 1) / 2
        rr = torch.sqrt(((yy - cy) / h) ** 2 + ((xx - cx) / w) ** 2)
        keep = (rr <= keep_ratio / 2).to(torch.float32)
        keep[0, 0] = dc_scale
        return keep  # [H,W]

    def forward(self, s64):
        # We explicitly disable autocast inside DCT to avoid dtype issues
        with torch.autocast(device_type="cuda", enabled=False):
            u = self.proj(s64.float()).squeeze(1)
            U = self.dct.dct2(u)
            B = self.B.to(U.device, U.dtype)
            U_f = U * B
            w64 = self.dct.idct2(U_f).unsqueeze(1)
            w64 = F.avg_pool2d(w64, kernel_size=3, stride=1, padding=1)
        return w64.to(dtype=s64.dtype)


# =========================
# ------- C2 MODEL --------
# =========================

class BlurPool(nn.Module):
    """
    Simple blur‑then‑downsample block used to replace stride‑2 convolutions.

    This avoids aliasing artifacts when reducing spatial resolution in deeper
    stages of the network.
    """
    def __init__(self, ch, filt=(1, 2, 1)):
        super().__init__()
        f = torch.tensor(filt, dtype=torch.float32)
        k = (f[:, None] * f[None, :])
        k = (k / k.sum()).view(1, 1, 3, 3).repeat(ch, 1, 1, 1)
        self.register_buffer("k", k)
        self.groups = ch

    def forward(self, x):
        return F.conv2d(x, self.k, stride=2, padding=1, groups=self.groups)


class ResNet34LF(nn.Module):
    """
    ResNet‑34 for grayscale classification with a low‑frequency companion stem
    and a watermark logit head.

    Key modifications relative to canonical ResNet‑34:
      * Input is 1‑channel; conv1 is adapted accordingly.
      * Stem stride is set to 1 for higher initial resolution.
      * Stride‑2 convolutions in layer2/3/4 are replaced with a combination of
        stride‑1 convs and BlurPool to reduce aliasing.
      * A low‑frequency branch operates on the original grayscale input and is
        fused into the early ResNet features.
      * A scalar watermark head operates on the last feature map and its output
        is used to gate class logits via a learned affine term.
    """
    def __init__(self, num_classes: int):
        super().__init__()
        base = models.resnet34(weights=None)

        # Stem for 1‑channel input
        old_conv1 = base.conv1
        base.conv1 = nn.Conv2d(
            1,
            old_conv1.out_channels,
            kernel_size=old_conv1.kernel_size,
            stride=(1, 1),          # stem stride=1
            padding=old_conv1.padding,
            bias=(old_conv1.bias is not None),
        )
        nn.init.kaiming_normal_(base.conv1.weight, mode="fan_out", nonlinearity="relu")
        if base.conv1.bias is not None:
            nn.init.zeros_(base.conv1.bias)

        self.base = base

        # Replace stride‑2 in layer2/3/4 with BlurPool downsampling
        self._wrap_blur(base.layer2)
        self._wrap_blur(base.layer3)
        self._wrap_blur(base.layer4)

        # Low‑frequency branch at the input level; fused after layer1
        self.lpf = nn.AvgPool2d(3, 1, 1)
        self.lf_stem = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1, bias=False),
            nn.GroupNorm(8, 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, 1, 1, bias=False),
            nn.GroupNorm(8, 16),
            nn.ReLU(inplace=True),
        )
        self.fuse1 = nn.Conv2d(64 + 16, 64, 1, 1, 0, bias=False)

        # Watermark head at the last feature map
        self.wm_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

        # Classification head
        self.base.fc = nn.Linear(self.base.fc.in_features, num_classes)

        # Affine shift of logits conditioned on wm_logit
        self.wm_affine = nn.Parameter(torch.zeros(num_classes))

    @staticmethod
    def _wrap_blur(layer):
        """
        Replace stride‑2 downsample convolutions inside ResNet blocks with
        stride‑1 + BlurPool to avoid aliasing.
        """
        for b in layer:
            if b.downsample is not None:
                seq = []
                for sm in b.downsample:
                    if isinstance(sm, nn.Conv2d) and sm.stride == (2, 2):
                        sm.stride = (1, 1)
                        seq += [sm, BlurPool(sm.out_channels)]
                    else:
                        seq.append(sm)
                b.downsample = nn.Sequential(*seq)

    def forward(self, x, gate: bool = True, detach_gate: bool = False,
                detach_affine: bool = False, return_raw: bool = False):
        """
        Forward pass.

        Args:
            x:  [B, 1, H, W] grayscale input in [-1, 1].
            gate: if True, add watermark‑dependent affine offset to logits.
            detach_gate: if True, detach wm_logit from gradients before gating.
            detach_affine: if True, detach the wm_affine parameter in gating.
            return_raw: if True, also return the ungated logits_raw.

        Returns:
            logits (optionally gated), wm_logit, and optionally logits_raw.
        """
        b = self.base
        x0 = b.relu(b.bn1(b.conv1(x)))
        x0 = b.maxpool(x0)
        x1 = b.layer1(x0)

        # Low‑frequency branch: process raw grayscale input and fuse into x1
        xl = self.lpf(x)
        xl = self.lf_stem(xl)
        xl = F.interpolate(xl, size=x1.shape[-2:], mode='bilinear', align_corners=False)

        x1f = self.fuse1(torch.cat([x1, xl], dim=1))
        x2 = b.layer2(x1f)
        x3 = b.layer3(x2)
        x4 = b.layer4(x3)

        pooled = b.avgpool(x4).view(x4.size(0), -1)
        logits_raw = self.base.fc(pooled)          # clean logits
        wm_logit = self.wm_head(x4).squeeze(1)     # [B]

        logits = logits_raw
        if gate:
            g = torch.tanh(wm_logit).unsqueeze(1)  # [-1, 1]
            if detach_gate:
                g = g.detach()
            w = self.wm_affine.view(1, -1)
            if detach_affine:
                w = w.detach()
            logits = logits_raw + g * w

        if return_raw:
            return logits, wm_logit, logits_raw
        return logits, wm_logit, x4


# =========================
# -------- TRAINER --------
# =========================

@dataclass
class ControllerState:
    """
    State of the global watermark controller.

    eps   : scalar ε controlling the overall perturbation magnitude.
    r_skip: fraction of ε routed through skip‑64 (latent gets 1 − r_skip).
    ema_delta: EMA of Δ(C2) (soft probability difference), used for stability.
    """
    eps: float = 0.0
    r_skip: float = RSKIP_INIT
    ema_delta: float = 0.0


class WatermarkTrainer:
    """
    High‑level trainer encapsulating:
      * frozen C1 (GuardC1) used for CAM‑based ROI selection and stability checks;
      * frozen AE with external watermark injection API;
      * generator G (ROI masks + latent/skip heads) trained to embed a robust watermark;
      * discriminator/classifier C2 with an explicit watermark head;
      * EMA copy of C2 for more stable Δ(C2) measurements;
      * controller managing ε and r_skip based on perceptual and classification metrics.
    """
    def __init__(self):
        # === Data ===
        self.train_loader, self.val_loader, self.classes = make_loaders()
        self.num_classes = len(self.classes)

        # Strict consistency between C1 checkpoint classes and dataset classes
        ckpt_classes = read_ckpt_classes(C1_CKPT)
        if len(ckpt_classes) != len(self.classes):
            raise ValueError(
                f"[C1 CHECK] Number of classes mismatch: ckpt={len(ckpt_classes)} "
                f"vs data={len(self.classes)}"
            )
        if ckpt_classes != self.classes:
            raise ValueError(
                "[C1 CHECK] Names or order of classes mismatch:\n"
                f"ckpt={ckpt_classes}\n"
                f"data={self.classes}"
            )

        print(f"[data ] classes ({self.num_classes}): {self.classes}")

        # Persist class names for reproducibility
        cls_path = Path(OUT_ROOT) / "classes.txt"
        cls_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cls_path, "w", encoding="utf-8") as f:
            for c in self.classes:
                f.write(c + "\n")

        # === C1: frozen guard classifier with auto channel adaptation ===
        self.c1 = GuardC1(
            C1_CKPT,
            DEVICE,
            expected_num_classes=len(self.classes),
            dataset_classes=self.classes,
            strict_names=True,
        )

        # === AE: frozen universal autoencoder with external WM API ===
        AEClass = import_ae_class()
        self.ae = AEClass().to(DEVICE)

        # Load AE checkpoint robustly with several fallback formats
        chk = torch.load(AE_PTH, map_location=DEVICE)
        if isinstance(chk, dict) and "state_dict" in chk and isinstance(chk["state_dict"], dict):
            sd = chk["state_dict"]
        elif isinstance(chk, dict) and "uae" in chk and isinstance(chk["uae"], dict):
            sd = chk["uae"]
        elif isinstance(chk, dict) and "enc" in chk and "dec" in chk \
                and isinstance(chk["enc"], dict) and isinstance(chk["dec"], dict):
            sd = {}
            for k, v in chk["enc"].items():
                sd[f"enc.{k}"] = v
            for k, v in chk["dec"].items():
                sd[f"dec.{k}"] = v
        else:
            sd = chk if isinstance(chk, dict) else chk

        sd_fixed = {}
        for k, v in sd.items():
            nk = k[7:] if isinstance(k, str) and k.startswith("module.") else k
            sd_fixed[nk] = v
        missing, unexpected = self.ae.load_state_dict(sd_fixed, strict=False)
        if missing or unexpected:
            print("[AE LOAD] missing keys:", missing)
            print("[AE LOAD] unexpected keys:", unexpected)
        self.ae.eval()
        for p in self.ae.parameters():
            p.requires_grad_(False)

        # === Generator components (latent, skip‑64, ROI masks) ===
        self.lat_ch = LAT_CH
        self.s64_ch = S64_CH
        self.mask_lat = MiniUNetMask(in_ch=self.lat_ch, base=32).to(DEVICE)
        self.mask_64  = MiniUNetMask(in_ch=self.s64_ch,  base=32).to(DEVICE)
        self.g_lat    = GLat(self.lat_ch).to(DEVICE)
        self.g_64     = GDCT64(self.s64_ch, h=64, w=64, keep_ratio=DCT_KEEP_RATIO,
                               dc_scale=DC_SCALE).to(DEVICE)
        # 1→512 projection for skip‑64 to feed full‑channel wm_skip, frozen
        self.g_64_proj = nn.Conv2d(1, self.s64_ch, kernel_size=1, bias=False).to(DEVICE)
        with torch.no_grad():
            # uniform spreading across channels; frozen to avoid "grainy" artifacts
            self.g_64_proj.weight.fill_(1.0 / math.sqrt(self.s64_ch))
        for p in self.g_64_proj.parameters():
            p.requires_grad_(False)

        self.lowtex   = LocalVariance(k=7).to(DEVICE)

        # === Discriminator / classifier C2 (low‑freq + watermark head) ===
        self.c2 = ResNet34LF(num_classes=len(self.classes)).to(DEVICE)

        # EMA copy of C2 for stable Δ(C2) statistics
        self.c2_ema = ResNet34LF(num_classes=len(self.classes)).to(DEVICE)
        self.c2_ema.load_state_dict(self.c2.state_dict(), strict=True)
        for p in self.c2_ema.parameters():
            p.requires_grad_(False)

        self.ema_tau = 0.995  # EMA smoothing factor; 0.99–0.999 are typical

        # === Optimizers ===
        self.opt_g  = torch.optim.AdamW(
            list(self.mask_lat.parameters()) + list(self.mask_64.parameters()) +
            list(self.g_lat.parameters())    + list(self.g_64.parameters()),
            lr=LR_G, weight_decay=WEIGHT_DECAY
        )
        self.opt_c2 = torch.optim.AdamW(self.c2.parameters(), lr=LR_C2, weight_decay=WEIGHT_DECAY)

        self.scaler = torch.amp.GradScaler("cuda", enabled=MIXED_PRECISION)
        self.ctrl   = ControllerState()
        self.current_epoch = 1

    # ---------- helpers ----------

    @staticmethod
    def _to01(x_normed: torch.Tensor) -> torch.Tensor:
        """Map normalized inputs from [-1, 1] back to [0, 1]."""
        return (x_normed * 0.5 + 0.5).clamp(0, 1)

    def ae_features(self, x01: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract AE features for grayscale luminance in [0, 1].

        x01: [B,1,H,W] or [B,3,H,W] in [0,1].
        For RGB, we derive luminance Y first; for grayscale we reuse the channel.
        Returns a dictionary with keys:
            'latent' : [B, LAT_CH, 32, 32]
            's64'    : [B, S64_CH, 64, 64]
        """
        if x01.size(1) == 3:
            y = 0.299 * x01[:, 0:1] + 0.587 * x01[:, 1:2] + 0.114 * x01[:, 2:3]
        else:
            y = x01
        with torch.no_grad():
            s = self.ae.enc(y)
        return s  # dict with keys 'latent', 's64'

    @torch.no_grad()
    def _ema_update(self, ema_model: nn.Module, src_model: nn.Module, tau: float):
        """
        Exponential moving average update for model parameters and buffers.

        ema = tau * ema + (1 - tau) * src

        Only floating‑point tensors are updated; integer buffers such as
        'num_batches_tracked' are left unchanged to avoid numerical issues.
        """
        ema_sd = ema_model.state_dict()
        src_sd = src_model.state_dict()
        for k, v_src in src_sd.items():
            v_ema = ema_sd.get(k, None)
            if v_ema is None:
                continue
            if not torch.is_floating_point(v_ema) or not torch.is_floating_point(v_src):
                continue
            if v_ema.dtype != v_src.dtype:
                v_src = v_src.to(dtype=v_ema.dtype)
            if v_ema.device != v_src.device:
                v_src = v_src.to(device=v_ema.device)
            v_ema.mul_(tau).add_(v_src, alpha=1.0 - tau)

    # ---------- ROI helpers ----------

    @staticmethod
    def topk_binary_mask(Q: torch.Tensor, keep_frac: float = 0.33) -> torch.Tensor:
        """
        Convert a soft saliency map Q ∈ [0,1]^{B×1×H×W} into a hard binary mask.

        The mask keeps the top 'keep_frac' fraction of pixels per sample.
        """
        B, _, H, W = Q.shape
        k = max(1, int(H * W * keep_frac))
        flat = Q.view(B, 1, -1)
        topk_vals, _ = torch.topk(flat, k, dim=2)
        thr = topk_vals[:, :, -1].view(B, 1, 1, 1)
        return (Q >= thr).float()

    @staticmethod
    def border_ring(h: int, w: int, inner: float = 0.06, outer: float = 0.20,
                    device=None, dtype=None) -> torch.Tensor:
        """
        Construct a binary "ring" mask near the image borders.

        Pixels whose normalized distance to the closest border lies in
        [inner, outer] are set to 1, others to 0.
        """
        yy = torch.linspace(0, 1, h, device=device, dtype=dtype).view(h, 1)
        xx = torch.linspace(0, 1, w, device=device, dtype=dtype).view(1, w)
        d = torch.minimum(torch.minimum(yy, 1 - yy), torch.minimum(xx, 1 - xx))
        ring = ((d <= outer) & (d >= inner)).float().view(1, 1, h, w)
        return ring

    def _alphas_from_ctrl(self):
        """
        Compute (alpha_lat, alpha_skip) from the current controller state.

        Enforces:
          * r_skip_eff ≤ 0.67  → latent carries ≥ 33% of the total energy.
          * r_lat_eff  ≥ 0.33  → minimal latent contribution.

        Returns:
            alpha_lat, alpha_skip, r_lat_eff, r_skip_eff
        """
        r_skip_eff = float(min(self.ctrl.r_skip, 0.67))
        r_lat_eff = float(max(1.0 - r_skip_eff, 0.33))

        alpha_lat = float(self.ctrl.eps) * r_lat_eff
        alpha_skip = float(self.ctrl.eps) * r_skip_eff
        return alpha_lat, alpha_skip, r_lat_eff, r_skip_eff

    # ---------- masks ----------

    def build_masks(self, Z, S64, x_in, S_lat, S_64):
        """
        Build ROI masks for latent and skip‑64 spaces, combining several cues:

          * C1 Grad‑CAM saliency maps S_lat (32×32) and S_64 (64×64);
          * local texture estimates N_32 / N_64 (preferring non‑flat regions);
          * learned soft masks from MiniUNetMask (mask_lat/mask_64);
          * hard top‑k selection and a border ring prior;
          * light morphological cleanup.

        Returns:
            P_lat, P_64, N_32, N_64, M_lat, M_64
        """
        # x_in normalized [-1,1]; build low‑texture maps on luminance in [0,1]
        x01 = self._to01(x_in)
        if x01.size(1) == 3:
            Y = 0.2989 * x01[:, 0:1] + 0.5870 * x01[:, 1:2] + 0.1140 * x01[:, 2:3]
        else:
            Y = x01  # already grayscale

        N_64 = self.lowtex(F.interpolate(Y, size=(64, 64), mode='bilinear', align_corners=False))
        N_32 = self.lowtex(F.interpolate(Y, size=(32, 32), mode='bilinear', align_corners=False))

        M_lat = self.mask_lat(Z)      # [B,1,32,32]
        M_64  = self.mask_64(S64)     # [B,1,64,64]

        # Base soft priority combining CAM and texture
        w1, w2 = 0.8, 0.2
        P_lat = torch.clamp(w1 * (1.0 - S_lat) + w2 * N_32, 0, 1) * M_lat
        P_64  = torch.clamp(w1 * (1.0 - S_64) + w2 * N_64, 0, 1) * M_64

        # Hard CAM gating: suppress areas with high C1 saliency above a threshold
        tau_lat, tau_64 = 0.60, 0.60
        hard_lat = (S_lat <= tau_lat).float()
        hard_64  = (S_64 <= tau_64).float()

        gamma = 3.0  # sharpen (1 - S)^gamma for stronger CAM exclusion
        P_lat = P_lat * hard_lat * (1.0 - S_lat).pow(gamma)
        P_64  = P_64  * hard_64  * (1.0 - S_64).pow(gamma)

        # Top‑k selection using a combined score Q that prefers low CAM & low texture
        Q_64  = (1.0 - S_64).pow(2.0) * (1.0 - N_64)
        Q_lat = (1.0 - S_lat).pow(2.0) * (1.0 - N_32)

        hard64 = self.topk_binary_mask(Q_64, keep_frac=0.38)
        hard32 = self.topk_binary_mask(Q_lat, keep_frac=0.25)

        # Border ring encourages watermark to live slightly away from the image center
        ring64 = self.border_ring(64, 64, inner=0.08, outer=0.32,
                                  device=S_64.device, dtype=S_64.dtype)
        ring32 = F.interpolate(ring64, size=(32, 32), mode="nearest")

        # Morphological opening/closing to remove tiny speckles
        def morph_open_close(m, k=3, iters=1):
            for _ in range(iters):
                m = F.max_pool2d(m, k, 1, k // 2)      # dilate
                m = -F.max_pool2d(-m, k, 1, k // 2)    # erode
            return m

        hard64 = morph_open_close(hard64, k=3, iters=1)
        hard32 = morph_open_close(hard32, k=3, iters=1)

        # Apply all constraints at once
        P_64  = P_64  * hard64 * ring64
        P_lat = P_lat * hard32 * ring32

        # Suppress extremely smooth regions in skip‑64
        with torch.no_grad():
            ultra_smooth = (N_64 < 0.20).float()
        P_64 = P_64 * (1.0 - 0.5 * ultra_smooth)

        return P_lat, P_64, N_32, N_64, M_lat, M_64

    # ---------- generator step ----------

    def step_generator(self, batch, epoch: int, it: int, out_root, *, log_images: bool = True):
        """
        Single generator optimization step (no C2 update).

        The generator:
          * queries AE taps (latent and skip‑64) on the clean input;
          * constructs ROI masks using C1 CAM and texture;
          * synthesizes latent and skip‑64 patterns and normalizes their energy;
          * embeds them via AE.embed_external_wm_gray;
          * computes a mixture of leak/TV/sparsity losses and C1‑consistency terms;
          * updates generator parameters by backpropagating through AE.

        Returns a dictionary of scalars and (optionally) tensors for logging.
        """
        import torch.nn.functional as F

        DEV = next(self.g_lat.parameters()).device
        self.mask_lat.train()
        self.mask_64.train()
        self.g_lat.train()
        self.g_64.train()

        # ---------- data ----------
        x, y, paths = batch
        x = x.to(DEV)
        y = y.to(DEV)
        x_ae = self._to01(x)                 # [B,1,H,W] in [0,1]
        B, C, H, W = x_ae.shape

        # ---------- AE "no‑op" baseline (watermark off) ----------
        with torch.no_grad():
            y_base = self.ae.embed_external_wm_gray(
                x_ae, wm_lat=None, wm_skip=None
            ).clamp(0, 1)
            y_base = torch.nan_to_num(y_base, nan=0.0, posinf=1.0, neginf=0.0)

        # ---------- AE taps & ROI ----------
        taps = self.ae_features(x_ae)
        Z_true, S64_true = taps["latent"], taps["s64"]

        S = self.c1.cam(x)
        S_64 = F.interpolate(S, size=(64, 64), mode='bilinear', align_corners=False)
        S_lat = F.interpolate(S, size=(32, 32), mode='bilinear', align_corners=False)

        P_lat, P_64, *_ = self.build_masks(Z_true, S64_true, x, S_lat, S_64)

        # ---------- generator patterns ----------
        z_pat = self.g_lat(Z_true)       # [B,*,32,32]
        w64_1 = self.g_64(S64_true)      # [B,1,64,64]
        w64   = self.g_64_proj(w64_1)    # [B,*,64,64]

        # constrain support to ROI
        z_pat = z_pat * P_lat
        w64   = w64 * P_64

        # --- energy normalization of patterns (unit RMS inside ROI) ---
        def _unit_rms(t, mask):
            t = torch.nan_to_num(t, nan=0.0)
            denom = mask.sum(dim=(1, 2, 3), keepdim=True).clamp_min(1.0)
            rms = torch.sqrt(((t ** 2) * mask).sum(dim=(1, 2, 3), keepdim=True) / denom + 1e-12)
            return t / rms

        z_pat = _unit_rms(z_pat, P_lat)
        w64   = _unit_rms(w64, P_64)

        # ---------- ratios / normalization with latent ≥ 33% ----------
        alpha_lat, alpha_skip, r_lat_eff, r_skip_eff = self._alphas_from_ctrl()
        eps = 1e-6
        area_lat = (P_lat.flatten(1).sum(1) + eps)
        area_64  = (P_64.flatten(1).sum(1) + eps)
        A_lat, A_64 = 32 * 32, 64 * 64
        alpha_lat_b  = alpha_lat * (A_lat / area_lat).sqrt().view(B, 1, 1, 1)
        alpha_skip_b = alpha_skip * (A_64 / area_64).sqrt().view(B, 1, 1, 1)

        # damping when ROI heavily overlaps C1 CAM peaks
        with torch.no_grad():
            overlap_proxy = (P_lat * S_lat).mean() + (P_64 * S_64).mean()
        if float(overlap_proxy) > 0.05:
            scale = max(0.5, 1.0 - 2.5 * (float(overlap_proxy) - 0.04))
            alpha_lat_b  = alpha_lat_b * scale
            alpha_skip_b = alpha_skip_b * scale

        # ---------- embed watermark (gray) ----------
        x_wm_pre = self.ae.embed_external_wm_gray(
            x_ae, wm_lat=z_pat, wm_skip=w64,
            alpha_lat=alpha_lat_b, alpha_skip=alpha_skip_b,
            roi_lat_32=P_lat, roi_skip_64=P_64
        ).clamp(0, 1)
        x_wm_pre = torch.nan_to_num(x_wm_pre, nan=0.0, posinf=1.0, neginf=0.0)

        # ---------- kickstart boost if perturbation is too small at epoch 1 ----------
        with torch.no_grad():
            delta_abs = (x_wm_pre - y_base).abs().mean().item()
        if epoch == 1 and delta_abs < 0.0025:
            boost = min(4.0, 0.0040 / max(delta_abs, 1e-6))
            x_wm_pre = self.ae.embed_external_wm_gray(
                x_ae, wm_lat=z_pat, wm_skip=w64,
                alpha_lat=alpha_lat_b * boost, alpha_skip=alpha_skip_b * boost,
                roi_lat_32=P_lat, roi_skip_64=P_64
            ).clamp(0, 1)
            x_wm_pre = torch.nan_to_num(x_wm_pre, nan=0.0, posinf=1.0, neginf=0.0)

        x_wm = x_wm_pre  # final watermarked reconstruction

        # ---------- Shapley‑style attribution for latent vs skip contributions ----------
        with torch.no_grad():
            x_lat_pre = self.ae.embed_external_wm_gray(
                x_ae, wm_lat=z_pat, wm_skip=None,
                alpha_lat=alpha_lat_b, alpha_skip=None,
                roi_lat_32=P_lat, roi_skip_64=None
            ).clamp(0, 1)
            x_skip_pre = self.ae.embed_external_wm_gray(
                x_ae, wm_lat=None, wm_skip=w64,
                alpha_lat=None, alpha_skip=alpha_skip_b,
                roi_lat_32=None, roi_skip_64=P_64
            ).clamp(0, 1)

            d_lat  = 0.5 * ((x_wm - x_skip_pre) + (x_lat_pre - y_base))
            d_skip = 0.5 * ((x_wm - x_lat_pre) + (x_skip_pre - y_base))
            D_lat  = d_lat.mean(dim=1, keepdim=True)
            D_skip = d_skip.mean(dim=1, keepdim=True)
            e_lat0  = float((d_lat[0]  ** 2).sum().item())
            e_skip0 = float((d_skip[0] ** 2).sum().item())

        # ---------- generator losses ----------
        res = x_wm - y_base

        # ROI union lifted to full resolution
        roi_full = F.interpolate(
            torch.clamp(P_64 + F.interpolate(P_lat, size=(64, 64), mode="nearest"), 0, 1),
            size=(H, W), mode="bilinear", align_corners=False
        )

        # Leak loss penalizes energy outside ROI (perceptual guardrail)
        L_leak = (res.abs() * (1.0 - roi_full)).mean()

        # High‑frequency leakage: Laplacian residual outside ROI
        lap_k = GLat.lap_kernel(DEV, dtype=res.dtype).repeat(C, 1, 1, 1)
        lap = F.conv2d(res, lap_k, padding=1, groups=C).abs()
        L_hf = (lap * (1.0 - roi_full)).mean()

        # Spatial smoothness of ROI masks (total variation)
        L_tv = (F.l1_loss(P_64[:, :, 1:, :], P_64[:, :, :-1, :]) +
                F.l1_loss(P_64[:, :, :, 1:], P_64[:, :, :, :-1]) +
                F.l1_loss(P_lat[:, :, 1:, :], P_lat[:, :, :-1, :]) +
                F.l1_loss(P_lat[:, :, :, 1:], P_lat[:, :, :, :-1]))
        # Sparsity penalty: encourage small area of active ROI
        L_sparse = (P_64.mean() + P_lat.mean())

        # C1 guard: enforce similarity between C1 predictions on clean and wm images
        with torch.no_grad():
            logits_c1_clean = self.c1.logits(x)
        logits_c1_wm = self.c1.logits((x_wm - 0.5) / 0.5)
        p_clean = torch.softmax(logits_c1_clean, dim=1)
        p_wm    = torch.softmax(logits_c1_wm, dim=1)
        m = 0.5 * (p_clean + p_wm)
        L_js = 0.5 * (F.kl_div((p_clean + 1e-12).log(), m, reduction='batchmean') +
                      F.kl_div((p_wm + 1e-12).log(), m, reduction='batchmean'))
        L_cos = 1.0 - F.cosine_similarity(logits_c1_clean, logits_c1_wm, dim=1).mean()
        L_c1 = LAMBDA_C1_GUARD * (L_js + L_cos)

        # Early epochs: weaker leak/HF weights; later epochs: stronger enforcement
        W_LEAK = 0.2 if epoch <= 2 else 0.5
        W_HF   = 0.2 if epoch <= 2 else 0.5

        L = (W_LEAK * L_leak +
             W_HF   * L_hf +
             L_c1 +
             0.05 * L_tv +
             0.02 * L_sparse)

        self.opt_g.zero_grad(set_to_none=True)
        L.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.mask_lat.parameters()) + list(self.mask_64.parameters()) +
            list(self.g_lat.parameters())    + list(self.g_64.parameters()),
            max_norm=1.0
        )
        self.opt_g.step()

        # ---------- metrics / logging ----------
        with torch.no_grad():
            c1_acc_clean = (logits_c1_clean.argmax(1) == y).float().mean().item()
            c1_acc_wm    = (logits_c1_wm.argmax(1) == y).float().mean().item()
            c1_drop      = max(0.0, c1_acc_clean - c1_acc_wm)
            C1_JS        = float(L_js.item())
            C1_cos       = 1.0 - F.cosine_similarity(
                logits_c1_clean, logits_c1_wm, dim=1
            ).mean().item()
            PSNR         = psnr_torch(x_wm, y_base).item()
            SSIM         = ssim_torch(x_wm, y_base).item()
            overlap      = (P_lat * S_lat).mean().item() + (P_64 * S_64).mean().item()

            # Seismic visualization of signed difference
            diff_signed = (x_wm - y_base).mean(dim=1, keepdim=True)
            p = 99.0
            lo = torch.quantile(
                diff_signed.flatten(1), (100 - p) / 200, dim=1, keepdim=True
            ).view(B, 1, 1, 1)
            hi = torch.quantile(
                diff_signed.flatten(1), (100 + p) / 200, dim=1, keepdim=True
            ).view(B, 1, 1, 1)
            seismic = torch.clamp((diff_signed - lo) / (hi - lo + 1e-8), 0, 1)

        out = {
            "loss": float(L.item()),
            "L_leak": float(L_leak.item()), "L_hf_out": float(L_hf.item()),
            "L_tv": float(L_tv.item()), "L_sparse": float(L_sparse.item()),
            "c1_acc_clean": c1_acc_clean, "c1_acc_wm": c1_acc_wm, "c1_drop": c1_drop,
            "C1_JS": C1_JS, "C1_cos": C1_cos,
            "PSNR": PSNR, "SSIM": SSIM,
            "overlap": overlap,
            "labels": y.detach().cpu(),
        }
        if log_images:
            out.update({
                "x": x.detach(),
                "x_wm": x_wm.detach(),
                "y_ref": y_base.detach(),
                "seismic": seismic.detach(),
                "S": S.detach(),
                "D_lat": D_lat.detach(),
                "D_skip": D_skip.detach(),
                "E_lat": e_lat0, "E_skip": e_skip0,
                "P_lat": P_lat.detach(), "P_64": P_64.detach(),
                "paths": paths,
            })
        return out

    # ---------- discriminator step ----------

    def step_c2(self, batch, epoch: int):
        """
        Single C2 optimization step.

        C2 is trained to:
          * maintain strong classification performance on clean images,
          * increase accuracy and confidence on watermarked images,
          * rely on watermark presence (via wm_logit gating) for the target Δ(C2),
          * maintain a controlled margin between clean and watermarked logits.
        """
        self.c2.train()
        self.mask_lat.eval()
        self.mask_64.eval()
        self.g_lat.eval()
        self.g_64.eval()

        # Optionally freeze parts of C2 in very early epochs
        base_should_freeze = (epoch <= getattr(self, "freeze_c2_base_epochs", 0))
        for p in self.c2.base.parameters():
            p.requires_grad_(not base_should_freeze)
        for p in self.c2.lf_stem.parameters():
            p.requires_grad_(not base_should_freeze)
        for p in self.c2.fuse1.parameters():
            p.requires_grad_(not base_should_freeze)
        for p in self.c2.wm_head.parameters():
            p.requires_grad_(True)
        self.c2.wm_affine.requires_grad_(True)

        x, y, _ = batch
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        x_ae = self._to01(x)

        # Build watermarked version for C2 using current G (no gradients into G/AE)
        with torch.no_grad():
            S = self.c1.cam(x)
            S_64 = F.interpolate(S, size=(64, 64), mode='bilinear', align_corners=False)
            S_lat = F.interpolate(S, size=(32, 32), mode='bilinear', align_corners=False)

            taps = self.ae_features(x_ae)
            Z_true, S64_true = taps["latent"], taps["s64"]

            P_lat, P_64, *_ = self.build_masks(Z_true, S64_true, x, S_lat, S_64)

            z_pat = self.g_lat(Z_true) * P_lat
            w64_1 = self.g_64(S64_true)
            w64 = self.g_64_proj(w64_1) * P_64

            # unit‑RMS normalization as in generator step
            def _unit_rms(t, mask):
                denom = mask.sum(dim=(1, 2, 3), keepdim=True).clamp_min(1.0)
                rms = torch.sqrt(((t ** 2) * mask).sum(dim=(1, 2, 3), keepdim=True) / denom + 1e-12)
                return t / rms

            z_pat = _unit_rms(z_pat, P_lat)
            w64 = _unit_rms(w64, P_64)

            # same α as in G, with a small boost specifically for C2
            alpha_lat, alpha_skip, r_lat_eff, r_skip_eff = self._alphas_from_ctrl()
            B = x_ae.size(0)
            eps = 1e-6
            area_lat = (P_lat.flatten(1).sum(1) + eps)
            area_64 = (P_64.flatten(1).sum(1) + eps)
            A_lat, A_64 = 32 * 32, 64 * 64
            alpha_lat_b = alpha_lat * (A_lat / area_lat).sqrt().view(B, 1, 1, 1)
            alpha_skip_b = alpha_skip * (A_64 / area_64).sqrt().view(B, 1, 1, 1)

            # mild boost for C2 so that separability appears early
            k_c2 = 1.5 if epoch <= 2 else 1.2
            x_wm_pre = self.ae.embed_external_wm_gray(
                x_ae, wm_lat=z_pat, wm_skip=w64,
                alpha_lat=alpha_lat_b * k_c2, alpha_skip=alpha_skip_b * k_c2,
                roi_lat_32=P_lat, roi_skip_64=P_64
            ).clamp(0, 1)

            x_wm_pre = torch.nan_to_num(x_wm_pre, nan=0.0, posinf=1.0, neginf=0.0)
            x_wm_n = (x_wm_pre - 0.5) / 0.5
            x_wm_n = torch.nan_to_num(x_wm_n, nan=0.0, posinf=0.0, neginf=0.0)

        # --- focal BCE helper for the watermark head ---
        def focal_bce_with_logits(logits: torch.Tensor, targets: torch.Tensor,
                                  gamma: float = 2.0) -> torch.Tensor:
            prob = torch.sigmoid(logits)
            ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
            pt = prob * targets + (1.0 - prob) * (1.0 - targets)
            w = (1.0 - pt).pow(gamma)
            return (w * ce).mean()

        with torch.amp.autocast(device_type="cuda", enabled=MIXED_PRECISION):
            logits_clean_ng, wm_clean, _ = self.c2(x, gate=False)
            logits_wm, wm_wm, _ = self.c2(x_wm_n, gate=True)
            logits_clean_full, _, _ = self.c2(x, gate=True)

            # Confidence‑weighted CE on watermarked samples
            with torch.no_grad():
                p_wm_sm = torch.softmax(logits_wm, dim=1)
                pwm_true = p_wm_sm.gather(1, y.view(-1, 1)).squeeze(1)
            wm_w = (1.0 - pwm_true).detach()
            L_cls_wm = (F.cross_entropy(logits_wm, y, reduction='none') * (1.0 + wm_w)).mean()

            # Controlled CE on clean samples (with temperature)
            T_CLEAN = 1.5 if epoch <= 3 else 1.3
            GAMMA_CLN = 0.02 if epoch >= 6 else 0.0
            L_cls_clean = F.cross_entropy(logits_clean_ng / T_CLEAN, y)

            # Margin between clean and wm logits for the true class
            z_c = logits_clean_ng.gather(1, y.view(-1, 1)).squeeze(1)
            z_w = logits_wm.gather(1, y.view(-1, 1)).squeeze(1)
            L_margin = F.relu(1.5 - (z_w - z_c)).mean()

            # Watermark head BCE: positive on watermarked, negative on clean
            wm_act = torch.sigmoid(wm_wm).mean().item()
            gamma_wm = (3.0 if epoch <= 2 else 2.5) if wm_act < 0.56 else (
                2.5 if wm_act < 0.60 else 2.0
            )
            L_bce_pos = focal_bce_with_logits(wm_wm, torch.ones_like(wm_wm), gamma=gamma_wm)
            L_bce_neg = focal_bce_with_logits(wm_clean, torch.zeros_like(wm_clean), gamma=gamma_wm)
            L_bce = L_bce_pos + L_bce_neg

            # Encourage wm_logit(wm) > wm_logit(clean) by a margin
            L_wm_logit_margin = F.relu(1.0 - (wm_wm - wm_clean)).mean()
            W_WM_MARGIN = 1.5 if wm_act < 0.52 else (1.0 if wm_act < 0.56 else 0.6)

            # Pairwise margin on normalized logit differences (true vs rest)
            other = logits_clean_ng.clone()
            other.scatter_(1, y.view(-1, 1), float("-inf"))
            other_w = logits_wm.clone()
            other_w.scatter_(1, y.view(-1, 1), float("-inf"))
            s_c = z_c - torch.logsumexp(other, dim=1)
            s_w = z_w - torch.logsumexp(other_w, dim=1)
            L_pair = F.softplus(0.5 - (s_w - s_c)).mean()

            # Soft target Δ on probabilities for the true class
            p_clean_sm = torch.softmax(logits_clean_full, dim=1)
            p_wm_sm2 = torch.softmax(logits_wm, dim=1)
            idx = y.view(-1, 1)
            pclean_true2 = p_clean_sm.gather(1, idx).squeeze(1)
            pwm_true2 = p_wm_sm2.gather(1, idx).squeeze(1)
            soft_delta = (pwm_true2 - pclean_true2).mean()
            L_bal = 0.35 * ((TARGET_DELTA - soft_delta) ** 2)

            # JS divergence between full class distributions (encourage separation)
            m_mix = 0.5 * (p_clean_sm + p_wm_sm2)
            js_sep = 0.5 * (F.kl_div((p_clean_sm + 1e-12).log(), m_mix, reduction='batchmean') +
                            F.kl_div((p_wm_sm2 + 1e-12).log(), m_mix, reduction='batchmean'))
            COEF_JS = 0.20 if wm_act < 0.55 else (0.15 if wm_act < 0.58 else 0.10)
            L_js_sep = -COEF_JS * js_sep  # negative sign: we want large JS

            # Entropy regularization: avoid over‑confident clean predictions
            ent_clean = -(p_clean_sm * (p_clean_sm + 1e-12).log()).sum(dim=1).mean()
            L_ent_clean = 0.09 * ent_clean
            L_ent = 0.02 * ent_clean.mean()

            L_C2 = (1.4 * L_cls_wm
                    + 0.3 * GAMMA_CLN * L_cls_clean
                    + 1.2 * L_margin
                    + 1.5 * L_bce
                    + 0.8 * L_bal
                    + 0.5 * L_pair
                    + W_WM_MARGIN * L_wm_logit_margin
                    + L_js_sep
                    + L_ent_clean
                    + L_ent)

        self.opt_c2.zero_grad(set_to_none=True)
        self.scaler.scale(L_C2).backward()
        self.scaler.unscale_(self.opt_c2)
        torch.nn.utils.clip_grad_norm_(self.c2.parameters(), max_norm=1.0)
        self.scaler.step(self.opt_c2)
        self.scaler.update()

        # EMA update for evaluation‑mode Δ(C2)
        self._ema_update(self.c2_ema, self.c2, self.ema_tau)

        with torch.no_grad():
            logits_clean_e, _, _ = self.c2_ema(x, gate=True)
            logits_wm_e, _, _ = self.c2_ema(x_wm_n, gate=True)
            p_clean_e = torch.softmax(logits_clean_e, dim=1)
            p_wm_e = torch.softmax(logits_wm_e, dim=1)
            idx = y.view(-1, 1)
            pclean_true_e = p_clean_e.gather(1, idx).squeeze(1)
            pwm_true_e = p_wm_e.gather(1, idx).squeeze(1)
            delta_prob = (pwm_true_e - pclean_true_e).mean().item()
            acc_clean = (logits_clean_e.argmax(1) == y).float().mean().item()
            acc_wm = (logits_wm_e.argmax(1) == y).float().mean().item()
            delta = acc_wm - acc_clean
            c2_cos = cosine_logits(logits_clean_e, logits_wm_e)

        return {
            'L_C2': L_C2.item(),
            'c2_acc_clean': acc_clean,
            'c2_acc_wm': acc_wm,
            'delta': delta,
            'delta_prob': delta_prob,
            'c2_cos': c2_cos,
        }

    # ---------- controller ----------

    def controller_update(self, gstats, c2stats, batch_idx, total_batches, epoch):
        """
        Update the global controller (ε and r_skip) based on current statistics.

        Signals used:
          * soft_delta / delta_now  : Δ(C2) in probability and accuracy space;
          * psnr_now                : reconstruction fidelity between wm and base;
          * overlap                 : overlap between ROI and high‑CAM regions;
          * c1_drop                 : C1 accuracy degradation induced by watermark.

        The update logic is conservative: it blocks ε increases when visual quality
        or C1 stability deteriorate, and adjusts r_skip to keep latent contribution
        ≥ 33% of the energy.
        """
        soft_delta = float(c2stats.get('delta_prob', c2stats.get('delta', 0.0)))
        delta_now = float(c2stats.get('delta', 0.0))
        psnr_now = float(gstats.get('PSNR', 99.0))
        overlap = float(gstats.get('overlap', 0.0))
        c1_drop = float(gstats.get('c1_drop', 0.0))

        # EMA over soft Δ for smoother decisions
        EMAD_prev = getattr(self.ctrl, "ema_delta", 0.0)
        self.ctrl.ema_delta = EMAD_BETA * EMAD_prev + (1.0 - EMAD_BETA) * soft_delta

        # Target ε under linear warm‑up across EPS_WARMUP_FRAC of total epochs
        warmup_epochs = max(1, int(EPOCHS * EPS_WARMUP_FRAC))
        eps_warm = EPS_MAX * min(1.0, epoch / warmup_epochs)
        eps_target = eps_warm

        # Do not increase ε if C1 drop is too large
        if c1_drop > C1_DROP_BLOCK_INCREASE:
            eps_target = min(self.ctrl.eps, eps_target)

        # Anti‑stall: if Δ≈0, EMA small, and PSNR very high, nudge ε upward
        if (abs(delta_now) < 1e-3) and (self.ctrl.ema_delta < 0.02) and (psnr_now > 60.0):
            eps_target = max(eps_target, self.ctrl.eps + 0.015)

        # Adjust ε at a coarse granularity depending on quality and separation
        if (batch_idx + 1) % CTRL_UPDATE_EVERY == 0:
            bad_quality = (psnr_now < PSNR_MIN) or (overlap > 0.05)
            weak_sep = (self.ctrl.ema_delta < TARGET_DELTA - TARGET_HYST)
            good_sep = (self.ctrl.ema_delta > TARGET_DELTA + TARGET_HYST) and (psnr_now >= PSNR_MIN)

            if bad_quality:
                eps_target = min(self.ctrl.eps, eps_target)  # do not grow ε
            elif weak_sep:
                eps_target = max(eps_target, self.ctrl.eps + 0.002)  # gently push ε up
            elif good_sep:
                eps_target = min(eps_target, max(EPS_MIN, self.ctrl.eps - 0.001))  # slightly relax

        # Smooth move towards eps_target and clamp
        k = 0.20 if epoch <= 2 else 0.10  # faster at the very beginning
        self.ctrl.eps += k * (eps_target - self.ctrl.eps)
        self.ctrl.eps = float(max(EPS_MIN, min(EPS_MAX, self.ctrl.eps)))

        # r_skip corridor and hysteresis; upper bound ensures latent ≥ 33%
        LOW, HIGH = 0.60, 0.67
        if delta_now < 0.0:
            # separation goes in the wrong direction → increase latent share
            self.ctrl.r_skip = max(LOW, self.ctrl.r_skip - 0.05)
        elif (self.ctrl.ema_delta < 0.10) and (psnr_now > 30.0):
            # weak separation, acceptable quality → increase skip share (up to cap)
            self.ctrl.r_skip = min(HIGH, self.ctrl.r_skip + 0.04)
        elif (self.ctrl.ema_delta > 0.35) and (psnr_now >= 27.0):
            # separation already decent → mild increase in skip contribution
            self.ctrl.r_skip = min(HIGH, self.ctrl.r_skip + 0.02)
        elif c1_drop > 0.05:
            # C1 is sensitive → shift a bit more energy into skip (less visible channel)
            self.ctrl.r_skip = min(HIGH, self.ctrl.r_skip + 0.02)

        if self.ctrl.r_skip > HIGH:
            self.ctrl.r_skip = float(HIGH)

    # ---------- collage ----------

    def save_collage(self, x, x_wm, aux, paths, labels, epoch, it, out_dir: Path):
        """
        Build a diagnostic collage for a single sample (index 0 in the batch).

        Layout (6 tiles horizontally):
          1) Original (grayscale)
          2) AE baseline reconstruction (watermark off)
          3) Watermarked reconstruction with controller weights and energy split
          4) Seismic diff (x_wm − base) in red/blue on white
          5) Latent contribution Δ_lat (red/blue)
          6) Skip‑64 contribution Δ_skip (red/blue)

        Collages are saved into:
          OUT_ROOT/<epoch>/<class_name>/<stem>_itXXXXX_collage.png

        where <class_name> is derived from the dataset label.
        """
        import torchvision.transforms.functional as TF
        import math
        import torch.nn.functional as F
        from pathlib import Path

        idx = 0
        W = H = IMAGE_SIZE
        pad = 8
        cap_h = 28
        tiles = 6
        canvas_w = tiles * W + (tiles + 1) * pad
        canvas_h = H + cap_h + 2 * pad

        def to_pil_01(t3):
            """[1,3,H,W] in [0,1] → PIL.Image (RGB)."""
            return TF.to_pil_image(t3.squeeze(0).cpu())

        def to_redblue_on_white(v_signed):
            """
            Map signed scalar field v ∈ [-1,1] to a red/blue overlay on white.

            Positive values go to red, negative to blue, with intensity proportional
            to |v|.
            """
            v = v_signed.clamp(-1, 1)
            a = v.abs()
            C = torch.cat([(v > 0).float(), torch.zeros_like(v), (v < 0).float()], dim=1)
            rgb = (1.0 - a).repeat(1, 3, 1, 1) + a * C
            return rgb.clamp(0, 1)

        def percentile_scale_signed(t_signed, q=0.99, eps=1e-6):
            """
            Scale a signed tensor so that the q‑quantile of |t| maps to amplitude 1.
            """
            a = t_signed.detach().abs().flatten()
            k = max(1, int(math.ceil(a.numel() * q)))
            s = float(a.kthvalue(k).values.clamp_min(eps))
            return (t_signed.clamp(-s, s) / s).clamp(-1, 1)

        x01 = self._to01(x[idx:idx + 1])
        xre = aux.get('y_ref', x01)[idx:idx + 1]
        xwm = x_wm[idx:idx + 1].clamp(0, 1)

        # Seismic diff
        diff_signed = (xwm - xre).mean(dim=1, keepdim=True)
        diff_signed = F.interpolate(diff_signed, size=(H, W), mode='bilinear', align_corners=False)
        diff_scaled = percentile_scale_signed(diff_signed, q=0.99)
        diff_rgb = to_redblue_on_white(diff_scaled)

        # Latent and skip contributions
        Dlat = aux['D_lat'][idx:idx + 1]
        Dskp = aux['D_skip'][idx:idx + 1]
        if Dlat.shape[-2:] != (H, W):
            Dlat = F.interpolate(Dlat, size=(H, W), mode='bilinear', align_corners=False)
        if Dskp.shape[-2:] != (H, W):
            Dskp = F.interpolate(Dskp, size=(H, W), mode='bilinear', align_corners=False)
        Dlat_sc = percentile_scale_signed(Dlat, q=0.99)
        Dskp_sc = percentile_scale_signed(Dskp, q=0.99)
        Dlat_rgb = to_redblue_on_white(Dlat_sc)
        Dskp_rgb = to_redblue_on_white(Dskp_sc)

        # Convert to PIL tiles
        pil_original = to_pil_01(x01)
        pil_recon = to_pil_01(xre)
        pil_wm = to_pil_01(xwm)
        pil_diff = to_pil_01(diff_rgb)
        pil_lat_rb = to_pil_01(Dlat_rgb)
        pil_s64_rb = to_pil_01(Dskp_rgb)

        # Caption with controller weights and energy percentages
        r_skip = float(self.ctrl.r_skip)
        r_lat = float(1.0 - r_skip)
        E_lat = float(aux.get('E_lat', 0.0))
        E_skip = float(aux.get('E_skip', 0.0))
        E_sum = max(1e-12, E_lat + E_skip)
        e_lat_pct = 100.0 * (E_lat / E_sum)
        e_skip_pct = 100.0 * (E_skip / E_sum)

        cap1 = "Original"
        cap2 = "AE Recon"
        cap3 = (f"Watermarked | r_lat={r_lat * 100:.1f}% r_skip={r_skip * 100:.1f}% | "
                f"E_lat={e_lat_pct:.1f}% E_skip={e_skip_pct:.1f}%")
        cap4 = "Seismic diff (x_wm - base)"
        cap5 = "Latent contribution Δ_lat (red–blue)"
        cap6 = "Skip64 contribution Δ_skip (red–blue)"

        canvas = Image.new("RGB", (canvas_w, canvas_h), (12, 12, 12))
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("arial.ttf", 18)
        except Exception:
            font = ImageFont.load_default()

        tiles_with_caps = [
            (pil_original, cap1),
            (pil_recon, cap2),
            (pil_wm, cap3),
            (pil_diff, cap4),
            (pil_lat_rb, cap5),
            (pil_s64_rb, cap6),
        ]

        x_off = pad
        y_off = pad
        for pil_img, caption in tiles_with_caps:
            canvas.paste(pil_img, (x_off, y_off))
            bbox = draw.textbbox((0, 0), caption, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            cx = x_off + (W - tw) // 2
            cy = y_off + H + (cap_h - th) // 2
            draw.rectangle([cx - 4, cy - 2, cx + tw + 4, cy + th + 2], fill=(0, 0, 0))
            draw.text((cx, cy), caption, fill=(255, 255, 255), font=font)
            x_off += W + pad

        # Class name for folder structure
        class_name = "unknown"
        if labels is not None:
            lbl_tensor = labels[idx]
            try:
                cls_idx = int(lbl_tensor.item()) if hasattr(lbl_tensor, "item") else int(lbl_tensor)
            except Exception:
                cls_idx = -1
            if 0 <= cls_idx < len(self.classes):
                class_name = self.classes[cls_idx]
        else:
            class_name = Path(paths[idx]).parent.name

        ep_dir = Path(out_dir) / f"{epoch:02d}" / class_name
        ensure_dir(ep_dir)

        stem = Path(paths[idx]).stem
        out_path = ep_dir / f"{stem}_it{it:05d}_collage.png"
        canvas.save(out_path)
        return out_path

    # ---------- train loop ----------

    def train(self):
        """
        Outer training loop.

        Each epoch alternates between:
          * generator steps (embedding and fidelity control),
          * C2 steps (discriminative training and Δ(C2) shaping),
          * controller updates for ε and r_skip,
          * periodic collage dumps for qualitative monitoring,
          * CSV logging of epoch‑level statistics.

        Training stops early when EMA Δ(C2) reaches TARGET_DELTA.
        """
        out_root = Path(OUT_ROOT)
        ensure_dir(out_root)

        for epoch in range(1, EPOCHS + 1):
            self.current_epoch = epoch
            t0 = time.time()

            # Epoch‑level accumulators
            c1_clean_sum = 0.0
            c1_wm_sum = 0.0
            c1_n = 0
            delta_prob_sum = 0.0

            c2_clean_sum = 0.0
            c2_wm_sum = 0.0
            delta_sum = 0.0
            c2_n = 0

            for it, batch in enumerate(self.train_loader, start=1):
                gstats  = self.step_generator(batch, epoch, it, out_root)
                c2stats = self.step_c2(batch, epoch)

                if it == 1 or (it % 10 == 0):
                    out_path = self.save_collage(
                        gstats["x"], gstats["x_wm"],
                        {
                            "y_ref": gstats["y_ref"],
                            "D_lat": gstats["D_lat"], "D_skip": gstats["D_skip"],
                            "S": gstats["S"], "seismic": gstats["seismic"],
                            "E_lat": gstats.get("E_lat", 0.0),
                            "E_skip": gstats.get("E_skip", 0.0),
                        },
                        gstats["paths"],
                        gstats["labels"],
                        epoch, it, out_root
                    )

                    print(f"[COLLAGE] saved: {out_path}")

                self.controller_update(gstats, c2stats, it - 1, len(self.train_loader), epoch)
                delta_prob_sum += c2stats.get('delta_prob', 0.0)

                # Accumulate epoch stats
                c1_clean_sum += gstats['c1_acc_clean']
                c1_wm_sum    += gstats['c1_acc_wm']
                c1_n += 1

                c2_clean_sum += c2stats['c2_acc_clean']
                c2_wm_sum    += c2stats['c2_acc_wm']
                delta_sum    += c2stats['delta']
                c2_n += 1

                if it % 5 == 0:
                    print(
                        f"[E{epoch:02d} it {it:05d}] "
                        f"eps={self.ctrl.eps:.4f} r_skip={self.ctrl.r_skip:.2f} | "
                        f"C1 acc c={gstats['c1_acc_clean']:.3f} "
                        f"wm={gstats['c1_acc_wm']:.3f} drop={gstats['c1_drop']:.3f} "
                        f"JS={gstats['C1_JS']:.4f} cos={gstats['C1_cos']:.3f} | "
                        f"C2 acc c={c2stats['c2_acc_clean']:.3f} "
                        f"wm={c2stats['c2_acc_wm']:.3f} Δ={c2stats['delta']:.3f} "
                        f"(EMA {self.ctrl.ema_delta:.3f}) | "
                        f"PSNR={gstats['PSNR']:.2f} SSIM={gstats['SSIM']:.3f} "
                        f"overlap={gstats['overlap']:.4f}"
                    )

            print(
                f"Epoch {epoch} done in {time.time() - t0:.1f}s | "
                f"EMA Δ={self.ctrl.ema_delta:.3f} | "
                f"eps={self.ctrl.eps:.4f} r_skip={self.ctrl.r_skip:.2f}"
            )

            # Epoch‑mean statistics
            c1_acc_clean_ep = (c1_clean_sum / max(1, c1_n))
            c1_acc_wm_ep    = (c1_wm_sum / max(1, c1_n))
            c1_drop_ep      = c1_acc_clean_ep - c1_acc_wm_ep

            c2_acc_clean_ep = (c2_clean_sum / max(1, c2_n))
            c2_acc_wm_ep    = (c2_wm_sum / max(1, c2_n))
            delta_mean_ep   = (delta_sum / max(1, c2_n))
            delta_ema_ep    = self.ctrl.ema_delta
            delta_prob_ep   = (delta_prob_sum / max(1, c2_n))

            print(
                f"[E{epoch:02d} MEAN] ... Δ={delta_mean_ep:.3f} "
                f"(soft {delta_prob_ep:.3f}, EMA {delta_ema_ep:.3f})"
            )
            print(
                f"[E{epoch:02d} MEAN] "
                f"C1 c={c1_acc_clean_ep:.3f} wm={c1_acc_wm_ep:.3f} "
                f"drop={c1_drop_ep:.3f} | "
                f"C2 c={c2_acc_clean_ep:.3f} wm={c2_acc_wm_ep:.3f} "
                f"Δ={delta_mean_ep:.3f} (EMA {delta_ema_ep:.3f})"
            )

            # CSV epoch summary
            import csv, os
            summary_csv = os.path.join(OUT_ROOT, "wm_epoch_summary.csv")
            file_exists = os.path.isfile(summary_csv)
            with open(summary_csv, "a", newline="") as f:
                w = csv.writer(f)
                if not file_exists:
                    w.writerow([
                        "epoch", "c1_acc_clean", "c1_acc_wm", "c1_drop",
                        "c2_acc_clean", "c2_acc_wm",
                        "delta_mean", "delta_mean_soft", "delta_ema"
                    ])
                w.writerow([
                    epoch,
                    f"{c1_acc_clean_ep:.6f}", f"{c1_acc_wm_ep:.6f}", f"{c1_drop_ep:.6f}",
                    f"{c2_acc_clean_ep:.6f}", f"{c2_acc_wm_ep:.6f}",
                    f"{delta_mean_ep:.6f}", f"{delta_prob_ep:.6f}", f"{delta_ema_ep:.6f}",
                ])

            # Stop when EMA Δ reaches target
            if self.ctrl.ema_delta >= TARGET_DELTA:
                print(">>> STOP: Target Δ reached.")
                break


# =========================
# --------- MAIN ----------
# =========================

if __name__ == "__main__":
    if AE_PY_PATH:
        sys.path.append(AE_PY_PATH)
    trainer = WatermarkTrainer()
    trainer.train()
