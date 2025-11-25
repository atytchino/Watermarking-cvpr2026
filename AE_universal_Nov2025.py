# We'll write a complete Python file implementing the requested autoencoder with:
# - Pretraining on three datasets (AFHQ color + two grayscale datasets)
# - Balanced RGB/Gray batches
# - External watermark injection surfaces (latent & skip64) with full-channel access
# - Two entrypoints for WM: embed_external_wm_rgb / embed_external_wm_gray
# - Checkpoints each epoch and best; default epochs=10
# - Split PSNR/SSIM logs for grayscale and RGB
#
# The file will be saved at /mnt/data/AE_universal_20251022_external_wm.py

"""
Universal AE (Y-core) with external watermark injection surfaces.
- Pretraining on three datasets: AFHQ (RGB), MRI (Gray), ORNL (Gray)
- Balanced batches 1:1 (color vs gray)
- External watermark API: full-channel access to latent[1024,32,32] and skip64[512,64,64]
- Two entrypoints: embed_external_wm_rgb / embed_external_wm_gray
- Checkpoint every epoch and best; default epochs=10
- Split PSNR/SSIM logs (Y overall, Y-color, Y-gray, and RGB-color)

This file is a focused evolution of the previously shared hybrid AE training script,
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
# -------- PATHS FOR DATASETS-NOT MORE THAN 3 ----------
# =========================

#TLD_TRAIN = r"E:\TLD\Train"
#TLD_VAL   = r"E:\TLD\Val"

#AFHQ_TRAIN= r"E:\AFHQ_LOWMEM\train"
#AFHQ_VAL  = r"E:\AFHQ_LOWMEM\val"

ORNL_TRAIN= r"E:\ORNL_GRAYSCALE\Train"
ORNL_VAL  = r"E:\ORNL_GRAYSCALE\Val"

TRAIN_ROOT_PAIRS = [(ORNL_TRAIN, ORNL_VAL),(None, None), (None,None)]

SAVE_ROOT = Path(r"E:\Universal_ae_ver2_ORNL")
SAVE_ROOT.mkdir(parents=True, exist_ok=True)
CKPT_DIR = SAVE_ROOT
LOG_DIR  = SAVE_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# -------- UTILS ----------
# =========================

def seed_all(seed=123456789):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def to_luma_Y(x: torch.Tensor) -> torch.Tensor:
    """
    x: [B,C,H,W] in [0,1].
    C==3 -> Y=0.299R+0.587G+0.114B
    C==1 -> as is
    Return [B,1,H,W]
    """
    if x.size(1) == 1:
        return x
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return y

def rgb_to_ycbcr(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    y  = 0.299*r + 0.587*g + 0.114*b
    cb = (-0.168736)*r - 0.331264*g + 0.5*b + 0.5
    cr = 0.5*r - 0.418688*g - 0.081312*b + 0.5
    return y.clamp(0,1), cb.clamp(0,1), cr.clamp(0,1)

def ycbcr_to_rgb(y: torch.Tensor, cb: torch.Tensor, cr: torch.Tensor) -> torch.Tensor:
    cb_c = cb - 0.5
    cr_c = cr - 0.5
    r = y + 1.402*cr_c
    g = y - 0.344136*cb_c - 0.714136*cr_c
    b = y + 1.772*cb_c
    return torch.cat([r,g,b], dim=1).clamp(0,1)

# ---- PSNR/SSIM per-image ----

def psnr_tensor(x: torch.Tensor, y: torch.Tensor, data_range: float = 1.0, eps: float = 1e-12) -> torch.Tensor:
    """
    Per-image PSNR: returns [B]
    """
    mse = (x - y).pow(2).flatten(1).mean(dim=1).clamp_min(eps)
    return 10.0 * torch.log10((data_range ** 2) / mse)

def _gaussian_window(window_size: int, sigma: float, device, dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).unsqueeze(0)  # [1,W]
    window_2d = (g.t() @ g)         # [W,W]
    return window_2d

def ssim_tensor(x: torch.Tensor, y: torch.Tensor, data_range: float = 1.0,
                window_size: int = 11, sigma: float = 1.5, eps: float = 1e-12) -> torch.Tensor:
    """
    Per-image SSIM: returns [B], supports 1 or 3 channels (averages over channels).
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

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2) + eps)
    ssim_per_img = ssim_map.clamp(0, 1).flatten(2).mean(dim=2).mean(dim=1)
    return ssim_per_img

# ---- Meters & split logging ----

from dataclasses import dataclass

@dataclass
class AvgMeter:
    sum: float = 0.0
    n: int = 0
    def update(self, val, k: int = 1):
        if val is None: return
        self.sum += float(val) * k
        self.n += k
    @property
    def avg(self) -> float:
        return self.sum / max(1, self.n)

class SplitMeters:
    def __init__(self):
        self.psnr_y_all = AvgMeter(); self.ssim_y_all = AvgMeter()
        self.psnr_y_color = AvgMeter(); self.ssim_y_color = AvgMeter()
        self.psnr_y_gray  = AvgMeter(); self.ssim_y_gray  = AvgMeter()
        self.psnr_rgb_color = AvgMeter(); self.ssim_rgb_color = AvgMeter()
    def log_line(self, ep: int, it: int, loss_val: float) -> str:
        return (f"[E{ep:02d} it {it:05d}] loss={loss_val:.4f} | "
                f"Y: psnr={self.psnr_y_all.avg:.2f} ssim={self.ssim_y_all.avg:.3f} | "
                f"Ycolor={self.psnr_y_color.avg:.2f}/{self.ssim_y_color.avg:.3f} "
                f"Ygray={self.psnr_y_gray.avg:.2f}/{self.ssim_y_gray.avg:.3f} | "
                f"RGBcolor={self.psnr_rgb_color.avg:.2f}/{self.ssim_rgb_color.avg:.3f}")

@torch.no_grad()
def update_split_metrics(meters: SplitMeters, x: torch.Tensor, y_hat: torch.Tensor, is_color_mask: torch.Tensor):
    """
    x, y_hat: [B,C,H,W] in [0,1]; C=1 or 3; is_color_mask: [B] bool
    Метрики всегда считаем в float32 без AMP, чтобы не ловить конфликт типов.
    """
    # на всякий случай выключаем autocast и приводим к float32
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
            # для цветных считаем PSNR/SSIM по RGB
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

def detect_dataset_mode(root: str,
                        max_sample: int = 100,
                        name: Optional[str] = None) -> str:
    """
    Определяет режим датасета по 100 (или меньше) случайным картинкам из root.
    Возвращает "rgb" или "gray".
    Бросает RuntimeError, если внутри датасета смешано слишком много.
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
            # если картинка битая — считаем её цветной, чтобы не завалить всё в gray
            is_color = True
        if is_color:
            n_color += 1
        else:
            n_gray += 1

    mismatches = min(n_color, n_gray)
    mode = "rgb" if n_color >= n_gray else "gray"

    ds_name = name or os.path.basename(os.path.normpath(root)) or root
    print(f"[DATASET CHECK] {ds_name}: mode={mode.upper()}, "
          f"n_color={n_color}, n_gray={n_gray}, sampled={k}, mismatches={mismatches}")

    if mismatches > 5:
        raise RuntimeError(
            f"Dataset '{ds_name}' under {root} is not homogeneous enough: "
            f"{n_color} color, {n_gray} gray in {k} samples (mismatches={mismatches}). "
            f"Приведи датасет в порядок (разнеси RGB и GRAY по разным корням)."
        )
    if mismatches >= 1:
        print(f"[WARN] Dataset '{ds_name}' has some mixed images "
              f"(mismatches={mismatches} <= 5). Продолжаем, но это подозрительно.")

    return mode


def detect_is_color_pil(img: Image.Image, nearly_gray_thresh: int = 2) -> bool:
    """
    Heuristic:
    - If PIL mode is single-channel -> gray.
    - Else convert to RGB and check max diff between channels; if <= thresh -> gray.
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
    Single dataset over multiple roots.

    Поведение:
    - Для каждой картинки сохраняем флаг is_color (True/False)
      по детектору nearly_gray.
    - В __getitem__:
        * если is_color=True  -> вернём [3,H,W]
        * если is_color=False -> вернём [1,H,W]
    - Смешивание RGB и grayscale в одних roots теперь допускается;
      режим работы AE будет определяться на уровне тренера.
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
                    # если не смогли открыть — считаем цветной, чтобы не завалить всё в gray
                    flags.append(True)
            self._flags = flags

    def __len__(self) -> int:
        return len(self.paths)

    def is_color_flags(self) -> List[bool]:
        """
        Возвращает список flag'ов True/False для каждого файла.
        Если флаги ещё не просканированы и scan_flags=True, сканируем лениво.
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

        # Определяем, цветная картинка или нет
        if self._flags is not None:
            is_color = self._flags[idx]
        else:
            with Image.open(p) as im_tmp:
                is_color = detect_is_color_pil(im_tmp) if self.scan_flags else True

        with Image.open(p) as im:
            if is_color:
                # 3-канальный RGB
                img = im.convert("RGB").resize((self.size, self.size), Image.BICUBIC)
                np_img = np.array(img, dtype=np.uint8, copy=True)      # [H,W,3]
                x = torch.from_numpy(np_img).permute(2, 0, 1).to(torch.float32).div_(255.0)  # [3,H,W]
            else:
                # 1-канальный grayscale
                img = im.convert("L").resize((self.size, self.size), Image.BICUBIC)
                np_img = np.array(img, dtype=np.uint8, copy=True)      # [H,W]
                x = torch.from_numpy(np_img).unsqueeze(0).to(torch.float32).div_(255.0)      # [1,H,W]

        return x, is_color, p

class CollateWithFlags:
    """
    Пикабельный collate_fn, который:
    - принимает batch = список (x, is_color, path), где x: [C,H,W], C=1 или 3
    - конвертирует все x к expected_c (1 или 3) с предупреждениями
    - возвращает:
        xs    : [B, expected_c, H, W]
        flags : [B] bool (оригинальный is_color)
        paths : список путей
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
                # нужно конвертировать
                if ec == 3 and C == 1:
                    # grayscale -> RGB
                    x_conv = x.repeat(3, 1, 1)
                    print(f"[WARN] converting GRAY -> RGB for {p} (AE mode={mode})")
                elif ec == 1 and C == 3:
                    # RGB -> grayscale (Y) через ту же формулу, что to_luma_Y
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
    Обёртка для совместимости со старым кодом.
    Возвращает пикабельный объект CollateWithFlags.
    """
    return CollateWithFlags(expected_c, ae_mode)

class AlternatingBatchSampler(Sampler):
    """
    1:1 batches of color vs gray indices (fallback to what's available if one side runs out).
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
            # первая половина — цвет
            for _ in range(self.half):
                if pc < len(ic):
                    batch.append(ic[pc]); pc += 1
                elif pg < len(ig):
                    batch.append(ig[pg]); pg += 1
            # вторая половина — серые
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


def decide_ae_mode_from_dataset(train_ds: MixedImageDataset,
                                max_sample: int = 100) -> Tuple[str, int]:
    """
    Режим определяем по флагам train_ds.is_color_flags().

    Возвращает:
      ae_mode: "rgb" или "gray"
      expected_c: 3 или 1

    Логика:
      - берём до max_sample случайных индексов
      - если среди них присутствуют обе группы:
          * считаем min(count_color, count_gray) как "несоответствующие"
          * если >=1 -> warning
          * если >5  -> RuntimeError
      - режим выбираем по большинству: если color >= gray -> "rgb", иначе "gray"
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

    # определяем режим по большинству
    if n_color >= n_gray:
        ae_mode = "rgb"
        mismatches = n_gray  # gray-примеры в RGB-режиме
    else:
        ae_mode = "gray"
        mismatches = n_color # color-примеры в GRAY-режиме

    if mismatches >= 1:
        print(f"[WARN] dataset not homogeneous in 100-sample check: "
              f"{n_color} color, {n_gray} gray (mode={ae_mode}, mismatches={mismatches}).")
    if mismatches > 5:
        raise RuntimeError(
            f"Too many mismatching images ({mismatches}) in 100-sample check. "
            f"Приведи датасет в порядок (или разнеси по отдельным train-папкам)."
        )

    expected_c = 3 if ae_mode == "rgb" else 1
    print(f"[MODE] AE training mode: {ae_mode.upper()} "
          f"(expected channels={expected_c}) based on {k} samples: "
          f"{n_color} color, {n_gray} gray.")

    return ae_mode, expected_c

# =========================
# -------- MODEL ----------
# =========================

class ConvGNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.gn   = nn.GroupNorm(num_groups=min(32, out_ch), num_channels=out_ch)
        self.act  = nn.GELU()
    def forward(self, x):
        return self.act(self.gn(self.conv(x)))


class EncoderY(nn.Module):
    """
    Generic encoder used both for Y (in_ch=1) and CbCr (in_ch=2).

    Produces:
      - s['s64']   = [B, d_skip, 64, 64]  (skip)
      - s['latent']= [B, d_lat,  32, 32]  (latent)
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
    Generic decoder; typically out_ch=1 for Y branch or out_ch=2 for CbCr branch.

    32->64 (d_lat->d_skip), fuse skip64, then 64->128->256->512 -> out_ch channels.
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
    AE v2.0 with separate Y and CbCr branches.

    - Accepts [B,3,H,W] RGB or [B,1,H,W] grayscale in [0,1].
    - Internally converts RGB -> YCbCr and runs:
        * Y branch (1ch) through enc_y/dec_y
        * CbCr branch (2ch) through enc_c/dec_c
    - For grayscale input, only Y branch is used.
    - External WM injection surfaces remain:
        * wm_lat, wm_skip  -> Y-branch latent and skip64
        * wm_cbcr_64       -> CbCr residual at 64x64, upsampled to full-res
    """
    def __init__(self):
        super().__init__()
        # Y (luma) branch
        self.enc_y = EncoderY(in_ch=1)
        self.dec_y = DecoderY(out_ch=1)
        # CbCr (chroma) branch – only used when input is RGB
        self.enc_c = EncoderY(in_ch=2)
        self.dec_c = DecoderY(out_ch=2)
        # Backward-compatible aliases (old code may look at .enc/.dec)
        self.enc = self.enc_y
        self.dec = self.dec_y

    # ===== Base forward (pretraining / plain reconstruction) =====
    def forward_plain(self, x01: torch.Tensor) -> torch.Tensor:
        if x01.size(1) == 3:
            # RGB → YCbCr, run both branches
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
        return self.forward_plain(x01)

    # ===== External WM surfaces (kept API-compatible) =====
    @staticmethod
    def _l2_normalize_per_sample(t: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        B = t.size(0)
        n = t.flatten(1).norm(p=2, dim=1).clamp_min(eps)
        return t / n.view(B, 1, 1, 1)

    @staticmethod
    def _ensure_size(t: torch.Tensor, size_hw: Tuple[int, int]) -> torch.Tensor:
        if t.shape[-2:] == size_hw:
            return t
        return F.interpolate(t, size_hw, mode="bilinear", align_corners=False)

    def embed_external_wm(self,
                          x01: torch.Tensor,
                          wm_lat: Optional[torch.Tensor] = None,    # [B,1024,32,32] for Y-branch
                          wm_skip: Optional[torch.Tensor] = None,   # [B,512,64,64]  for Y-branch
                          wm_cbcr_64: Optional[torch.Tensor] = None,# [B,2,64,64], RGB only
                          alpha_lat: float = 0.10,
                          alpha_skip: float = 0.10,
                          alpha_cbcr: float = 0.05,
                          roi_lat_32: Optional[torch.Tensor] = None,# [B,1,32,32] 0/1
                          roi_skip_64: Optional[torch.Tensor] = None # [B,1,64,64] 0/1
                          ) -> torch.Tensor:
        """
        External WM injection (no training logic here).

        Surfaces:
        - wm_lat   : injected into Y-branch latent [B,1024,32,32]
        - wm_skip  : injected into Y-branch skip64 [B,512,64,64]
        - wm_cbcr_64: chroma residuals ΔCb,ΔCr at 64×64 (RGB only), upsampled to full-res.

        All wm tensors are L2-normalized per-sample; spatial size is auto-resized.
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
            # Decode chroma and optionally add chroma WM residuals
            assert s_c is not None
            c_hat = self.dec_c(s_c)                               # [B,2,H,W]
            cb_hat, cr_hat = c_hat[:, 0:1], c_hat[:, 1:2]

            if wm_cbcr_64 is not None:
                # external chroma residuals (ΔCb, ΔCr @64x64) -> upscale to full resolution
                wm_cbcr_64 = self._ensure_size(wm_cbcr_64, (64, 64))
                assert wm_cbcr_64.size(1) == 2, \
                    f"wm_cbcr_64 must have 2 channels (got {wm_cbcr_64.size(1)})"
                wm_cbcr_64 = self._l2_normalize_per_sample(wm_cbcr_64)
                d_cbcr = F.interpolate(wm_cbcr_64, scale_factor=8, mode="bilinear", align_corners=False)
                d_cb, d_cr = d_cbcr[:, 0:1], d_cbcr[:, 1:2]
                cb_hat = torch.clamp(cb_hat + alpha_cbcr * d_cb, 0.0, 1.0)
                cr_hat = torch.clamp(cr_hat + alpha_cbcr * d_cr, 0.0, 1.0)

            return ycbcr_to_rgb(y_hat, cb_hat, cr_hat)
        else:
            # grayscale: only Y branch is active
            return y_hat

    # Conveniences with explicit types
    def embed_external_wm_rgb(self, x_rgb: torch.Tensor, **kwargs) -> torch.Tensor:
        assert x_rgb.size(1) == 3, "x_rgb must be [B,3,H,W]"
        return self.embed_external_wm(x_rgb, **kwargs)

    def embed_external_wm_gray(self, x_gray: torch.Tensor, **kwargs) -> torch.Tensor:
        assert x_gray.size(1) == 1, "x_gray must be [B,1,H,W]"
        return self.embed_external_wm(x_gray, **kwargs)


# =========================
# -------- TRAIN ----------
# =========================

def parse_args():
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
    p.add_argument("--scan-flags", action="store_true", default=True, help="scan image color/gray flags at init")
    return p.parse_args()

def tv_loss(z: torch.Tensor) -> torch.Tensor:
    """Isotropic TV (not used in pretrain, kept for possible future use)."""
    return (z[:,:,:-1,:]-z[:,:,1:,:]).abs().mean() + (z[:,:,:,:-1]-z[:,:,:,1:]).abs().mean()

def train_loop(args):
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

    # --- dataset roots: до 3 штук, берем только существующие ---

        # сюда позже можно добавить третий датасет (train, val)
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

    # ---- проверяем КАЖДЫЙ train-дatasets по 100 случайным картинкам ----
    train_modes: List[str] = []
    for tr in train_roots:
        ds_name = os.path.basename(os.path.normpath(tr)) or tr
        mode = detect_dataset_mode(tr, max_sample=100, name=ds_name)
        train_modes.append(mode)

    unique_train_modes = set(train_modes)
    if len(unique_train_modes) > 1:
        # в одном запуске смешивать RGB и GRAY датасеты нельзя
        raise RuntimeError(
            f"Mixed dataset types in one run: {train_modes}. "
            f"Все train-дatasets в этом запуске должны быть либо RGB, либо GRAYSCALE."
        )

    ae_mode = train_modes[0]                   # "rgb" или "gray"
    expected_c = 3 if ae_mode == "rgb" else 1
    print(f"[AE MODE] Global AE mode: {ae_mode.upper()} (expected channels={expected_c}) "
          f"for train_roots={train_roots}")

    # --- MixedImageDataset и collate ---
    train_ds = MixedImageDataset(train_roots, size=args.size, scan_flags=args.scan_flags)
    val_ds = MixedImageDataset(val_roots, size=args.size, scan_flags=args.scan_flags)
    collate_fn = make_collate_fn(expected_c, ae_mode)

    # build balanced batch sampler (1:1) как раньше
    flags = train_ds.is_color_flags()
    idx_color = [i for i,f in enumerate(flags) if f]
    idx_gray  = [i for i,f in enumerate(flags) if not f]

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
        batch_sampler = AlternatingBatchSampler(idx_color, idx_gray, batch_size=args.bs,
                                                drop_last=True, shuffle=True)
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

    # --- дальше код train_loop как в твоём файле: логирование, валидация ---
    # добавим только early stop после блока VAL.

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
                y_hat = ae(x)                          # AE сам посмотрит на C=1 или 3
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
                # запись в CSV оставляем как у тебя

        # ----- VAL -----
        ae.eval()
        val_m = SplitMeters()
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=args.amp):
            for xv, is_color_v, _ in val_loader:
                xv = xv.to(device, non_blocking=True)
                is_color_v = is_color_v.to(device)
                yv = ae(xv)
                update_split_metrics(val_m, xv, yv, is_color_v)

        print(f"[VAL E{epoch:02d}] "
              f"Y: psnr={val_m.psnr_y_all.avg:.2f} ssim={val_m.ssim_y_all.avg:.3f} | "
              f"Ycolor={val_m.psnr_y_color.avg:.2f}/{val_m.ssim_y_color.avg:.3f} "
              f"Ygray={val_m.psnr_y_gray.avg:.2f}/{val_m.ssim_y_gray.avg:.3f} | "
              f"RGBcolor={val_m.psnr_rgb_color.avg:.2f}/{val_m.ssim_rgb_color.avg:.3f} | "
              f"time={time.time()-t0:.1f}s")

        # --- early stop по SSIM_Y ---
        # SSIM у нас в [0,1], так что 98.9% = 0.989
        if val_m.ssim_y_all.avg >= 0.989:
            state = ae.module.state_dict() if isinstance(ae, nn.DataParallel) else ae.state_dict()
            early_path = CKPT_DIR / "universal_ae_early_stop.pth"
            torch.save(state, early_path)
            print(f"[EARLY STOP] SSIM_Y={val_m.ssim_y_all.avg:.4f} >= 0.989 at epoch {epoch}. "
                  f"Saved early-stop checkpoint to {early_path}")
            break

        # --- сохранение чекпоинтов и best по PSNR_Y как было ---
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
    # kept for compatibility
    return parse_args()

if __name__ == "__main__":
    args = parse_cli()
    train_loop(args)
