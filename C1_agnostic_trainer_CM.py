# ORNL/ANY C1 Trainer — ResNet34 (grayscale) with dual checkpoints and per-epoch confusion matrices


import os
import csv
import time
import argparse
from pathlib import Path
from typing import Optional, List, Tuple
from collections import Counter
from torch import amp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.data._utils.collate import default_convert
import random
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torchvision.models import resnet34

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # be robust to truncated images


# ----------------------------
# Argument parsing
# ----------------------------
def parse_args():
    """
    Parse command-line arguments for the dataset-agnostic C1 ResNet34 trainer.

    The interface is intentionally compact and exposes only the hyperparameters
    that are typically reported in the experimental sections of ML papers:
    learning rate, batch size, augmentation strength, number of epochs, and
    basic training-management options.
    """
    p = argparse.ArgumentParser(description="C1 ResNet34 (grayscale) trainer (dataset-agnostic)")

    # Basic configuration
    p.add_argument("--train_dir", type=str, required=True,
                   help="Path to the training image root (ImageFolder-style layout).")
    p.add_argument("--val_dir",   type=str, required=True,
                   help="Path to the validation image root (ImageFolder-style layout).")
    p.add_argument("--save_dir",  type=str, required=True,
                   help="Directory where checkpoints and logs will be written.")
    p.add_argument("--img_size",  type=int, default=512,
                   help="Target spatial resolution (images are resized to img_size×img_size).")
    p.add_argument("--epochs",    type=int, default=35,
                   help="Number of training epochs.")
    p.add_argument("--batch_size", type=int, default=32,
                   help="Mini-batch size.")
    p.add_argument("--lr",        type=float, default=1e-4,
                   help="Base learning rate for AdamW.")
    p.add_argument("--aug",       type=str, default="medium",
                   choices=["none", "light", "medium", "strong"],
                   help="Augmentation regime applied to the training set.")
    p.add_argument("--devices",   type=str, default="",
                   help="CUDA_VISIBLE_DEVICES string, e.g. '', '0', '1', '0,1'.")
    p.add_argument("--num_workers", type=int, default=8,
                   help="Number of worker processes for the DataLoader.")
    p.add_argument("--log_every",   type=int, default=25,
                   help="Logging frequency in training iterations.")
    p.add_argument("--classes_txt", type=str, default=None,
                   help="Optional text file that fixes the class order (one class name per line).")

    # Training management
    p.add_argument("--amp",        type=str, default="fp16",
                   choices=["none", "fp16", "bf16"],
                   help="Automatic mixed precision mode.")
    p.add_argument("--resume",     type=str, default=None,
                   help="Path to a checkpoint to resume from.")
    p.add_argument("--save_last",  type=int, default=1,
                   help="If non-zero, always keep last_state.pth up to date.")
    p.add_argument("--early_stop", type=int, default=0,
                   help="Enable early stopping if set to a non-zero value.")
    p.add_argument("--patience",   type=int, default=5,
                   help="Number of epochs without improvement before early stopping.")
    p.add_argument("--seed",       type=int, default=1337,
                   help="Random seed for reproducibility.")

    # Regularization
    p.add_argument("--label_smoothing", type=float, default=0.05,
                   help="Label smoothing factor for cross-entropy.")
    p.add_argument("--mixup_alpha", type=float, default=0.2,
                   help="Alpha parameter for the Beta distribution in MixUp.")
    p.add_argument("--mixup_p",     type=float, default=0.3,
                   help="Probability of applying MixUp to a given mini-batch.")

    return p.parse_args()


# ----------------------------
# Utils
# ----------------------------
class ForceRGB:
    """
    Simple transform that converts any PIL image to a valid 3-channel RGB image.

    This is used as the very first step in the preprocessing pipeline to make
    sure that unusual input modes such as 'I;16', 'F', 'P', 'LA', or 'RGBA'
    are handled in a uniform way before subsequent grayscale conversion.
    """
    def __call__(self, im: Image.Image) -> Image.Image:
        # Any 'I;16', 'F', 'P', 'LA', 'RGBA' etc. -> canonical 3-channel RGB.
        return im.convert("RGB")


def set_seed(seed: int):
    """
    Set all relevant random seeds (Python, NumPy, PyTorch) for reproducibility.

    Note that full determinism is not guaranteed because some CUDA operations
    are inherently non-deterministic, but this function substantially reduces
    run-to-run variability.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def format_confmat(cm: torch.Tensor, classes: List[str]) -> str:
    """
    Format a confusion matrix as a human-readable string table.

    The rows correspond to ground-truth classes and the columns to predictions.
    An additional column with row-wise sums is appended at the end.
    """
    n = cm.size(0)
    colw = max(5, max(len(c) for c in classes) + 1)
    header = " " * colw + "".join(f"{c:>{colw}}" for c in classes) + " | sum\n"
    sep = "-" * (colw * (n + 1) + 6) + "\n"
    lines = [header, sep]
    for i in range(n):
        row = "".join(f"{int(cm[i, j]):>{colw}}" for j in range(n))
        lines.append(f"{classes[i]:>{colw}}{row} | {int(cm[i].sum().item())}\n")
    lines.append(sep)
    return "".join(lines)


def write_csv_row(csv_path: Path, header: List[str], row: List):
    """
    Append a single row to a CSV file, creating the file and writing the header
    on the first invocation.
    """
    exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(header)
        w.writerow(row)


def macro_f1_from_cm(cm: torch.Tensor) -> Tuple[float, List[float]]:
    """
    Compute macro-averaged F1 score from a confusion matrix.

    Args:
        cm: Confusion matrix of shape [C, C], where C is the number of classes.

    Returns:
        macro_f1: Scalar F1 score averaged uniformly over classes.
        per_class_f1: List of per-class F1 scores.
    """
    cm = cm.float()
    tp = cm.diag()
    pred = cm.sum(0).clamp_min(1e-9)
    true = cm.sum(1).clamp_min(1e-9)
    prec = (tp / pred)
    rec  = (tp / true)
    f1 = (2 * prec * rec / (prec + rec).clamp_min(1e-9)).nan_to_num(0.0)
    return float(f1.mean().item()), f1.tolist()


# ----------------------------
# Datasets & Transforms
# ----------------------------
class ImageFolderFlex(ImageFolder):
    """
    ImageFolder variant with optional fixed class ordering via a text file.

    If `classes` is provided, the class names and their order are taken from
    that list, and the directory is required to contain all of them. This is
    useful to guarantee that train/val splits share identical label ordering,
    which is important for consistent checkpoint reuse.
    """
    def __init__(self, root: str, classes: Optional[List[str]] = None, **kwargs):
        self._forced_classes = classes
        super().__init__(root, **kwargs)

    def find_classes(self, directory: str):
        if self._forced_classes is None:
            return super().find_classes(directory)
        available, _ = super().find_classes(directory)
        missing = [c for c in self._forced_classes if c not in available]
        if missing:
            raise FileNotFoundError(f"[classes] Missing in {directory}: {missing}")
        return self._forced_classes, {c: i for i, c in enumerate(self._forced_classes)}


def load_classes_from_txt(path: Optional[str]) -> Optional[List[str]]:
    """
    Load a list of class names from a text file, one class per non-empty line.

    If `path` is None, the function returns None and the default ImageFolder
    class discovery is used instead.
    """
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"--classes_txt not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def list_images(root: str) -> List[str]:
    """
    Recursively collect all image file paths under the given root directory.

    The function filters files by a fixed set of common image extensions and
    returns a flat list of absolute paths.
    """
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    paths = []
    rpath = Path(root)
    if not rpath.exists():
        return []
    for p in rpath.rglob("*"):
        if p.suffix.lower() in exts:
            paths.append(str(p))
    return paths


def detect_is_color_pil(img: Image.Image, nearly_gray_thresh: int = 2) -> bool:
    """
    Heuristic detector of color vs grayscale images.

    Returns:
        True  if the image is considered color,
        False if the image is considered grayscale.

    Logic:
        * If the PIL mode is single-channel (L, I;16, I, F, 1), the image is
          treated as grayscale.
        * Otherwise the image is converted to RGB and the maximum absolute
          difference between channel pairs (R-G, R-B, G-B) is computed.
          If this maximum difference is less than or equal to the threshold,
          the image is considered nearly gray and thus treated as grayscale.
    """
    if img.mode in ("L", "I;16", "I", "F", "1"):
        return False
    rgb = img.convert("RGB")
    arr = np.asarray(rgb, dtype=np.int16)
    diff_rg = np.abs(arr[..., 0] - arr[..., 1])
    diff_rb = np.abs(arr[..., 0] - arr[..., 2])
    diff_gb = np.abs(arr[..., 1] - arr[..., 2])
    maxdiff = int(max(
        diff_rg.max(initial=0),
        diff_rb.max(initial=0),
        diff_gb.max(initial=0),
    ))
    return maxdiff > nearly_gray_thresh


def detect_dataset_mode(root: str,
                        max_sample: int = 100,
                        name: Optional[str] = None) -> str:
    """
    Infer whether a dataset is predominantly RGB or grayscale.

    The function samples at most `max_sample` images from `root` and classifies
    each image as color or grayscale using `detect_is_color_pil`. It then
    returns one of two string flags: 'rgb' or 'gray'.

    If color and grayscale images are substantially mixed inside `root`, the
    function:
        * allows up to 5 minority samples with a warning,
        * raises a RuntimeError if more than 5 mismatches are observed.
    """
    files = list_images(root)
    if not files:
        raise RuntimeError(f"[C1] No images found under dataset root: {root}")

    k = min(max_sample, len(files))
    sample_paths = random.sample(files, k)

    n_color = 0
    n_gray = 0
    for p in sample_paths:
        try:
            with Image.open(p) as im:
                is_color = detect_is_color_pil(im)
        except Exception:
            # If an image cannot be opened, treat it as color to avoid
            # falsely collapsing the dataset into purely grayscale.
            is_color = True
        if is_color:
            n_color += 1
        else:
            n_gray += 1

    mismatches = min(n_color, n_gray)
    mode = "rgb" if n_color >= n_gray else "gray"

    ds_name = name or os.path.basename(os.path.normpath(root)) or root
    print(f"[C1 DATASET CHECK] {ds_name}: mode={mode.upper()}, "
          f"n_color={n_color}, n_gray={n_gray}, sampled={k}, mismatches={mismatches}")

    if mismatches > 5:
        raise RuntimeError(
            f"[C1] Dataset '{ds_name}' under {root} is not homogeneous enough: "
            f"{n_color} color, {n_gray} gray in {k} samples (mismatches={mismatches}). "
            f"Please separate RGB and GRAY images into different roots."
        )
    if mismatches >= 1:
        print(f"[WARN] C1 dataset '{ds_name}' has some mixed images "
              f"(mismatches={mismatches} <= 5). Continuing, but this is potentially suspicious.")

    return mode


def build_transforms(img_size: int, aug: str) -> Tuple[T.Compose, T.Compose]:
    """
    Construct training and validation transform pipelines.

    All images are first mapped to RGB, then to single-channel grayscale, and
    finally resized to the target resolution. The training pipeline optionally
    adds geometric and photometric augmentations depending on the chosen level.
    """
    base = [
        ForceRGB(),
        T.Grayscale(num_output_channels=1),
        T.Resize(img_size, interpolation=T.InterpolationMode.BILINEAR),
    ]

    if aug == "none":
        train_tf = T.Compose(base + [
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ])
    elif aug == "light":
        train_tf = T.Compose(base + [
            T.RandomAffine(
                degrees=2, translate=(0.01, 0.01), scale=(0.99, 1.01), shear=1.0,
                interpolation=T.InterpolationMode.BILINEAR, fill=128
            ),
            T.RandomHorizontalFlip(p=0.10),
            T.ToTensor(), T.Normalize([0.5], [0.5]),
        ])
    elif aug == "medium":
        train_tf = T.Compose(base + [
            T.RandomAffine(
                degrees=3, translate=(0.02, 0.02), scale=(0.98, 1.02), shear=1.0,
                interpolation=T.InterpolationMode.BILINEAR, fill=128
            ),
            # If image orientation is semantically critical, this probability can be set to 0.0.
            T.RandomHorizontalFlip(p=0.10),
            T.ColorJitter(brightness=0.08, contrast=0.08),
            T.ToTensor(), T.Normalize([0.5], [0.5]),
            T.RandomErasing(p=0.15, scale=(0.01, 0.03), ratio=(0.3, 3.3), value=0),
        ])
    else:  # strong augmentation (slightly reduced compared to an aggressive "heavy" policy)
        train_tf = T.Compose(base + [
            T.RandomAffine(
                degrees=8, translate=(0.03, 0.03), scale=(0.96, 1.04), shear=3.0,
                interpolation=T.InterpolationMode.BILINEAR, fill=128
            ),
            T.RandomHorizontalFlip(p=0.35),
            T.RandomPerspective(distortion_scale=0.05, p=0.10),
            T.GaussianBlur(kernel_size=3, sigma=(0.4, 1.0)),
            T.ToTensor(), T.Normalize([0.5], [0.5]),
        ])

    val_tf = T.Compose([
        ForceRGB(),
        T.Grayscale(num_output_channels=1),
        T.Resize(img_size, interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(), T.Normalize([0.5], [0.5]),
    ])
    return train_tf, val_tf


def safe_collate(batch):
    """
    Windows-friendly collate function.

    Ensures that all tensors are contiguous and stored in resizable memory
    blocks before stacking them into a batch. This avoids subtle issues with
    some PyTorch backends on Windows.
    """
    imgs, labels = [], []
    for sample in batch:
        img, lab = sample[:2]
        if isinstance(img, torch.Tensor):
            img = img.contiguous().clone()
        else:
            img = default_convert(img).contiguous().clone()
        imgs.append(img)
        labels.append(int(lab))
    return torch.stack(imgs, 0), torch.tensor(labels, dtype=torch.long)


def maybe_make_weighted_sampler(ds, imbalance_ratio: float = 1.8):
    """
    Optionally construct a WeightedRandomSampler if the class distribution is imbalanced.

    The sampler is created only if the ratio between the most frequent and the
    least frequent class exceeds `imbalance_ratio`. Otherwise, the function
    returns None and the standard shuffle-based sampling is used.
    """
    # In recent torchvision versions ImageFolder exposes .targets; fall back to .samples otherwise.
    targets = getattr(ds, "targets", None)
    if targets is None:
        targets = [y for _, y in ds.samples]
    cnt = Counter(targets)
    if not cnt:
        return None
    cmax, cmin = max(cnt.values()), min(cnt.values())
    if cmin == 0 or (cmax / cmin) < imbalance_ratio:
        return None
    class_w = {c: 1.0 / max(1, cnt[c]) for c in cnt}
    weights = [class_w[y] for y in targets]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


# ----------------------------
# Model
# ----------------------------
class ResNet34Gray(nn.Module):
    """
    ResNet34 backbone adapted for grayscale inputs with a lightweight classifier head.

    Modifications relative to the canonical ResNet34:
        * The first convolution is changed to accept a single input channel.
        * The fully connected layer is replaced with a dropout-regularized
          linear classifier with `num_classes` outputs.
    """
    def __init__(self, num_classes: int):
        super().__init__()
        m = resnet34(weights=None)
        old = m.conv1
        m.conv1 = nn.Conv2d(1, old.out_channels, kernel_size=old.kernel_size,
                            stride=old.stride, padding=old.padding, bias=False)
        nn.init.kaiming_normal_(m.conv1.weight, mode="fan_out", nonlinearity="relu")
        inf = m.fc.in_features
        m.fc = nn.Sequential(nn.Dropout(0.25), nn.Linear(inf, num_classes))
        self.m = m

    def forward(self, x):  # x: B×1×H×W, normalized to [-1, 1] if used via the RGB wrapper
        return self.m(x)


class C1GrayWrapper(nn.Module):
    """
    Wrapper that exposes the 1-channel classifier as an RGB model.

    The wrapper assumes that the input is an RGB image normalized to [-1, 1].
    It performs the following steps:

        RGB[-1,1] → denormalization to [0,1] → luminance Y (BT.601) → re-normalization to [-1,1]
        → C1 grayscale classifier.

    The underlying 1-channel C1 model is kept in eval mode with gradients disabled.
    """
    def __init__(self, c1_1ch: nn.Module):
        super().__init__()
        self.c1 = c1_1ch.eval()
        for p in self.c1.parameters():
            p.requires_grad_(False)
        self.rgb2y = nn.Conv2d(3, 1, kernel_size=1, bias=False)
        with torch.no_grad():
            w = torch.tensor([0.2989, 0.5870, 0.1140], dtype=torch.float32).view(1, 3, 1, 1)
            self.rgb2y.weight.copy_(w)
        for p in self.rgb2y.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x_rgb_norm):  # x_rgb_norm: B×3×H×W in [-1, 1]
        x01 = x_rgb_norm * 0.5 + 0.5
        y01 = self.rgb2y(x01)
        y = (y01 - 0.5) / 0.5
        return self.c1(y)


# ----------------------------
# Losses & MixUp
# ----------------------------
class LabelSmoothingCE(nn.Module):
    """
    Cross-entropy loss with label smoothing.

    Label smoothing is implemented by constructing a softened target distribution
    and computing its cross-entropy with the model log-probabilities.
    """
    def __init__(self, smoothing: float = 0.0):
        super().__init__()
        assert 0.0 <= smoothing < 1.0
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        n_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)
        if self.smoothing > 0:
            with torch.no_grad():
                true_dist = torch.zeros_like(log_probs)
                true_dist.fill_(self.smoothing / (n_classes - 1))
                true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
            return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))
        return F.nll_loss(log_probs, target)


def apply_mixup(x: torch.Tensor, y: torch.Tensor, alpha: float):
    """
    Apply MixUp augmentation to a mini-batch.

    Args:
        x: Input tensor of shape [B, C, H, W].
        y: Integer class labels of shape [B].
        alpha: Beta distribution parameter; if <= 0, MixUp is disabled.

    Returns:
        x_mixed: Mixed inputs.
        y_a, y_b: Paired labels for the MixUp criterion.
        lam: Mixing coefficient λ.
    """
    if alpha <= 0:
        return x, y, y, 1.0
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx, :], y, y[idx], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Standard MixUp criterion that interpolates the loss for two label targets.

    The base loss `criterion` is typically LabelSmoothingCE with non-zero
    smoothing, but the formulation is generic and can be used with any
    cross-entropy-like loss that accepts logits and integer class labels.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b,)


# ----------------------------
# Train / Eval
# ----------------------------
@torch.no_grad()
def compute_confusion_matrix(model: nn.Module, loader: DataLoader, device: torch.device,
                             n_classes: int, amp_dtype=None) -> Tuple[torch.Tensor, float, float]:
    """
    Evaluate a model on a given DataLoader and accumulate a confusion matrix.

    Returns:
        cm:      Confusion matrix of shape [C, C].
        loss:    Mean cross-entropy loss over the dataset.
        acc:     Mean accuracy over the dataset.
    """
    model.eval()
    cm = torch.zeros(n_classes, n_classes, dtype=torch.long)
    total, correct, total_loss = 0, 0, 0.0
    criterion_eval = LabelSmoothingCE(smoothing=0.0).to(device)

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with amp.autocast("cuda", enabled=(amp_dtype is not None), dtype=amp_dtype):
            logits = model(x)
            loss = criterion_eval(logits, y)
        total_loss += loss.item() * x.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)
        for t, p in zip(y.view(-1), preds.view(-1)):
            cm[t.long(), p.long()] += 1

    return cm, total_loss / max(1, total), correct / max(1, total)


def train_one_epoch(model: nn.Module, loader: DataLoader, device: torch.device,
                    optimizer: torch.optim.Optimizer, criterion, n_classes: int,
                    mixup_alpha: float, mixup_p: float,
                    epoch: int, total_epochs: int,
                    log_every: int, amp_dtype=None, scaler=None,
                    use_channels_last: bool = True, mixup_stop_epoch: int = 10) -> Tuple[float, float, torch.Tensor]:
    """
    Train the classifier for a single epoch.

    The routine supports:
        * optional channels-last memory format for improved GPU throughput,
        * probabilistic MixUp with a warm-start schedule (disabled after a
          fixed number of epochs),
        * mixed-precision training via torch.amp.

    Returns:
        mean_loss:   Average training loss over the epoch.
        mean_acc:    Average training accuracy over the epoch.
        cm:          Confusion matrix accumulated over the epoch.
    """
    model.train()
    running_loss, running_correct, total = 0.0, 0, 0
    cm = torch.zeros(n_classes, n_classes, dtype=torch.long)
    n_batches = max(1, len(loader))
    t0, t_first = time.time(), None

    for step, (x, y) in enumerate(loader, start=1):
        if t_first is None:
            t_first = time.time() - t0
            print(f"[E{epoch}/{total_epochs}] first batch ready in {t_first:.1f}s")

        x = x.to(device, non_blocking=True)
        if use_channels_last:
            x = x.to(memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True)

        use_mix = (epoch <= mixup_stop_epoch) and (mixup_alpha > 0.0) and (torch.rand(1).item() < mixup_p)

        with amp.autocast("cuda", enabled=(amp_dtype is not None), dtype=amp_dtype):
            if use_mix:
                x, y_a, y_b, lam = apply_mixup(x, y, mixup_alpha)
                logits = model(x)
                loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
                preds = torch.argmax(logits, dim=1)
                y_for_stats = y_a
            else:
                logits = model(x)
                loss = criterion(logits, y)
                preds = torch.argmax(logits, dim=1)
                y_for_stats = y

        optimizer.zero_grad(set_to_none=True)
        if scaler and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * x.size(0)
        running_correct += (preds == y_for_stats).sum().item()
        total += x.size(0)
        for t, p in zip(y_for_stats.view(-1), preds.view(-1)):
            cm[t.long(), p.long()] += 1

        if (step % max(1, log_every) == 0) or (step == 1) or (step == n_batches):
            avg_loss = running_loss / max(1, total)
            acc = running_correct / max(1, total)
            dt = time.time() - t0
            ips = total / max(1e-6, dt)
            print(f"[E{epoch}/{total_epochs}] step {step:>4}/{n_batches:<4} | "
                  f"loss={avg_loss:.4f} acc={acc:.4f} | imgs/s={ips:.1f}")

    return running_loss / max(1, total), running_correct / max(1, total), cm


# ----------------------------
# Checkpoints
# ----------------------------
def get_state_dict_for_save(model: nn.Module) -> dict:
    """
    Retrieve a plain state_dict from a model, unwrapping DataParallel if needed.
    """
    return model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()


def save_dual_checkpoints(model_1ch: nn.Module, classes: List[str], save_dir: Path, image_size: int):
    """
    Export two complementary checkpoints for deployment:

        1) best_raw.pth
           A pure 1-channel model that expects grayscale input normalized with
           mean/std = [0.5] / [0.5].

        2) best_RGBwrapped.pth
           An RGB wrapper around the same backbone, using a fixed RGB→Y mapping
           and expecting 3-channel inputs normalized with mean/std = [0.5, 0.5, 0.5].

    The function does not modify the live model instance used during training.
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    # RAW (1ch)
    sd = get_state_dict_for_save(model_1ch)
    torch.save({
        "state_dict": sd,
        "wrapped": False,
        "classes": classes,
        "expected_in_ch": 1,
        "img_size": image_size,
        "normalize_mean": [0.5],
        "normalize_std":  [0.5],
        "version": "C1_RS34_GRAY_v1",
    }, save_dir / "best_raw.pth")

    # RGB-wrapped (3ch)
    tmp = ResNet34Gray(num_classes=len(classes))  # CPU-only temporary instance
    tmp.load_state_dict(sd)
    wrapped = C1GrayWrapper(tmp).cpu()
    torch.save({
        "state_dict": wrapped.state_dict(),
        "wrapped": True,
        "classes": classes,
        "expected_in_ch": 3,
        "img_size": image_size,
        "normalize_mean": [0.5, 0.5, 0.5],
        "normalize_std":  [0.5, 0.5, 0.5],
        "version": "C1_RS34_GRAY_v1_RGBwrapped",
    }, save_dir / "best_RGBwrapped.pth")
    print(f"[SAVE] {save_dir/'best_raw.pth'}")
    print(f"[SAVE] {save_dir/'best_RGBwrapped.pth'}")


# ----------------------------
# Main
# ----------------------------
def main():
    """
    Entry point for training the grayscale C1 classifier.

    The function performs:
        * dataset mode validation (RGB vs GRAY),
        * construction of training/validation datasets and loaders,
        * model/optimizer/scheduler/AMP initialization,
        * epoch-wise training and validation with per-epoch confusion matrices,
        * dual-checkpoint export (raw 1-channel and RGB-wrapped variants),
        * optional early stopping and 'last' checkpoint saving.
    """
    args = parse_args()
    set_seed(args.seed)

    # Set visible GPUs BEFORE the first CUDA call so that torch.cuda.* sees the restriction.
    if args.devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.devices

    # Enable fast matrix multiplications (TF32) on Ampere+ if available.
    try:
        # torch.backends.cuda.matmul.fp32_precision: "high" ~ TF32, "ieee" ~ strict FP32.
        torch.backends.cuda.matmul.fp32_precision = "high"
        torch.backends.cudnn.conv.fp32_precision = "tf32"
    except Exception:
        pass

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    train_dir = Path(args.train_dir).expanduser()
    val_dir   = Path(args.val_dir).expanduser()
    save_dir  = Path(args.save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)

    # ---- C1 dataset mode check (RGB vs GRAY) ----
    train_mode = detect_dataset_mode(str(train_dir), max_sample=100, name="TRAIN")
    val_mode = detect_dataset_mode(str(val_dir), max_sample=100, name="VAL")

    if train_mode != val_mode:
        raise RuntimeError(
            f"[C1] Train/Val dataset type mismatch: train={train_mode}, val={val_mode}. "
            f"In a single run, C1 train/val datasets must be of the same type (RGB or GRAY)."
        )

    c1_dataset_mode = train_mode
    print(f"[C1 MODE] dataset_type={c1_dataset_mode.upper()} "
          f"(detected from TRAIN/VAL roots)")

    classes_fixed = load_classes_from_txt(args.classes_txt)

    print(f"[gpu ] visible_count={torch.cuda.device_count()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"[gpu ] {i}: {torch.cuda.get_device_name(i)}")
    print(f"[paths] TRAIN={train_dir}\n[paths] VAL  ={val_dir}\n[save ] {save_dir}")
    print(f"[conf ] img_size={args.img_size}  epochs={args.epochs}  bs={args.batch_size}  aug={args.aug}")
    print(f"[conf ] lr={args.lr}  ls={args.label_smoothing}  mixup_alpha={args.mixup_alpha} p={args.mixup_p}")
    print(f"[dev  ] device={device}  cuda={torch.cuda.is_available()}")

    # Transforms & Datasets
    train_tf, val_tf = build_transforms(args.img_size, args.aug)
    train_ds = ImageFolderFlex(str(train_dir), classes=classes_fixed, transform=train_tf)
    val_ds   = ImageFolderFlex(str(val_dir),   classes=classes_fixed, transform=val_tf)

    if train_ds.classes != val_ds.classes:
        raise ValueError(f"[classes] Train/Val mismatch:\nTrain={train_ds.classes}\nVal  ={val_ds.classes}")
    classes = train_ds.classes
    n_classes = len(classes)
    print(f"[data ] classes ({n_classes}): {classes}")
    print(f"[data ] train images={len(train_ds)} | val images={len(val_ds)}")

    # Automatically enable a WeightedRandomSampler in the presence of explicit imbalance.
    train_sampler = maybe_make_weighted_sampler(train_ds, imbalance_ratio=1.8)

    # DataLoaders (automatic pin_memory / persistent_workers / prefetch_factor configuration)
    use_pin = torch.cuda.is_available()
    use_pw  = args.num_workers > 0
    prefetch = 3

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=(train_sampler is None), sampler=train_sampler,
        num_workers=args.num_workers, pin_memory=use_pin,
        drop_last=True, collate_fn=safe_collate,
        persistent_workers=use_pw, prefetch_factor=(prefetch if use_pw else None),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=use_pin,
        drop_last=False, collate_fn=safe_collate,
        persistent_workers=use_pw, prefetch_factor=(prefetch if use_pw else None),
    )
    print(f"[data ] train batches={len(train_loader)} | val batches={len(val_loader)}")

    # Model
    model = ResNet34Gray(num_classes=n_classes)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"[model] Using DataParallel on {torch.cuda.device_count()} GPUs")
    else:
        print("[model] Single GPU; DataParallel is OFF")
    model = model.to(device)
    if torch.cuda.is_available():
        model = model.to(memory_format=torch.channels_last)

    # AMP init
    if args.amp == "none":
        amp_dtype, scaler = None, amp.GradScaler("cuda", enabled=False)
    elif args.amp == "bf16":
        amp_dtype, scaler = torch.bfloat16, amp.GradScaler("cuda", enabled=False)
    else:  # fp16
        amp_dtype, scaler = torch.float16, amp.GradScaler("cuda", enabled=True)

    # Optimizer / Scheduler / Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    criterion = LabelSmoothingCE(smoothing=args.label_smoothing).to(device)

    # Resume
    start_epoch, best_val_acc = 1, -1.0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        best_val_acc = ck.get("best_val_acc", -1.0)
        start_epoch = ck.get("epoch", 0) + 1
        print(f"[resume] epoch={start_epoch} best_val_acc={best_val_acc:.4f}")

    # Logs
    metrics_csv = save_dir / "metrics.csv"
    header = ["epoch", "lr", "train_loss", "train_acc",
              "val_loss", "val_acc", "val_macroF1", "epoch_time_sec"]
    no_improve = 0

    MIXUP_STOP_EPOCH = 10  # MixUp is disabled after this epoch; no additional CLI flag is used.

    # Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        # Train
        train_loss, train_acc, train_cm = train_one_epoch(
            model, train_loader, device, optimizer, criterion, n_classes,
            mixup_alpha=args.mixup_alpha, mixup_p=args.mixup_p,
            epoch=epoch, total_epochs=args.epochs, log_every=args.log_every,
            amp_dtype=amp_dtype, scaler=scaler, use_channels_last=torch.cuda.is_available(),
            mixup_stop_epoch=MIXUP_STOP_EPOCH
        )
        print(f"[epoch {epoch:02d}/{args.epochs}] TRAIN done: loss={train_loss:.4f} acc={train_acc:.4f}",
              flush=True)

        # Validation
        val_cm, val_loss, val_acc = compute_confusion_matrix(
            model, val_loader, device, n_classes, amp_dtype=amp_dtype
        )
        val_f1_macro, _ = macro_f1_from_cm(val_cm)

        # Scheduler step
        scheduler.step()

        # Save confusion matrices to CSV (per-epoch diagnostics).
        cm_train_path = save_dir / f"confmat_train_epoch_{epoch:02d}.csv"
        cm_val_path   = save_dir / f"confmat_val_epoch_{epoch:02d}.csv"
        with open(cm_train_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([""] + classes)
            for i, row in enumerate(train_cm.tolist()):
                w.writerow([classes[i]] + row)
        with open(cm_val_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([""] + classes)
            for i, row in enumerate(val_cm.tolist()):
                w.writerow([classes[i]] + row)

        # Console log
        lr_now = optimizer.param_groups[0]["lr"]
        dt = time.time() - t0
        print(
            f"\n[epoch {epoch:02d}/{args.epochs}] lr={lr_now:.3e}  "
            f"train: loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val: loss={val_loss:.4f} acc={val_acc:.4f} F1={val_f1_macro:.4f}  (time {dt:.1f}s)",
            flush=True
        )

        print("[confmat][train]\n" + format_confmat(train_cm, classes))
        print("[confmat][val]\n" + format_confmat(val_cm, classes))

        # CSV log row
        write_csv_row(
            metrics_csv, header,
            [epoch, lr_now, f"{train_loss:.6f}", f"{train_acc:.6f}",
             f"{val_loss:.6f}", f"{val_acc:.6f}", f"{val_f1_macro:.6f}", f"{dt:.2f}"]
        )

        # Track best & export dual checkpoints
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_for_export = model if not isinstance(model, nn.DataParallel) else model.module
            save_dual_checkpoints(model_for_export, classes, save_dir, args.img_size)
            no_improve = 0
        else:
            no_improve += 1

        # Save last
        if args.save_last:
            torch.save({
                "epoch": epoch,
                "model": get_state_dict_for_save(model),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_val_acc": best_val_acc,
                "classes": classes,
            }, save_dir / "last_state.pth")

        # Early stopping
        if args.early_stop and no_improve >= args.patience:
            print(f"[early-stop] no improvement for {args.patience} epochs; stop at {epoch}")
            break

    print(f"\n[done] Best val acc = {best_val_acc:.4f}")
    print(f"[tips] Use best_RGBwrapped.pth in your RGB watermark pipeline; "
          f"best_raw.pth for pure grayscale inference.")


if __name__ == "__main__":
    main()
