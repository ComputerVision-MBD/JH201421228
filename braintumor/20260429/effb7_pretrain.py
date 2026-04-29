from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import albumentations as A
import nibabel as nib
import numpy as np
import optuna
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

DATA_ROOT = Path("/mnt/c/Users/park/Desktop/dataroot/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData")
MODS = ("flair", "t1", "t1ce", "t2")
ENCODER_CHOICES = ("tf_efficientnet_b2", "tf_efficientnet_b3")


def find_cases(root: Path):
    cases = []
    for case_dir in sorted(root.iterdir()):
        if not case_dir.is_dir():
            continue
        images = [next(case_dir.glob(f"*_{mod}.nii*"), None) for mod in MODS]
        mask = next(case_dir.glob("*_seg.nii*"), None)
        if all(images) and mask is not None:
            cases.append((tuple(images), mask))
    return cases


def split_cases(cases, val_ratio=0.2, seed=42):
    shuffled = list(cases)
    random.Random(seed).shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_ratio))
    return shuffled[n_val:], shuffled[:n_val]


def normalize(img):
    brain = img > 0
    if brain.any():
        img = img.copy()
        img[brain] = (img[brain] - img[brain].mean()) / (img[brain].std() + 1e-6)
    return img


class BratsDataset(Dataset):
    def __init__(self, cases, train: bool, cache_mode: str):
        self.train = train
        self.cache_mode = cache_mode
        self.case_index: list[tuple[tuple[Path, ...], Path]] = []
        self.case_cache: dict[int, tuple[list[np.ndarray], np.ndarray]] = {}
        self.samples: list[tuple[int, int]] = []
        self.aug = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
            ]
        )

        for case_id, (image_paths, mask_path) in enumerate(cases):
            image_paths = tuple(Path(path) for path in image_paths)
            mask_path = Path(mask_path)
            self.case_index.append((image_paths, mask_path))

            mask_volume = np.asarray(nib.load(str(mask_path)).dataobj, dtype=np.float32)
            tumor_slices = np.where(mask_volume.any(axis=(0, 1)))[0]
            for z in tumor_slices:
                self.samples.append((case_id, int(z)))

    def __len__(self):
        return len(self.samples)

    def _load_case_arrays(self, case_id: int):
        if self.cache_mode == "lazy" and case_id in self.case_cache:
            return self.case_cache[case_id]

        image_paths, mask_path = self.case_index[case_id]
        images = [np.asarray(nib.load(str(path)).dataobj, dtype=np.float32) for path in image_paths]
        mask = np.asarray(nib.load(str(mask_path)).dataobj, dtype=np.float32)

        if self.cache_mode == "lazy":
            self.case_cache[case_id] = (images, mask)
        return images, mask

    def __getitem__(self, idx):
        case_id, z = self.samples[idx]
        image_volumes, mask_volume = self._load_case_arrays(case_id)

        image = np.stack([normalize(volume[..., z]) for volume in image_volumes], axis=-1)
        mask = (mask_volume[..., z] > 0).astype(np.float32)

        if self.train:
            out = self.aug(image=image, mask=mask)
            image, mask = out["image"], out["mask"]

        image = np.ascontiguousarray(image.transpose(2, 0, 1))
        mask = np.ascontiguousarray(mask[None])
        return torch.from_numpy(image), torch.from_numpy(mask)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class EfficientNetUNet(nn.Module):
    def __init__(self, encoder_name: str, pretrained: bool = True):
        super().__init__()
        self.encoder_name = encoder_name
        self.encoder = timm.create_model(
            encoder_name,
            features_only=True,
            in_chans=4,
            pretrained=pretrained,
        )
        c1, c2, c3, c4, c5 = self.encoder.feature_info.channels()
        decoder_channels = {
            "tf_efficientnet_b2": (256, 160, 96, 64),
            "tf_efficientnet_b3": (288, 160, 96, 64),
        }[encoder_name]
        d4, d3, d2, d1 = decoder_channels
        self.dec4 = DecoderBlock(c5, c4, d4)
        self.dec3 = DecoderBlock(d4, c3, d3)
        self.dec2 = DecoderBlock(d3, c2, d2)
        self.dec1 = DecoderBlock(d2, c1, d1)
        self.head = nn.Conv2d(d1, 1, kernel_size=1)

    def forward(self, x):
        input_size = x.shape[-2:]
        f1, f2, f3, f4, f5 = self.encoder(x)
        x = self.dec4(f5, f4)
        x = self.dec3(x, f3)
        x = self.dec2(x, f2)
        x = self.dec1(x, f1)
        x = self.head(x)
        return F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)

    def encoder_parameters(self):
        return self.encoder.parameters()

    def decoder_parameters(self):
        modules = [self.dec4, self.dec3, self.dec2, self.dec1, self.head]
        for module in modules:
            yield from module.parameters()


def dice_loss(logits, masks):
    probs = torch.sigmoid(logits)
    inter = (probs * masks).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
    return 1 - ((2 * inter + 1e-6) / (union + 1e-6)).mean()


def dice_score(logits, masks):
    preds = (torch.sigmoid(logits) > 0.5).float()
    inter = (preds * masks).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
    return float(((2 * inter + 1e-6) / (union + 1e-6)).mean())


def parse_args():
    parser = argparse.ArgumentParser(description="EfficientNet-B2/B3 pretrained U-Net with efficient Optuna training")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--encoder-name", type=str, default="tf_efficientnet_b3", choices=ENCODER_CHOICES)
    parser.add_argument("--trials", type=int, default=16)
    parser.add_argument("--epochs", "--search-epochs", dest="search_epochs", type=int, default=4)
    parser.add_argument("--final-epochs", type=int, default=24)
    parser.add_argument("--limit-cases", type=int, default=20)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--target-effective-batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--n-startup-trials", type=int, default=5)
    parser.add_argument("--study-name", type=str, default=None)
    parser.add_argument("--storage", type=str, default=None, help="예: sqlite:///optuna.db")
    parser.add_argument("--result-root", type=Path, default=Path(__file__).with_name("optimize_results"))
    parser.add_argument("--cache-mode", type=str, default="lazy", choices=("none", "lazy"))
    parser.add_argument("--eta-min-scale", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--warmup-start-factor", type=float, default=0.3)
    parser.add_argument("--freeze-encoder-epochs", type=int, default=1)
    parser.add_argument("--early-stop-patience", type=int, default=6)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-3)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--encoder-lr-min", type=float, default=5e-6)
    parser.add_argument("--encoder-lr-max", type=float, default=5e-5)
    parser.add_argument("--decoder-lr-min", type=float, default=5e-5)
    parser.add_argument("--decoder-lr-max", type=float, default=5e-4)
    parser.add_argument("--weight-decay-min", type=float, default=1e-6)
    parser.add_argument("--weight-decay-max", type=float, default=1e-3)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True, help="Use pretrained encoder weights")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True, help="Use CUDA AMP when available")
    return parser.parse_args()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logger(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(log_path.stem)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    return logger


def make_run_dir(result_root: Path, study_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = result_root / f"{study_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def format_hparams(hparams: dict) -> str:
    return ", ".join(
        f"{key}={value:.1e}" if isinstance(value, float) else f"{key}={value}"
        for key, value in hparams.items()
    )


def get_group_lrs(optimizer):
    enc_lr = optimizer.param_groups[0]["lr"]
    dec_lr = optimizer.param_groups[1]["lr"]
    return enc_lr, dec_lr


def compute_accumulation_steps(batch_size: int, target_effective_batch: int) -> int:
    if target_effective_batch <= 0:
        return 1
    return max(1, math.ceil(target_effective_batch / batch_size))


def set_encoder_trainable(model: EfficientNetUNet, trainable: bool):
    for param in model.encoder_parameters():
        param.requires_grad = trainable


def autocast_context(device: torch.device, amp_enabled: bool):
    if amp_enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def make_loaders(train_ds, valid_ds, batch_size: int, num_workers: int, device: torch.device):
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        common["persistent_workers"] = True

    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        **common,
    )
    valid_loader = DataLoader(
        valid_ds,
        shuffle=False,
        **common,
    )
    return train_loader, valid_loader


def make_optimizer(model, encoder_lr: float, decoder_lr: float, weight_decay: float):
    return torch.optim.AdamW(
        [
            {"params": model.encoder_parameters(), "lr": encoder_lr},
            {"params": model.decoder_parameters(), "lr": decoder_lr},
        ],
        weight_decay=weight_decay,
    )


class WarmupCosineScheduler:
    def __init__(
        self,
        optimizer,
        total_epochs: int,
        warmup_epochs: int,
        warmup_start_factor: float,
        eta_min_scale: float,
    ):
        self.optimizer = optimizer
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.total_epochs = max(1, total_epochs)
        self.warmup_epochs = min(max(0, warmup_epochs), max(0, self.total_epochs - 1))
        self.warmup_start_factor = min(max(warmup_start_factor, 1e-3), 1.0)
        self.eta_min_scale = min(max(eta_min_scale, 0.0), 1.0)
        self.finished_epochs = 0
        self._set_scale(self._scale_for_finished_epochs(0))

    def _scale_for_finished_epochs(self, finished_epochs: int):
        if self.warmup_epochs > 0 and finished_epochs < self.warmup_epochs:
            progress = finished_epochs / self.warmup_epochs
            return self.warmup_start_factor + (1.0 - self.warmup_start_factor) * progress

        cosine_epochs = max(1, self.total_epochs - self.warmup_epochs)
        progress = (finished_epochs - self.warmup_epochs) / cosine_epochs
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.eta_min_scale + (1.0 - self.eta_min_scale) * cosine

    def _set_scale(self, scale: float):
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base_lr * scale

    def step(self):
        self.finished_epochs += 1
        self._set_scale(self._scale_for_finished_epochs(self.finished_epochs))


def build_warmup_cosine_scheduler(
    optimizer,
    total_epochs: int,
    warmup_epochs: int,
    warmup_start_factor: float,
    eta_min_scale: float,
):
    return WarmupCosineScheduler(
        optimizer=optimizer,
        total_epochs=total_epochs,
        warmup_epochs=warmup_epochs,
        warmup_start_factor=warmup_start_factor,
        eta_min_scale=eta_min_scale,
    )


def run_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    train: bool,
    desc: str,
    hparams: dict,
    accumulation_steps: int,
    amp_enabled: bool,
    grad_clip_norm: float,
    freeze_encoder: bool,
):
    model.train() if train else model.eval()
    if freeze_encoder:
        model.encoder.eval()
    set_encoder_trainable(model, trainable=not freeze_encoder)

    loss_sum, dice_sum = 0.0, 0.0
    enc_lr, dec_lr = get_group_lrs(optimizer)
    bar = tqdm(loader, desc=desc, leave=False)
    hp_text = format_hparams(hparams)

    if train:
        optimizer.zero_grad(set_to_none=True)

    for step, (images, masks) in enumerate(bar, start=1):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with autocast_context(device, amp_enabled):
            logits = model(images)
            loss = F.binary_cross_entropy_with_logits(logits, masks) + dice_loss(logits, masks)

        if train:
            scaled_loss = loss / accumulation_steps
            scaler.scale(scaled_loss).backward()

            should_step = step % accumulation_steps == 0 or step == len(loader)
            if should_step:
                if grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        loss_sum += float(loss.item())
        dice = dice_score(logits, masks)
        dice_sum += dice
        bar.set_postfix(
            hp=hp_text,
            enc_lr=f"{enc_lr:.1e}",
            dec_lr=f"{dec_lr:.1e}",
            loss=f"{loss.item():.4f}",
            dice=f"{dice:.4f}",
        )

    return loss_sum / len(loader), dice_sum / len(loader)


def main():
    args = parse_args()
    study_name = args.study_name or f"brats_{args.encoder_name}_efficient"
    run_dir = make_run_dir(args.result_root, study_name)
    search_best_model_path = run_dir / "search_best_model.pt"
    final_model_path = run_dir / "final_model.pt"
    result_path = run_dir / "best_params.json"
    log_path = run_dir / "optimize.log"
    config_path = run_dir / "config.json"
    trials_csv_path = run_dir / "trials.csv"

    logger = setup_logger(log_path)
    seed_everything(args.seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    config = {
        "data_root": str(args.data_root),
        "encoder_name": args.encoder_name,
        "trials": args.trials,
        "search_epochs": args.search_epochs,
        "final_epochs": args.final_epochs,
        "limit_cases": args.limit_cases,
        "batch_sizes": args.batch_sizes,
        "target_effective_batch": args.target_effective_batch,
        "seed": args.seed,
        "num_workers": args.num_workers,
        "n_startup_trials": args.n_startup_trials,
        "study_name": study_name,
        "storage": args.storage,
        "result_root": str(args.result_root),
        "run_dir": str(run_dir),
        "cache_mode": args.cache_mode,
        "scheduler": "warmup_cosine",
        "eta_min_scale": args.eta_min_scale,
        "warmup_epochs": args.warmup_epochs,
        "warmup_start_factor": args.warmup_start_factor,
        "freeze_encoder_epochs": args.freeze_encoder_epochs,
        "early_stop_patience": args.early_stop_patience,
        "early_stop_min_delta": args.early_stop_min_delta,
        "grad_clip_norm": args.grad_clip_norm,
        "encoder_lr_min": args.encoder_lr_min,
        "encoder_lr_max": args.encoder_lr_max,
        "decoder_lr_min": args.decoder_lr_min,
        "decoder_lr_max": args.decoder_lr_max,
        "weight_decay_min": args.weight_decay_min,
        "weight_decay_max": args.weight_decay_max,
        "pretrained": args.pretrained,
        "amp": args.amp,
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    cases = find_cases(args.data_root)
    if args.limit_cases:
        cases = cases[: args.limit_cases]
    if len(cases) < 2:
        raise RuntimeError(f"학습 가능한 케이스가 부족합니다: {args.data_root}")

    train_cases, valid_cases = split_cases(cases, seed=args.seed)
    train_ds = BratsDataset(train_cases, train=True, cache_mode=args.cache_mode)
    valid_ds = BratsDataset(valid_cases, train=False, cache_mode=args.cache_mode)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = args.amp and device.type == "cuda"
    global_best = {"dice": -1.0}

    print(f"device={device}")
    print(f"cases: train={len(train_cases)}, valid={len(valid_cases)}")
    print(f"slices: train={len(train_ds)}, valid={len(valid_ds)}")
    print(f"search epochs per trial={args.search_epochs}")
    print(f"final epochs={args.final_epochs}")
    print("scheduler=warmup_cosine")
    print(f"encoder={args.encoder_name}")
    print(f"encoder lr range={args.encoder_lr_min:.1e} ~ {args.encoder_lr_max:.1e}")
    print(f"decoder lr range={args.decoder_lr_min:.1e} ~ {args.decoder_lr_max:.1e}")
    print(f"target effective batch={args.target_effective_batch}")
    print(f"freeze encoder epochs={args.freeze_encoder_epochs}")
    print(f"cache mode={args.cache_mode}")
    print(f"amp={amp_enabled}")
    print(f"result dir={run_dir}")
    logger.info("device=%s", device)
    logger.info("cases: train=%d, valid=%d", len(train_cases), len(valid_cases))
    logger.info("slices: train=%d, valid=%d", len(train_ds), len(valid_ds))
    logger.info("search epochs per trial=%d", args.search_epochs)
    logger.info("final epochs=%d", args.final_epochs)
    logger.info("scheduler=warmup_cosine | eta_min_scale=%.2f | warmup_epochs=%d | warmup_start_factor=%.2f", args.eta_min_scale, args.warmup_epochs, args.warmup_start_factor)
    logger.info("encoder=%s | pretrained=%s", args.encoder_name, args.pretrained)
    logger.info("encoder lr range=%.2e ~ %.2e", args.encoder_lr_min, args.encoder_lr_max)
    logger.info("decoder lr range=%.2e ~ %.2e", args.decoder_lr_min, args.decoder_lr_max)
    logger.info("weight decay range=%.2e ~ %.2e", args.weight_decay_min, args.weight_decay_max)
    logger.info("target effective batch=%d", args.target_effective_batch)
    logger.info("freeze encoder epochs=%d | early stop patience=%d | grad clip=%.2f", args.freeze_encoder_epochs, args.early_stop_patience, args.grad_clip_norm)
    logger.info("cache mode=%s | amp=%s", args.cache_mode, amp_enabled)
    logger.info("result dir=%s", run_dir)

    def objective(trial: optuna.Trial):
        seed_everything(args.seed + trial.number)

        encoder_lr = trial.suggest_float("encoder_lr", args.encoder_lr_min, args.encoder_lr_max, log=True)
        decoder_lr = trial.suggest_float("decoder_lr", args.decoder_lr_min, args.decoder_lr_max, log=True)
        weight_decay = trial.suggest_float("weight_decay", args.weight_decay_min, args.weight_decay_max, log=True)
        batch_size = trial.suggest_categorical("batch_size", args.batch_sizes)
        accumulation_steps = compute_accumulation_steps(batch_size, args.target_effective_batch)
        effective_batch_size = batch_size * accumulation_steps
        hparams = {
            "encoder_lr": encoder_lr,
            "decoder_lr": decoder_lr,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "accum_steps": accumulation_steps,
            "eff_batch": effective_batch_size,
            "encoder": args.encoder_name,
        }
        logger.info("trial %03d start | hp=%s", trial.number, format_hparams(hparams))

        train_loader, valid_loader = make_loaders(
            train_ds,
            valid_ds,
            batch_size=batch_size,
            num_workers=args.num_workers,
            device=device,
        )

        model = EfficientNetUNet(args.encoder_name, pretrained=args.pretrained).to(device)
        optimizer = make_optimizer(model, encoder_lr=encoder_lr, decoder_lr=decoder_lr, weight_decay=weight_decay)
        scheduler = build_warmup_cosine_scheduler(
            optimizer,
            total_epochs=args.search_epochs,
            warmup_epochs=args.warmup_epochs,
            warmup_start_factor=args.warmup_start_factor,
            eta_min_scale=args.eta_min_scale,
        )
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

        best_valid_dice = 0.0
        for epoch in range(1, args.search_epochs + 1):
            freeze_encoder = epoch <= args.freeze_encoder_epochs
            start_enc_lr, start_dec_lr = get_group_lrs(optimizer)
            train_loss, train_dice = run_epoch(
                model,
                train_loader,
                optimizer,
                scaler,
                device,
                train=True,
                desc=f"trial {trial.number} train {epoch}",
                hparams=hparams,
                accumulation_steps=accumulation_steps,
                amp_enabled=amp_enabled,
                grad_clip_norm=args.grad_clip_norm,
                freeze_encoder=freeze_encoder,
            )
            valid_loss, valid_dice = run_epoch(
                model,
                valid_loader,
                optimizer,
                scaler,
                device,
                train=False,
                desc=f"trial {trial.number} valid {epoch}",
                hparams=hparams,
                accumulation_steps=1,
                amp_enabled=amp_enabled,
                grad_clip_norm=args.grad_clip_norm,
                freeze_encoder=False,
            )
            scheduler.step()
            end_enc_lr, end_dec_lr = get_group_lrs(optimizer)
            best_valid_dice = max(best_valid_dice, valid_dice)

            message = (
                f"trial {trial.number:03d} search epoch {epoch:02d} | "
                f"enc_lr={start_enc_lr:.2e}->{end_enc_lr:.2e}, "
                f"dec_lr={start_dec_lr:.2e}->{end_dec_lr:.2e}, "
                f"wd={weight_decay:.2e}, micro_bs={batch_size}, accum={accumulation_steps}, eff_bs={effective_batch_size}, "
                f"freeze_enc={freeze_encoder} | "
                f"train loss={train_loss:.4f}, dice={train_dice:.4f} | "
                f"valid loss={valid_loss:.4f}, dice={valid_dice:.4f}"
            )
            print(message)
            logger.info(message)

            trial.report(valid_dice, epoch)
            if epoch >= max(2, args.freeze_encoder_epochs + 1) and trial.should_prune():
                logger.info("trial %03d pruned at epoch %02d | valid dice=%.4f", trial.number, epoch, valid_dice)
                raise optuna.TrialPruned()

        if best_valid_dice > global_best["dice"]:
            global_best["dice"] = best_valid_dice
            torch.save(model.state_dict(), search_best_model_path)
            logger.info("new search best model saved | dice=%.4f | path=%s", best_valid_dice, search_best_model_path)

        logger.info("trial %03d end | best valid dice=%.4f", trial.number, best_valid_dice)
        return best_valid_dice

    sampler = optuna.samplers.TPESampler(seed=args.seed, n_startup_trials=args.n_startup_trials, multivariate=True)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=max(1, args.freeze_encoder_epochs + 1))
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name=study_name,
        storage=args.storage,
        load_if_exists=args.storage is not None,
    )
    study.optimize(objective, n_trials=args.trials)

    best_params = study.best_params
    best_accumulation_steps = compute_accumulation_steps(best_params["batch_size"], args.target_effective_batch)
    best_effective_batch_size = best_params["batch_size"] * best_accumulation_steps
    final_best_dice = None

    if args.final_epochs > 0:
        final_hparams = {
            **best_params,
            "accum_steps": best_accumulation_steps,
            "eff_batch": best_effective_batch_size,
            "encoder": args.encoder_name,
        }
        logger.info("final training start | hp=%s", format_hparams(final_hparams))
        train_loader, valid_loader = make_loaders(
            train_ds,
            valid_ds,
            batch_size=best_params["batch_size"],
            num_workers=args.num_workers,
            device=device,
        )
        model = EfficientNetUNet(args.encoder_name, pretrained=args.pretrained).to(device)
        optimizer = make_optimizer(
            model,
            encoder_lr=best_params["encoder_lr"],
            decoder_lr=best_params["decoder_lr"],
            weight_decay=best_params["weight_decay"],
        )
        scheduler = build_warmup_cosine_scheduler(
            optimizer,
            total_epochs=args.final_epochs,
            warmup_epochs=args.warmup_epochs,
            warmup_start_factor=args.warmup_start_factor,
            eta_min_scale=args.eta_min_scale,
        )
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

        best_final_valid_dice = 0.0
        epochs_without_improvement = 0
        for epoch in range(1, args.final_epochs + 1):
            freeze_encoder = epoch <= args.freeze_encoder_epochs
            start_enc_lr, start_dec_lr = get_group_lrs(optimizer)
            train_loss, train_dice = run_epoch(
                model,
                train_loader,
                optimizer,
                scaler,
                device,
                train=True,
                desc=f"final train {epoch}",
                hparams=final_hparams,
                accumulation_steps=best_accumulation_steps,
                amp_enabled=amp_enabled,
                grad_clip_norm=args.grad_clip_norm,
                freeze_encoder=freeze_encoder,
            )
            valid_loss, valid_dice = run_epoch(
                model,
                valid_loader,
                optimizer,
                scaler,
                device,
                train=False,
                desc=f"final valid {epoch}",
                hparams=final_hparams,
                accumulation_steps=1,
                amp_enabled=amp_enabled,
                grad_clip_norm=args.grad_clip_norm,
                freeze_encoder=False,
            )
            scheduler.step()
            end_enc_lr, end_dec_lr = get_group_lrs(optimizer)

            message = (
                f"final epoch {epoch:02d} | "
                f"enc_lr={start_enc_lr:.2e}->{end_enc_lr:.2e}, "
                f"dec_lr={start_dec_lr:.2e}->{end_dec_lr:.2e}, "
                f"wd={best_params['weight_decay']:.2e}, micro_bs={best_params['batch_size']}, "
                f"accum={best_accumulation_steps}, eff_bs={best_effective_batch_size}, freeze_enc={freeze_encoder} | "
                f"train loss={train_loss:.4f}, dice={train_dice:.4f} | "
                f"valid loss={valid_loss:.4f}, dice={valid_dice:.4f}"
            )
            print(message)
            logger.info(message)

            if valid_dice > best_final_valid_dice + args.early_stop_min_delta:
                best_final_valid_dice = valid_dice
                epochs_without_improvement = 0
                torch.save(model.state_dict(), final_model_path)
                logger.info("new final best model saved | dice=%.4f | path=%s", valid_dice, final_model_path)
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= args.early_stop_patience:
                logger.info(
                    "early stopping at epoch %02d | best valid dice=%.4f | patience=%d",
                    epoch,
                    best_final_valid_dice,
                    args.early_stop_patience,
                )
                break

        final_best_dice = best_final_valid_dice
        logger.info("final training end | best valid dice=%.4f", final_best_dice)

    result = {
        "search_best_dice": study.best_value,
        "best_params": best_params,
        "best_accumulation_steps": best_accumulation_steps,
        "best_effective_batch_size": best_effective_batch_size,
        "final_best_dice": final_best_dice,
        "scheduler": "warmup_cosine",
        "eta_min_scale": args.eta_min_scale,
        "warmup_epochs": args.warmup_epochs,
        "warmup_start_factor": args.warmup_start_factor,
        "freeze_encoder_epochs": args.freeze_encoder_epochs,
        "target_effective_batch": args.target_effective_batch,
        "cache_mode": args.cache_mode,
        "pretrained": args.pretrained,
        "amp": amp_enabled,
        "encoder_name": args.encoder_name,
    }
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame(
        [
            {
                "trial": trial.number,
                "state": str(trial.state),
                "value": trial.value,
                "accumulation_steps": compute_accumulation_steps(
                    int(trial.params["batch_size"]),
                    args.target_effective_batch,
                )
                if "batch_size" in trial.params
                else None,
                "effective_batch_size": (
                    int(trial.params["batch_size"])
                    * compute_accumulation_steps(int(trial.params["batch_size"]), args.target_effective_batch)
                )
                if "batch_size" in trial.params
                else None,
                **trial.params,
            }
            for trial in study.trials
        ]
    ).to_csv(trials_csv_path, index=False)

    print("\nBest result")
    print(json.dumps(result, indent=2))
    print(f"saved dir: {run_dir}")
    print(f"saved search model: {search_best_model_path}")
    if final_best_dice is not None:
        print(f"saved final model: {final_model_path}")
    print(f"saved params: {result_path}")
    logger.info("best result: %s", json.dumps(result, ensure_ascii=False))
    logger.info("saved search model: %s", search_best_model_path)
    if final_best_dice is not None:
        logger.info("saved final model: %s", final_model_path)
    logger.info("saved params: %s", result_path)
    logger.info("saved trials: %s", trials_csv_path)
    logger.info("saved config: %s", config_path)


if __name__ == "__main__":
    main()
