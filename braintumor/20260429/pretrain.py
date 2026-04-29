from __future__ import annotations

import argparse
import json
import logging
import os
import random
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
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

DATA_ROOT = Path("/mnt/c/Users/park/Desktop/dataroot/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData")
MODS = ("flair", "t1", "t1ce", "t2")


def find_cases(root: Path):
    cases = []
    for case_dir in sorted(root.iterdir()):
        if not case_dir.is_dir():
            continue
        images = [next(case_dir.glob(f"*_{mod}.nii*"), None) for mod in MODS]
        mask = next(case_dir.glob("*_seg.nii*"), None)
        if all(images) and mask is not None:
            cases.append((images, mask))
    return cases


def split_cases(cases, val_ratio=0.2, seed=42):
    random.Random(seed).shuffle(cases)
    n_val = max(1, int(len(cases) * val_ratio))
    return cases[n_val:], cases[:n_val]


def normalize(img):
    brain = img > 0
    if brain.any():
        img = img.copy()
        img[brain] = (img[brain] - img[brain].mean()) / (img[brain].std() + 1e-6)
    return img


class BratsDataset(Dataset):
    def __init__(self, cases, train: bool):
        self.train = train
        self.samples = []
        self.aug = A.Compose([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5)])

        for image_paths, mask_path in cases:
            mask = nib.load(str(mask_path)).get_fdata()
            tumor_slices = np.where(mask.any(axis=(0, 1)))[0]
            for z in tumor_slices:
                self.samples.append((image_paths, mask_path, int(z)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_paths, mask_path, z = self.samples[idx]

        image = []
        for path in image_paths:
            slice_2d = np.asarray(nib.load(str(path)).dataobj[..., z], dtype=np.float32)
            image.append(normalize(slice_2d))
        image = np.stack(image, axis=-1)

        mask = np.asarray(nib.load(str(mask_path)).dataobj[..., z], dtype=np.float32)
        mask = (mask > 0).astype(np.float32)

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


class TimmUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = timm.create_model("resnet18", features_only=True, in_chans=4, pretrained=True)
        c1, c2, c3, c4, c5 = self.encoder.feature_info.channels()
        self.dec4 = DecoderBlock(c5, c4, 256)
        self.dec3 = DecoderBlock(256, c3, 128)
        self.dec2 = DecoderBlock(128, c2, 64)
        self.dec1 = DecoderBlock(64, c1, 64)
        self.head = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        input_size = x.shape[-2:]
        f1, f2, f3, f4, f5 = self.encoder(x)
        x = self.dec4(f5, f4)
        x = self.dec3(x, f3)
        x = self.dec2(x, f2)
        x = self.dec1(x, f1)
        x = self.head(x)
        return F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)


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
    parser = argparse.ArgumentParser(description="Optuna TPE + CosineAnnealingLR for BraTS segmentation with pretrained encoder")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--epochs", "--search-epochs", dest="search_epochs", type=int, default=3)
    parser.add_argument("--final-epochs", type=int, default=15)
    parser.add_argument("--limit-cases", type=int, default=20)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--n-startup-trials", type=int, default=5)
    parser.add_argument("--study-name", type=str, default="brats_seg_optuna_cosine_pretrained")
    parser.add_argument("--storage", type=str, default=None, help="예: sqlite:///optuna.db")
    parser.add_argument("--result-root", type=Path, default=Path(__file__).with_name("optimize_results"))
    parser.add_argument("--eta-min", type=float, default=1e-6, help="CosineAnnealingLR minimum learning rate")
    parser.add_argument("--lr-min", type=float, default=3e-5, help="Optuna learning rate lower bound for pretrained encoder")
    parser.add_argument("--lr-max", type=float, default=1e-4, help="Optuna learning rate upper bound for pretrained encoder")
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


def run_epoch(model, loader, optimizer, device, train: bool, desc: str, hparams: dict):
    model.train() if train else model.eval()
    loss_sum, dice_sum = 0.0, 0.0
    current_lr = optimizer.param_groups[0]["lr"]

    bar = tqdm(loader, desc=desc, leave=False)
    hp_text = format_hparams(hparams)
    for images, masks in bar:
        images, masks = images.to(device), masks.to(device)

        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = F.binary_cross_entropy_with_logits(logits, masks) + dice_loss(logits, masks)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        loss_sum += float(loss.item())
        dice = dice_score(logits, masks)
        dice_sum += dice
        bar.set_postfix(
            hp=hp_text,
            lr=f"{current_lr:.1e}",
            loss=f"{loss.item():.4f}",
            dice=f"{dice:.4f}",
        )

    return loss_sum / len(loader), dice_sum / len(loader)


def make_loaders(train_ds, valid_ds, batch_size: int, num_workers: int):
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, valid_loader


def build_cosine_scheduler(optimizer, epochs: int, eta_min: float):
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs),
        eta_min=eta_min,
    )


def main():
    args = parse_args()
    run_dir = make_run_dir(args.result_root, args.study_name)
    search_best_model_path = run_dir / "search_best_model.pt"
    final_model_path = run_dir / "final_model.pt"
    result_path = run_dir / "best_params.json"
    log_path = run_dir / "optimize.log"
    config_path = run_dir / "config.json"
    trials_csv_path = run_dir / "trials.csv"

    logger = setup_logger(log_path)
    seed_everything(args.seed)

    config = {
        "data_root": str(args.data_root),
        "trials": args.trials,
        "search_epochs": args.search_epochs,
        "final_epochs": args.final_epochs,
        "limit_cases": args.limit_cases,
        "batch_sizes": args.batch_sizes,
        "seed": args.seed,
        "num_workers": args.num_workers,
        "n_startup_trials": args.n_startup_trials,
        "study_name": args.study_name,
        "storage": args.storage,
        "result_root": str(args.result_root),
        "run_dir": str(run_dir),
        "scheduler": "cosine",
        "eta_min": args.eta_min,
        "lr_min": args.lr_min,
        "lr_max": args.lr_max,
        "pretrained": True,
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    cases = find_cases(args.data_root)
    if args.limit_cases:
        cases = cases[: args.limit_cases]
    if len(cases) < 2:
        raise RuntimeError(f"학습 가능한 케이스가 부족합니다: {args.data_root}")

    train_cases, valid_cases = split_cases(cases, seed=args.seed)
    train_ds = BratsDataset(train_cases, train=True)
    valid_ds = BratsDataset(valid_cases, train=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    global_best = {"dice": -1.0}

    print(f"device={device}")
    print(f"cases: train={len(train_cases)}, valid={len(valid_cases)}")
    print(f"slices: train={len(train_ds)}, valid={len(valid_ds)}")
    print(f"search epochs per trial={args.search_epochs}")
    print(f"final epochs={args.final_epochs}")
    print("scheduler=cosine")
    print(f"lr range={args.lr_min:.1e} ~ {args.lr_max:.1e}")
    print("pretrained=True")
    print(f"result dir={run_dir}")
    logger.info("device=%s", device)
    logger.info("cases: train=%d, valid=%d", len(train_cases), len(valid_cases))
    logger.info("slices: train=%d, valid=%d", len(train_ds), len(valid_ds))
    logger.info("search epochs per trial=%d", args.search_epochs)
    logger.info("final epochs=%d", args.final_epochs)
    logger.info("scheduler=cosine | eta_min=%.2e", args.eta_min)
    logger.info("lr range=%.2e ~ %.2e", args.lr_min, args.lr_max)
    logger.info("pretrained=True")
    logger.info("result dir=%s", run_dir)

    def objective(trial: optuna.Trial):
        seed_everything(args.seed + trial.number)

        lr = trial.suggest_float("lr", args.lr_min, args.lr_max, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", args.batch_sizes)
        hparams = {
            "lr": lr,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "scheduler": "cosine",
            "pretrained": True,
        }
        logger.info("trial %03d start | hp=%s", trial.number, format_hparams(hparams))

        train_loader, valid_loader = make_loaders(
            train_ds,
            valid_ds,
            batch_size=batch_size,
            num_workers=args.num_workers,
        )

        model = TimmUNet().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = build_cosine_scheduler(optimizer, epochs=args.search_epochs, eta_min=args.eta_min)

        best_valid_dice = 0.0
        for epoch in range(1, args.search_epochs + 1):
            start_lr = optimizer.param_groups[0]["lr"]
            train_loss, train_dice = run_epoch(
                model,
                train_loader,
                optimizer,
                device,
                train=True,
                desc=f"trial {trial.number} train {epoch}",
                hparams=hparams,
            )
            valid_loss, valid_dice = run_epoch(
                model,
                valid_loader,
                optimizer,
                device,
                train=False,
                desc=f"trial {trial.number} valid {epoch}",
                hparams=hparams,
            )
            scheduler.step()
            end_lr = optimizer.param_groups[0]["lr"]
            best_valid_dice = max(best_valid_dice, valid_dice)

            message = (
                f"trial {trial.number:03d} search epoch {epoch:02d} | "
                f"lr={start_lr:.2e}->{end_lr:.2e}, wd={weight_decay:.2e}, bs={batch_size}, sched=cosine | "
                f"train loss={train_loss:.4f}, dice={train_dice:.4f} | "
                f"valid loss={valid_loss:.4f}, dice={valid_dice:.4f}"
            )
            print(message)
            logger.info(message)

            trial.report(valid_dice, epoch)
            if trial.should_prune():
                logger.info("trial %03d pruned at epoch %02d | valid dice=%.4f", trial.number, epoch, valid_dice)
                raise optuna.TrialPruned()

        if best_valid_dice > global_best["dice"]:
            global_best["dice"] = best_valid_dice
            torch.save(model.state_dict(), search_best_model_path)
            logger.info("new search best model saved | dice=%.4f | path=%s", best_valid_dice, search_best_model_path)

        logger.info("trial %03d end | best valid dice=%.4f", trial.number, best_valid_dice)
        return best_valid_dice

    sampler = optuna.samplers.TPESampler(seed=args.seed, n_startup_trials=args.n_startup_trials, multivariate=True)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=1)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=args.storage is not None,
    )
    study.optimize(objective, n_trials=args.trials)

    best_params = study.best_params
    final_best_dice = None

    if args.final_epochs > 0:
        final_hparams = {**best_params, "scheduler": "cosine", "pretrained": True}
        logger.info("final training start | hp=%s", format_hparams(final_hparams))
        train_loader, valid_loader = make_loaders(
            train_ds,
            valid_ds,
            batch_size=best_params["batch_size"],
            num_workers=args.num_workers,
        )
        model = TimmUNet().to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=best_params["lr"],
            weight_decay=best_params["weight_decay"],
        )
        scheduler = build_cosine_scheduler(optimizer, epochs=args.final_epochs, eta_min=args.eta_min)

        best_final_valid_dice = 0.0
        for epoch in range(1, args.final_epochs + 1):
            start_lr = optimizer.param_groups[0]["lr"]
            train_loss, train_dice = run_epoch(
                model,
                train_loader,
                optimizer,
                device,
                train=True,
                desc=f"final train {epoch}",
                hparams=final_hparams,
            )
            valid_loss, valid_dice = run_epoch(
                model,
                valid_loader,
                optimizer,
                device,
                train=False,
                desc=f"final valid {epoch}",
                hparams=final_hparams,
            )
            scheduler.step()
            end_lr = optimizer.param_groups[0]["lr"]

            message = (
                f"final epoch {epoch:02d} | "
                f"lr={start_lr:.2e}->{end_lr:.2e}, wd={best_params['weight_decay']:.2e}, "
                f"bs={best_params['batch_size']}, sched=cosine | "
                f"train loss={train_loss:.4f}, dice={train_dice:.4f} | "
                f"valid loss={valid_loss:.4f}, dice={valid_dice:.4f}"
            )
            print(message)
            logger.info(message)

            if valid_dice > best_final_valid_dice:
                best_final_valid_dice = valid_dice
                torch.save(model.state_dict(), final_model_path)
                logger.info("new final best model saved | dice=%.4f | path=%s", valid_dice, final_model_path)

        final_best_dice = best_final_valid_dice
        logger.info("final training end | best valid dice=%.4f", final_best_dice)

    result = {
        "search_best_dice": study.best_value,
        "best_params": best_params,
        "final_best_dice": final_best_dice,
        "scheduler": "cosine",
        "eta_min": args.eta_min,
        "lr_min": args.lr_min,
        "lr_max": args.lr_max,
        "pretrained": True,
    }
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame(
        [
            {
                "trial": trial.number,
                "state": str(trial.state),
                "value": trial.value,
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
