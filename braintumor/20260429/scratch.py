from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import nibabel as nib
import numpy as np
import torch
from monai.losses import DiceLoss
from monai.networks.nets import UNet
from torch.utils.data import DataLoader, Dataset

DATA_ROOT = Path("/mnt/c/Users/park/Desktop/dataroot/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BraTS 기본 segmentation 예제")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-cases", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-path", type=Path, default=Path(__file__).with_name("brats_seg_basic.pt"))
    return parser.parse_args()


def find_case_files(data_root: Path) -> list[tuple[Path, Path]]:
    cases = []
    for case_dir in sorted(data_root.iterdir()):
        if not case_dir.is_dir():
            continue
        flair = next(case_dir.glob("*_flair.nii*"), None)
        seg = next(case_dir.glob("*_seg.nii*"), None)
        if flair is not None and seg is not None:
            cases.append((flair, seg))
    return cases


def split_cases(cases: list[tuple[Path, Path]], val_ratio: float, seed: int) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    rng = random.Random(seed)
    rng.shuffle(cases)
    n_val = max(1, int(len(cases) * val_ratio))
    return cases[n_val:], cases[:n_val]


class BraTSSliceDataset(Dataset):
    def __init__(self, cases: list[tuple[Path, Path]]):
        self.samples: list[tuple[Path, Path, int]] = []
        for flair_path, seg_path in cases:
            seg = nib.load(str(seg_path)).get_fdata()
            tumor_slices = np.where(seg.any(axis=(0, 1)))[0]
            for z in tumor_slices:
                self.samples.append((flair_path, seg_path, int(z)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        flair_path, seg_path, z = self.samples[index]

        image = np.asarray(nib.load(str(flair_path)).dataobj[..., z], dtype=np.float32)
        mask = np.asarray(nib.load(str(seg_path)).dataobj[..., z], dtype=np.float32)

        image = (image - image.mean()) / (image.std() + 1e-6)
        mask = (mask > 0).astype(np.float32)

        image = torch.from_numpy(image[None])
        mask = torch.from_numpy(mask[None])
        return image, mask


def dice_score(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = (torch.sigmoid(logits) > 0.5).float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2 * intersection + 1e-6) / (union + 1e-6)
    return float(dice.mean().item())


def run_epoch(model, loader, optimizer, loss_fn, device: str, train: bool) -> tuple[float, float]:
    model.train() if train else model.eval()
    loss_sum = 0.0
    dice_sum = 0.0

    with torch.set_grad_enabled(train):
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            loss = loss_fn(logits, masks)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            loss_sum += float(loss.item())
            dice_sum += dice_score(logits, masks)

    return loss_sum / len(loader), dice_sum / len(loader)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cases = find_case_files(args.data_root)
    if args.limit_cases is not None:
        cases = cases[: args.limit_cases]
    if len(cases) < 2:
        raise SystemExit("학습할 케이스가 너무 적습니다.")

    train_cases, val_cases = split_cases(cases, val_ratio=args.val_ratio, seed=args.seed)
    train_ds = BraTSSliceDataset(train_cases)
    val_ds = BraTSSliceDataset(val_cases)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = DiceLoss(sigmoid=True)

    print(f"device: {device}")
    print(f"cases: train={len(train_cases)}, val={len(val_cases)}")
    print(f"slices: train={len(train_ds)}, val={len(val_ds)}")

    best_val_dice = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_dice = run_epoch(model, train_loader, optimizer, loss_fn, device, train=True)
        val_loss, val_dice = run_epoch(model, val_loader, optimizer, loss_fn, device, train=False)

        print(
            f"epoch {epoch:02d} | "
            f"train loss {train_loss:.4f}, train dice {train_dice:.4f} | "
            f"val loss {val_loss:.4f}, val dice {val_dice:.4f}"
        )

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), args.save_path)

    print(f"best val dice: {best_val_dice:.4f}")
    print(f"saved model: {args.save_path}")


if __name__ == "__main__":
    main()
