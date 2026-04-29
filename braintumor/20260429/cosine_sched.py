from __future__ import annotations

import argparse
import json
import logging
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn.functional as F
from code import DATA_ROOT, BratsDataset, TimmUNet, dice_loss, dice_score, find_cases, split_cases
from torch.utils.data import DataLoader
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Optuna TPE + CosineAnnealingLR for BraTS segmentation")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--epochs", "--search-epochs", dest="search_epochs", type=int, default=3)
    parser.add_argument("--final-epochs", type=int, default=15)
    parser.add_argument("--limit-cases", type=int, default=20)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--n-startup-trials", type=int, default=5)
    parser.add_argument("--study-name", type=str, default="brats_seg_optuna_cosine")
    parser.add_argument("--storage", type=str, default=None, help="예: sqlite:///optuna.db")
    parser.add_argument("--result-root", type=Path, default=Path(__file__).with_name("optimize_results"))
    parser.add_argument("--eta-min", type=float, default=1e-6, help="CosineAnnealingLR minimum learning rate")
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
    print(f"result dir={run_dir}")
    logger.info("device=%s", device)
    logger.info("cases: train=%d, valid=%d", len(train_cases), len(valid_cases))
    logger.info("slices: train=%d, valid=%d", len(train_ds), len(valid_ds))
    logger.info("search epochs per trial=%d", args.search_epochs)
    logger.info("final epochs=%d", args.final_epochs)
    logger.info("scheduler=cosine | eta_min=%.2e", args.eta_min)
    logger.info("result dir=%s", run_dir)

    def objective(trial: optuna.Trial):
        seed_everything(args.seed + trial.number)

        lr = trial.suggest_float("lr", 1e-5, 3e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", args.batch_sizes)
        hparams = {"lr": lr, "weight_decay": weight_decay, "batch_size": batch_size, "scheduler": "cosine"}
        logger.info("trial %03d start | hp=%s", trial.number, format_hparams(hparams))

        train_loader, valid_loader = make_loaders(train_ds, valid_ds, batch_size=batch_size, num_workers=args.num_workers)

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
        final_hparams = {**best_params, "scheduler": "cosine"}
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
