import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from mandelbrot_nn.data import IndexedTensorDataset, build_boundary_biased_dataset
from mandelbrot_nn.models import MLPRes, MLPFourierRes
from mandelbrot_nn.render import plot_model_heatmap_tiled
from mandelbrot_nn.utils import make_run_dir, save_config
from mandelbrot_nn.train import train_model


def load_cfg(path: str | None) -> dict:
    if path is None:
        return {}
    with open(path, "r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="experiments/configs/default.json")
    parser.add_argument("--tag", type=str, default="mandelbrot")
    args = parser.parse_args()

    # defaults
    cfg = {
        "xlim": (-2.4, 1.0),
        "res_for_ylim": (3840, 2160),
        "ycenter": 0.0,
        "max_iter_labels": 1000,
        "dataset_n_total": 1_000_000,
        "dataset_frac_boundary": 0.7,
        "boundary_band": (0.35, 0.95),
        "seed": 0,
        "model_name": "fourier_res",  # "mlp_res" | "fourier_res"
        "model_num_feats": 512,
        "model_sigma": 10.0,
        "model_hidden_dim": 512,
        "model_hidden_layers": 20,
        "model_act": "silu",
        "train_epochs": 100,
        "train_batch_size": 4096,
        "train_lr": 3e-4,
        "train_weight_decay": 1e-6,
        "train_grad_clip": 1.0,
        "train_amp": True,
        "preview_every": 1,
        "preview_res": (1920, 1080),
        "preview_tile": (512, 512),
        "final_res": (3840, 2160),
        "final_tile": (512, 512),
    }

    cfg.update(load_cfg(args.config))

    run_dir = make_run_dir(tag=args.tag)
    save_config(cfg, run_dir)
    print("Run dir:", run_dir)

    X, y, _ = build_boundary_biased_dataset(
        n_total=cfg["dataset_n_total"],
        frac_boundary=cfg["dataset_frac_boundary"],
        xlim=tuple(cfg["xlim"]),
        res_for_ylim=tuple(cfg["res_for_ylim"]),
        ycenter=cfg["ycenter"],
        max_iter=cfg["max_iter_labels"],
        band=tuple(cfg["boundary_band"]),
        seed=cfg["seed"],
    )

    print(
        "y stats:",
        "min", float(y.min()),
        "max", float(y.max()),
        "mean", float(y.mean()),
        "p50", float(np.quantile(y, 0.50)),
        "p99", float(np.quantile(y, 0.99)),
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=cfg["seed"], shuffle=True
    )

    train_ds = IndexedTensorDataset(X_train, y_train)
    val_ds = IndexedTensorDataset(X_val, y_val)

    if cfg["model_name"] == "fourier_res":
        model = MLPFourierRes(
            num_feats=cfg["model_num_feats"],
            sigma=cfg["model_sigma"],
            hidden_dim=cfg["model_hidden_dim"],
            num_blocks=cfg["model_hidden_layers"],
            act=cfg["model_act"],
            dropout=0.0,
        )
    else:
        model = MLPRes(
            hidden_dim=cfg["model_hidden_dim"],
            num_blocks=cfg["model_hidden_layers"],
            act=cfg["model_act"],
            dropout=0.0,
        )

    def render_fn(m, device, epoch: int):
        out_img = run_dir / "images" / f"render_epoch_{epoch:03d}.png"
        plot_model_heatmap_tiled(
            m,
            device,
            xlim=tuple(cfg["xlim"]),
            ycenter=cfg["ycenter"],
            res=tuple(cfg["preview_res"]),
            tile=tuple(cfg["preview_tile"]),
            fname=str(out_img),
            amp=cfg["train_amp"],
        )

    model = train_model(
        model,
        train_ds,
        val_ds,
        run_dir=run_dir,
        epochs=cfg["train_epochs"],
        batch_size=cfg["train_batch_size"],
        lr=cfg["train_lr"],
        weight_decay=cfg["train_weight_decay"],
        grad_clip=cfg["train_grad_clip"],
        amp=cfg["train_amp"],
        render_every=cfg["preview_every"],
        render_fn=render_fn,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    final_path = run_dir / "images" / "final_4k.png"
    plot_model_heatmap_tiled(
        model,
        device,
        xlim=tuple(cfg["xlim"]),
        ycenter=cfg["ycenter"],
        res=tuple(cfg["final_res"]),
        tile=tuple(cfg["final_tile"]),
        fname=str(final_path),
        amp=cfg["train_amp"],
    )


if __name__ == "__main__":
    main()
