from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


@torch.no_grad()
def eval_loss(model, loader, device, criterion):
    """Evaluate average loss on a dataloader.

    Args:
        model: PyTorch model.
        loader: Validation dataloader yielding (X, y, idx).
        device: Torch device.
        criterion: Loss function returning per-sample losses (reduction="none").

    Returns:
        Mean loss over all samples in the loader.
    """
    model.eval()
    tot = 0.0
    n = 0
    for Xb, yb, _ in loader:
        Xb = Xb.to(device)
        yb = yb.to(device)
        pred = model(Xb)
        loss = criterion(pred, yb).mean()
        tot += float(loss.item()) * Xb.size(0)
        n += Xb.size(0)
    return tot / max(1, n)


def train_model(
    model,
    train_dataset,
    val_dataset,
    *,
    run_dir,
    epochs=50,
    batch_size=4096,
    lr=3e-4,
    weight_decay=1e-6,
    grad_clip=1.0,
    amp=True,
    render_every=10,
    render_fn=None,
):
    """Train a regression model with stable defaults for Mandelbrot learning.

    Training setup:
      - SmoothL1 loss (robust to outliers and stable for this task)
      - AdamW optimizer
      - cosine annealing learning rate scheduler
      - optional AMP on CUDA
      - gradient clipping

    Side effects:
      - Writes metrics to `run_dir/metrics.csv`
      - Saves checkpoints to `run_dir/ckpt/`
      - Optionally renders preview images via `render_fn`

    Args:
        model: PyTorch model to train.
        train_dataset: Dataset providing (X, y, idx) for training.
        val_dataset: Dataset providing (X, y, idx) for validation.
        run_dir: Output directory Path-like with `ckpt/` and `images/` subfolders.
        epochs: Number of epochs.
        batch_size: Batch size.
        lr: Learning rate.
        weight_decay: AdamW weight decay.
        grad_clip: Max gradient norm. Set to None to disable.
        amp: If True, use mixed precision on CUDA.
        render_every: Render/checkpoint frequency in epochs.
        render_fn: Optional callback called on checkpoint epochs:
            render_fn(model, device, epoch) -> None.

    Returns:
        The trained model (same object, returned for convenience).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    criterion = nn.SmoothL1Loss(reduction="none")
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler(device=device.type, enabled=(amp and device.type == "cuda"))

    metrics_path = run_dir / "metrics.csv"
    with open(metrics_path, "w") as f:
        f.write("epoch,train_loss,val_loss\n")

    for epoch in range(1, epochs + 1):
        model.train()
        tot = 0.0
        n = 0

        for Xb, yb, _ in train_loader:
            Xb = Xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type ,enabled=(amp and device.type == "cuda")):
                pred = model(Xb)
                loss = criterion(pred, yb).mean()

            scaler.scale(loss).backward()

            if grad_clip is not None:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(opt)
            scaler.update()

            tot += float(loss.item()) * Xb.size(0)
            n += Xb.size(0)

        train_loss = tot / max(1, n)
        val_loss = eval_loss(model, val_loader, device, criterion)
        scheduler.step()

        with open(metrics_path, "a") as f:
            f.write(f"{epoch},{train_loss:.8f},{val_loss:.8f}\n")

        print(f"Epoch {epoch:03d} | train {train_loss:.6f} | val {val_loss:.6f}")

        if epoch == epochs or (epoch % render_every == 0):
            ckpt_path = run_dir / "ckpt" / f"model_epoch_{epoch:03d}.pt"
            torch.save(model.state_dict(), ckpt_path)
            if render_fn is not None:
                render_fn(model, device, epoch)

    return model
