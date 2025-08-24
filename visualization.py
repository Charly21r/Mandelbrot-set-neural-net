import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
import io
import imageio.v2 as imageio


def make_grid(xlim=(-2,1), ylim=(-1.5,1.5), res=(1200,1200)):
    xs = np.linspace(*xlim, res[0])
    ys = np.linspace(*ylim, res[1])
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    grid = np.stack([X.ravel(), Y.ravel()], axis=-1).astype(np.float32)
    return X, Y, grid  # X,Y -> (H,W); grid -> (H*W, 2)

@torch.no_grad()
def model_prob_grid(model, device, grid, res):
    model.eval()
    g = torch.from_numpy(grid).float().to(device)
    logits = model(g).squeeze(1)
    probs = torch.sigmoid(logits).reshape(res[1], res[0]).cpu().numpy()  # (H,W)
    return probs


def plot_probability_heatmap(model, device, epoch, xlim=(-2,1), ylim=(-1.5,1.5),
                             res=(1500,1500), outdir="images"):
    os.makedirs(outdir, exist_ok=True)
    X, Y, grid = make_grid(xlim, ylim, res)
    probs = model_prob_grid(model, device, grid, res)

    plt.figure(figsize=(8,8))
    plt.imshow(probs, extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
               origin="lower", aspect="auto", interpolation="lanczos")
    plt.title(f"Probability map @ epoch {epoch+1}")
    plt.xlabel("Real"); plt.ylabel("Imag")
    fname = os.path.join(outdir, f"prob_epoch_{epoch+1}.png")
    plt.savefig(fname, dpi=400, bbox_inches="tight")
    fname = os.path.join(outdir, f"prob_epoch_{epoch+1}.pdf")
    plt.savefig(fname, dpi=400, bbox_inches="tight")
    plt.close()
    print("Saved:", fname)


def plot_learning_curves(train_losses, val_losses, outpath=None):
    plt.figure(figsize=(6,4))
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.title("Learning Curves")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.legend()
    if outpath:
        os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
        plt.savefig(outpath, dpi=160, bbox_inches="tight")
        print("Saved:", outpath)
        plt.close()
    else:
        plt.show()

def plot_roc(y_true, y_prob, outpath=None):
    fpr, tpr, _ = roc_curve(y_true.ravel(), y_prob.ravel())
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(5,5))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0,1],[0,1],"--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curve"); plt.legend()
    if outpath:
        os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
        plt.savefig(outpath, dpi=160, bbox_inches="tight")
        print("Saved:", outpath)
        plt.close()
    else:
        plt.show()

def plot_confusion(y_true, y_pred, outpath=None):
    cm = confusion_matrix(y_true.ravel(), y_pred.ravel())
    plt.figure(figsize=(4.5,4))
    plt.imshow(cm, aspect="equal")
    for (i,j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.title("Confusion Matrix"); plt.xlabel("Pred"); plt.ylabel("True")
    if outpath:
        os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
        plt.savefig(outpath, dpi=160, bbox_inches="tight")
        print("Saved:", outpath)
        plt.close()
    else:
        plt.show()


def compare_grid(gt_mask, prob, threshold=0.5, xlim=(-2,1), ylim=(-1.5,1.5),
                 outpath="images/compare_grid.png"):
    """
    gt_mask: (H,W) boolean ground truth (in set?)
    prob   : (H,W) model probability
    """
    pred = (prob > threshold)
    err = np.abs(gt_mask.astype(np.float32) - prob.astype(np.float32))

    fig, axes = plt.subplots(1,4, figsize=(12,3.2))
    axes[0].imshow(gt_mask, origin="lower", extent=[*xlim,*ylim], aspect="auto")
    axes[0].set_title("Ground truth (mask)")
    axes[1].imshow(prob, origin="lower", extent=[*xlim,*ylim], aspect="auto")
    axes[1].set_title("Model probability")
    axes[2].imshow(pred, origin="lower", extent=[*xlim,*ylim], aspect="auto")
    axes[2].set_title(f"Prediction (>{threshold})")
    im = axes[3].imshow(err, origin="lower", extent=[*xlim,*ylim], aspect="auto")
    axes[3].set_title("Abs error")
    for ax in axes:
        ax.set_xlabel("Real"); ax.set_ylabel("Imag")
    fig.tight_layout()
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    plt.savefig(outpath, dpi=160, bbox_inches="tight")
    plt.close()
    print("Saved:", outpath)


def save_animation(checkpoint_paths, build_model_fn, device,
                   xlim=(-2,1), ylim=(-1.5,1.5), res=(400,400),
                   outpath="images/training.gif"):
    """
    checkpoint_paths: list of paths saved over training
    build_model_fn: lambda -> uninitialized model with same arch
    """

    frames = []
    for i, ckpt in enumerate(checkpoint_paths, 1):
        model = build_model_fn().to(device)
        sd = torch.load(ckpt, map_location=device)
        model.load_state_dict(sd)
        X, Y, grid = make_grid(xlim, ylim, res)
        probs = model_prob_grid(model, device, grid, res)
        fig = plt.figure(figsize=(6,6))
        plt.imshow(probs, extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
                   origin="lower", aspect="auto")
        plt.title(f"Step {i}")
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        frames.append(imageio.imread(buf))
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    imageio.mimsave(outpath, frames, duration=0.6)
    print("Saved:", outpath)