import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix

def make_grid(xlim=(-2,1), ylim=(-1.5,1.5), res=(400,400)):
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
                             res=(400,400), outdir="images"):
    os.makedirs(outdir, exist_ok=True)
    X, Y, grid = make_grid(xlim, ylim, res)
    probs = model_prob_grid(model, device, grid, res)

    plt.figure(figsize=(6,6))
    plt.imshow(probs, extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
               origin="lower", aspect="auto")
    # 0.5 contour (decision boundary) so it looks better
    cs = plt.contour(X, Y, probs, levels=[0.5], linewidths=1.0)
    plt.clabel(cs, inline=True, fontsize=8, fmt={0.5:"0.5"})
    plt.title(f"Probability map @ epoch {epoch+1}")
    plt.xlabel("Real"); plt.ylabel("Imag")
    fname = os.path.join(outdir, f"prob_epoch_{epoch+1}.png")
    plt.savefig(fname, dpi=160, bbox_inches="tight")
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