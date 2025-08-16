import numpy as np
import torch
import matplotlib.pyplot as plt

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
