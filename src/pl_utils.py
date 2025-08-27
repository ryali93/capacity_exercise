import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional

def _p_stretch(a, p=(2,98), valid=None, gamma=1.2):
    x = a.astype(np.float32)
    v = x[valid] if valid is not None and valid.any() else x[np.isfinite(x)]
    if v.size == 0: return np.clip(x, 0, 1)
    lo, hi = np.percentile(v, p)
    if hi <= lo: lo, hi = float(v.min()), float(v.max()+1e-6)
    x = np.clip((x - lo)/(hi - lo), 0, 1)
    if gamma != 1.0: x = np.clip(x,0,1)**(1.0/gamma)
    return x

def rgb_from_sample(sample: Dict, prange=(2,98)):
    ch, img, valid = sample["channels"], sample["image"].numpy(), sample["valid"].numpy()
    need = ["S2_B04","S2_B03","S2_B02"]
    if not all(n in ch for n in need): return None
    r = _p_stretch(img[ch.index("S2_B04")], prange, valid, gamma=1.2)
    g = _p_stretch(img[ch.index("S2_B03")], prange, valid, gamma=1.2)
    b = _p_stretch(img[ch.index("S2_B02")], prange, valid, gamma=1.2)
    return np.stack([r,g,b], -1)

def render_val_panel(rgb: Optional[np.ndarray], gt: np.ndarray, pred: np.ndarray, title: str):
    fig, ax = plt.subplots(1, 3, figsize=(12,4))
    if rgb is None:
        ax[0].imshow(np.zeros((*gt.shape,3)))
        ax[0].set_title("RGB (no S2)")
    else:
        ax[0].imshow(rgb); ax[0].set_title("RGB")
    ax[1].imshow(gt, vmin=0, vmax=1, cmap="RdBu"); ax[1].set_title("GT")
    ax[2].imshow(pred, vmin=0, vmax=1, cmap="RdBu"); ax[2].set_title("Pred")
    for a in ax: a.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    return fig
