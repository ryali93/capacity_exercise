import torch, numpy as np, matplotlib.pyplot as plt
from config import Cfg
from pl_data import MSDataModule
from pl_module import LitSeg
from pl_utils import render_val_panel
from typing import Tuple
import os

def load_db(cfg_path: str = "../config.yaml") -> Tuple[Cfg, MSDataModule]:
    cfg = Cfg.load(cfg_path)
    dm = MSDataModule(cfg)
    return cfg, dm

def load_model_from_ckpt(cfg_path: str, ckpt_path: str) -> LitSeg:
    cfg = Cfg.load(cfg_path)
    model = LitSeg.load_from_checkpoint(ckpt_path, cfg=cfg, map_location="cpu")
    model.eval()
    return model

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import List
from pl_utils import render_val_panel

def reorder_x_to_model_channels(x: torch.Tensor, ds_channels: List[str], model_channels: List[str]) -> torch.Tensor:
    """
    x: (C,H,W) del dataset actual
    ds_channels: canales en x
    model_channels: canales que espera el modelo (en el orden guardado en ckpt)
    """
    miss = [c for c in model_channels if c not in ds_channels]
    if miss:
        raise ValueError(f"Faltan canales requeridos por el modelo: {miss}\n"
                         f"ds_channels={ds_channels}\nmodel_channels={model_channels}")
    idx = [ds_channels.index(c) for c in model_channels]
    return x[idx]

@torch.no_grad()
def infer_on_subset_index(subset_ds, model: LitSeg, j: int, thr: float = None, show=True):
    """
    subset_ds: el dataset que cuelga de un DataLoader (p.ej. dm.val_dataloader().dataset)
               Puede ser un torch.utils.data.Subset envolviendo a ChannelSelector.
    j: índice dentro del subset
    """
    # desempaqueta si es Subset
    base = subset_ds.dataset if hasattr(subset_ds, "dataset") else subset_ds

    # toma el sample ya preparado por ChannelSelector (x,y,m,alert_id,rgb)
    s = base[j]
    x, y, m = s["x"], s["y"], s["m"]
    rgb, aid = s["rgb"], s["alert_id"]
    ds_channels = model.hparams.get("channels", None)  # si guardaste channels en hparams
    # Ojo: aquí ds_channels es lo que esperaba el modelo, no lo que trae el sample.
    # Si tu ChannelSelector devuelve también la lista "wanted" en el sample, úsala:
    sample_channels = getattr(base, "wanted", None) or getattr(subset_ds, "wanted", None)
    if sample_channels is None:
        # si no la expone, asume que coincide con model.hparams.channels
        sample_channels = model.hparams.channels

    # reordena x si hace falta
    x = reorder_x_to_model_channels(x, sample_channels, model.hparams.channels)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    prob = torch.sigmoid(model(x.unsqueeze(0).to(device))).cpu().numpy()[0, 0]
    thr = model.thr if thr is None else thr
    pred = (prob >= thr).astype(np.uint8)

    if show:
        gt = y[0].numpy()  # (H,W)
        fig = render_val_panel(rgb, gt, pred, title=f"Infer — {aid}")
        plt.show()

    return prob, pred, s

def find_indices_by_alert(subset_ds, alert_id: str, max_items: int = None) -> List[int]:
    base = subset_ds.dataset if hasattr(subset_ds, "dataset") else subset_ds
    hits = []
    for j in range(len(base)):
        s = base[j]
        if s["alert_id"] == alert_id:
            hits.append(j)
            if max_items and len(hits) >= max_items:
                break
    return hits

def indices_by_alert_sorted(dm: MSDataModule, split: str, alert_id: str, sensor_preference="S2", max_items=None):
    """
    split: "val" o "train"
    sensor_preference: "S2" | "S1" | "any"
    Devuelve índices dentro del subset correspondiente, ordenados por la fecha del sensor elegido.
    """
    dl = dm.val_dataloader() if split=="val" else dm.train_dataloader()
    subset = dl.dataset
    base = subset.dataset if hasattr(subset, "dataset") else subset  # ChannelSelector
    # ChannelSelector suele tener .base -> MultiSensorAlertDataset y .idxs mapeando índices
    ms_base = getattr(base, "base", None)
    idmap = getattr(base, "idxs", None)

    if ms_base is None or idmap is None:
        # fallback: lineal
        return find_indices_by_alert(subset, alert_id, max_items)

    # recopila (subset_idx, date)
    pairs = []
    for sub_idx, ms_idx in enumerate(idmap):
        r = ms_base.records[ms_idx]
        if r["alert_id"] != alert_id:
            continue
        # elige fecha
        dt = None
        if sensor_preference == "S2" and r.get("s2") is not None:
            dt = r["s2"].date
        elif sensor_preference == "S1" and r.get("s1") is not None:
            dt = r["s1"].date
        else:
            dt = (r["s2"].date if r.get("s2") else (r["s1"].date if r.get("s1") else None))
        if dt is not None:
            pairs.append((sub_idx, dt))
    pairs.sort(key=lambda t: t[1])
    out = [p[0] for p in pairs]
    return out[:max_items] if max_items else out


cfg, dm   = load_db("../config.yaml")
model     = load_model_from_ckpt("../config.yaml", "../src/runs/s2idxs1idx_wce_v2/best.ckpt")

val_subset = dm.val_dataloader().dataset

alert = "1389517"
idxs = indices_by_alert_sorted(dm, split="val", alert_id=alert, sensor_preference="S2", max_items=4)
print("idxs:", idxs)

prob, pred, s = infer_on_subset_index(val_subset, model, 78, thr=None, show=True)

print(prob)
print(pred)
print(s)