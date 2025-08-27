# src/metrics.py
import torch
from typing import Dict, Tuple, List, Optional

def masked_confusion(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, thr: float=0.5):
    """
    pred: (B,1,H,W) logits or probs [0-1]
    target: (B,1,H,W) float {0,1}
    mask: (B,1,H,W) float {0,1}
    """
    if pred.shape[1] == 1: pred = torch.sigmoid(pred)
    pred_bin = (pred >= thr).float()
    m = mask > 0.5
    p = pred_bin[m]
    t = target[m]
    tp = (p*t).sum()
    fp = (p*(1-t)).sum()
    fn = ((1-p)*t).sum()
    return tp, fp, fn

def f1_prec_rec(tp: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor) -> Dict[str, torch.Tensor]:
    prec = tp / (tp + fp + 1e-6)
    rec  = tp / (tp + fn + 1e-6)
    f1   = 2*prec*rec / (prec + rec + 1e-6)
    return {"f1": f1, "prec": prec, "rec": rec}

def iou_deforest(tp: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor) -> torch.Tensor:
    # IoU de la clase positiva (deforestación)
    return tp / (tp + fp + fn + 1e-6)

class PerAlertIoUAggregator:
    """
    Suma intersecciones y uniones por alert_id (enmascaradas) para mIoU por alerta.
    Llámalo por cada batch de valid y al final computa el promedio.
    """
    def __init__(self, thr: float=0.5):
        self.thr = thr
        self.store: Dict[str, Tuple[float,float]] = {}

    @torch.no_grad()
    def update(self, logits, target, mask, alert_ids):
        prob = torch.sigmoid(logits)
        pred = (prob >= self.thr).float()
        B = pred.shape[0]
        for b in range(B):
            a = alert_ids[b]
            m = mask[b,0] > 0.5
            if m.sum() == 0: continue
            p = pred[b,0][m]; t = target[b,0][m]
            inter = (p*t).sum().item()
            union = (p + t - p*t).sum().item() + 1e-6
            i, u = self.store.get(a, (0.0, 0.0))
            self.store[a] = (i + inter, u + union)

    def compute(self) -> Dict[str, float]:
        per_alert = {k: (i/u if u>0 else 0.0) for k,(i,u) in self.store.items()}
        mean_iou = sum(per_alert.values())/max(1,len(per_alert))
        return {"per_alert": per_alert, "mean_per_alert_iou": mean_iou}
