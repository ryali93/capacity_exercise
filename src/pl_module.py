import numpy as np
import torch
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from typing import Dict, List
from metrics import masked_confusion, f1_prec_rec, iou_deforest, PerAlertIoUAggregator
from pl_utils import render_val_panel

class LitSeg(pl.LightningModule):
    def __init__(self, in_ch: int, cfg, channels: List[str]):
        super().__init__()
        self.save_hyperparameters(ignore=["cfg"])
        self.cfg = cfg
        self.model = smp.Unet(encoder_name=cfg.training.encoder,
                              encoder_weights=cfg.training.encoder_weights,
                              in_channels=in_ch, classes=1)
        self.thr = cfg.training.threshold
        # accumulators (per epoch)
        self._reset_acc()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def _wbce_dice(self, logit, y, m):
        # pos_weight = auto
        pos = (y*m).sum()
        tot = m.sum()
        p = pos/(tot+1e-6)
        # N_neg/N_pos ~ (1-p)/p
        w = ((1-p)/(p+1e-6)).clamp(max=25.0)
        pos_weight = w.detach()
        
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logit, y, reduction="none", pos_weight=pos_weight
        )
        bce = (bce*m).sum()/(m.sum()+1e-6)

        prob = torch.sigmoid(logit)
        inter = ((prob*m)*(y*m)).sum()
        denom = ((prob*m) + (y*m)).sum() + 1e-6
        dice = 1 - (2*inter/denom)
        return 0.5*bce + 0.5*dice

    def _reset_acc(self):
        self.train_tp = self.train_fp = self.train_fn = 0.0
        self.val_tp   = self.val_fp   = self.val_fn   = 0.0
        self.val_alert_agg = PerAlertIoUAggregator(thr=self.thr)
        self._val_vis = []
        self._train_valid_px = 0.0
        self._train_pos_px   = 0.0
        self._val_valid_px   = 0.0
        self._val_pos_px     = 0.0

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.cfg.training.lr)
        return {"optimizer": opt}

    def _loss(self, logit, y, m):
        return self._wbce_dice(logit, y, m)

    # -------- train ----------
    def training_step(self, batch, batch_idx):
        x,y,m,ids,rgbs = batch
        if m.sum() == 0:
            return None
        logit = self.model(x)
        loss = self._loss(logit, y, m)

        tp,fp,fn = masked_confusion(logit, y, m, thr=self.thr)
        self.train_tp += float(tp.item()); self.train_fp += float(fp.item()); self.train_fn += float(fn.item())
        self._train_valid_px += float(m.sum().item())
        self._train_pos_px   += float((y*m).sum().item())

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def on_train_epoch_start(self): self._reset_acc()

    def on_train_epoch_end(self):
        if self._train_valid_px > 0:
            pos_rate = self._train_pos_px / self._train_valid_px
            # self.log("train/pos_rate", pos_rate)
        tp = torch.tensor(self.train_tp); fp = torch.tensor(self.train_fp); fn = torch.tensor(self.train_fn)
        pr = f1_prec_rec(tp,fp,fn); iou = iou_deforest(tp,fp,fn)
        self.log("train/iou_def", iou, prog_bar=True)
        self.log("train/f1", pr["f1"]); self.log("train/prec", pr["prec"]); self.log("train/rec", pr["rec"])

    # -------- val ----------
    def validation_step(self, batch, batch_idx):
        x,y,m,ids,rgbs = batch
        if m.sum() == 0:
            return None
        logit = self.model(x)
        loss = self._loss(logit, y, m)

        tp,fp,fn = masked_confusion(logit, y, m, thr=self.thr)
        self.val_tp += float(tp.item()); self.val_fp += float(fp.item()); self.val_fn += float(fn.item())
        self.val_alert_agg.update(logit, y, m, ids)
        self._val_valid_px += float(m.sum().item())
        self._val_pos_px   += float((y*m).sum().item())

        # save visualizations as panels
        if len(self._val_vis) < 6:
            prob = torch.sigmoid(logit).detach().cpu().numpy()
            pred = (prob >= self.thr).astype(np.uint8)
            for b in range(min(x.size(0), 6 - len(self._val_vis))):
                rgb = rgbs[b]  # HWC or None
                gt  = y[b,0].detach().cpu().numpy()
                pd  = pred[b,0]
                fig = render_val_panel(rgb, gt, pd, title=f"val panel — {ids[b]}")
                self._val_vis.append(fig)
                
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def on_validation_epoch_end(self):
        if self._val_valid_px > 0:
            pos_rate = self._val_pos_px / self._val_valid_px
            self.log("val/pos_rate", pos_rate, prog_bar=True)
        tp = torch.tensor(self.val_tp); fp = torch.tensor(self.val_fp); fn = torch.tensor(self.val_fn)
        pr = f1_prec_rec(tp,fp,fn); iou = iou_deforest(tp,fp,fn)
        agg = self.val_alert_agg.compute()
        self.log("val/iou_def", iou, prog_bar=True)
        self.log("val/f1", pr["f1"]); self.log("val/prec", pr["prec"]); self.log("val/rec", pr["rec"])
        self.log("val/mean_per_alert_iou", agg["mean_per_alert_iou"], prog_bar=True)

        # W&B images
        if isinstance(self.logger, pl.loggers.WandbLogger) and self._val_vis:
            self.logger.log_image(key="media/panels", images=[f for f in self._val_vis],
                                  caption=[f"epoch {self.current_epoch}"]*len(self._val_vis))
        self._val_vis.clear()
