# src/dataset.py
from pathlib import Path
from typing import Dict, List
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
import torch
from torch.utils.data import Dataset

from config import Cfg, S2Cfg, S1Cfg
from stats import StatsCollector
from sentinel2 import Sentinel2Adapter, Product as S2Prod
from sentinel1 import Sentinel1Adapter, Product as S1Prod
from raster import build_target_profile, window_from_center, reproject_to_profile
from temporal import TemporalSelector

class MultiSensorAlertDataset(Dataset):
    """
    PyTorch Dataset for loading multi-sensor (Sentinel-1, Sentinel-2 or others) satellite imagery and alert masks for change detection or alert-based tasks.
    This dataset indexes alert directories containing masks and associated Sentinel-1/Sentinel-2 products, supporting temporal pairing
    It loads image stacks, computes valid data masks, and returns tensors suitable for deep learning workflows.
    """
    def __init__(self, cfg: Cfg):
        self.cfg = cfg
        self.root = Path(cfg.root_dir)
        self.selector = TemporalSelector(cfg.temporal)
        self.stats = StatsCollector(cfg.stats.cache_path, cfg.stats.p_low, cfg.stats.p_high)

        self.use_s2 = S2Cfg(**(cfg.sensors.get("sentinel2", {}))).enabled
        self.use_s1 = S1Cfg(**(cfg.sensors.get("sentinel1", {}))).enabled
        if not self.use_s2 and not self.use_s1:
            raise RuntimeError("No hay sensores habilitados (sentinel1/sentinel2) en config.yaml")

        self.s2_adapter = Sentinel2Adapter(S2Cfg(**(cfg.sensors.get("sentinel2", {}))), self.stats)
        self.s1_adapter = Sentinel1Adapter(S1Cfg(**(cfg.sensors.get("sentinel1", {}))))
        self.records = self._index_alerts()
        self.stats.save()

    def _index_alerts(self):
        recs: List[Dict] = []
        for alert_dir in sorted(self.root.iterdir()):
            if not alert_dir.is_dir(): 
                continue

            mask_files = list(alert_dir.glob("*_mask.tif"))
            if not mask_files: 
                continue
            mask_path = mask_files[0]

            # derive an alert date: median S2 date or S1 date
            s2 = self.s2_adapter.discover(alert_dir)
            s1 = self.s1_adapter.discover(alert_dir)
            if not s2 and not s1: 
                continue
            
            # Alert date
            alert_date = (
                s2[len(s2)//2].date if s2 else
                (s1[len(s1)//2].date if s1 else np.datetime64('1970-01-01'))
            )

            if self.use_s2 and self.use_s1 and s2 and s1:
                if self.cfg.temporal.pairing == "closest_pair":
                    s2_sel, s1_sel = self.selector.best_pair(
                        s2, s1, alert_date, self.cfg.temporal.max_pair_delta_days
                    )
                    if s2_sel is None and s1_sel is None: 
                        continue
                    recs.append({"alert_id": alert_dir.name, "mask": mask_path, "alert_date": alert_date, "s2": s2_sel, "s1": s1_sel})
                elif self.cfg.temporal.expand_pairs:
                    pairs = self.selector.all_pairs(
                        s2, s1, alert_date, self.cfg.temporal.max_pair_delta_days
                    )
                    if not pairs:
                        s2_sel = self.selector.select_single(s2, alert_date)
                        s1_sel = self.selector.select_single(s1, alert_date)
                        if s2_sel is None and s1_sel is None: 
                            continue
                        recs.append({"alert_id": alert_dir.name, "mask": mask_path, "alert_date": alert_date, "s2": s2_sel, "s1": s1_sel})
                    else:
                        for s2p, s1p in pairs:
                            recs.append({"alert_id": alert_dir.name, "mask": mask_path, "alert_date": alert_date, "s2": s2p, "s1": s1p})
                else:
                    s2_sel = self.selector.select_single(s2, alert_date)
                    s1_sel = self.selector.select_single(s1, alert_date)
                    if s2_sel is None and s1_sel is None: continue
                    recs.append({"alert_id": alert_dir.name, "mask": mask_path, "alert_date": alert_date, "s2": s2_sel, "s1": s1_sel})
            
            elif self.use_s2 and s2:
                if getattr(self.cfg.temporal, "expand_pairs", False):
                    # expand all S2 in a temporal window
                    win = self.selector._filter_window(s2, alert_date,
                                                       self.cfg.temporal.pre_days,
                                                       self.cfg.temporal.post_days)
                    win = win or s2  # fallback: todo
                    for s2p in win:
                        recs.append({
                            "alert_id": alert_dir.name, "mask": mask_path, "alert_date": alert_date,
                            "s2": s2p, "s1": None
                        })
                else:
                    s2_sel = self.selector.select_single(s2, alert_date)
                    if s2_sel is None:
                        continue
                    recs.append({
                        "alert_id": alert_dir.name, "mask": mask_path, "alert_date": alert_date,
                        "s2": s2_sel, "s1": None
                    })
            
            elif self.use_s1 and s1:
                if getattr(self.cfg.temporal, "expand_pairs", False):
                    win = self.selector._filter_window(s1, alert_date,
                                                       self.cfg.temporal.pre_days,
                                                       self.cfg.temporal.post_days)
                    win = win or s1
                    for s1p in win:
                        recs.append({
                            "alert_id": alert_dir.name, "mask": mask_path, "alert_date": alert_date,
                            "s2": None, "s1": s1p
                        })
                else:
                    s1_sel = self.selector.select_single(s1, alert_date)
                    if s1_sel is None:
                        continue
                    recs.append({
                        "alert_id": alert_dir.name, "mask": mask_path, "alert_date": alert_date,
                        "s2": None, "s1": s1_sel
                    })
            
        out = []
        for r in recs:
            if (self.use_s2 and r["s2"] is not None) or (self.use_s1 and r["s1"] is not None):
                out.append(r)
        return out

    def _compute_quality(image: np.ndarray, names: List[str], valids: List[np.ndarray]):
        H, W = image.shape[-2], image.shape[-1]
        valid_inter = np.logical_and.reduce(valids) if valids else np.ones((H, W), bool)
        valid_union = np.logical_or.reduce(valids) if valids else np.zeros((H, W), bool)
        finite_mask = np.isfinite(image).all(axis=0) if image.size else np.zeros((H, W), bool)

        # Minimal signal in any channel (avoids black frames)
        tiny = 1e-5
        nonzero_any = np.any(image > tiny, axis=0)

        # Brightness S2 (if RGB exists); dynamic and safe threshold
        def get_ch(n): 
            return image[names.index(n)] if n in names else None
        b04, b03, b02 = map(get_ch, ["S2_B04","S2_B03","S2_B02"])
        if b02 is not None and b03 is not None and b04 is not None:
            brightness = (b02 + b03 + b04) / 3.0
            br_vals = brightness[finite_mask]
            # si todo es ~0, el p2=0; subimos a 0.01 para no marcar "limpio"
            thr = max(0.01, np.percentile(br_vals, 2) if br_vals.size > 100 else 0.01)
            bright_ok = brightness > thr
        else:
            bright_ok = nonzero_any

        valid = valid_inter & finite_mask & bright_ok
        return valid, valid_union, finite_mask

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict:
        r = self.records[idx]

        primary = None
        if self.use_s2 and r["s2"] is not None:
            primary = r["s2"]
        elif self.use_s1 and r["s1"] is not None:
            primary = r["s1"]
        else:
            # Fallback if the record only contains the other sensor
            primary = r["s2"] or r["s1"]

        with rasterio.open(primary.tif_path) as prim:
            if self.cfg.sampling.center_crops:
                win = window_from_center(prim.height, prim.width, self.cfg.sampling.crop_size)
            else:
                size = min(self.cfg.sampling.crop_size, prim.height, prim.width)
                win = Window(0, 0, size, size)
            target_profile = build_target_profile(prim, win)

        stacks, names, valids, dates = [], [], [], {}
        valid_s2 = None
        valid_s1 = None

        # Read S2 only if S2 is enabled
        if self.use_s2 and r["s2"] is not None:
            out = self.s2_adapter.read(r["s2"], target_profile, window=win)
            if out.channels.size:
                stacks.append(out.channels)
                names += out.names
                valids.append(out.valid)
                valid_s2 = out.valid
                dates["S2"] = out.date_iso

        # Read S1 only if S1 is enabled
        if self.use_s1 and r["s1"] is not None:
            out = self.s1_adapter.read(r["s1"], target_profile, window=win)
            if out.channels.size:
                stacks.append(out.channels)
                names += out.names
                valids.append(out.valid)
                valid_s1 = out.valid
                dates["S1"] = out.date_iso
        
        H, W = target_profile["height"], target_profile["width"]
        image = np.concatenate(stacks, axis=0).astype(np.float32) if stacks else np.zeros((0, H, W), np.float32)

        valid_inter = np.logical_and.reduce(valids) if valids else np.ones((H, W), bool)
        valid_union = np.logical_or.reduce(valids) if valids else np.zeros((H, W), bool)
        finite_mask = np.isfinite(image).all(axis=0) if image.size else np.zeros((H, W), bool)
        
        tiny = 1e-5
        nonzero_any = np.any(image > tiny, axis=0)

        # Brightness S2 (if RGB exists); dynamic and safe threshold
        def get_ch(n): 
            return image[names.index(n)] if n in names else None
        
        b04, b03, b02 = map(get_ch, ["S2_B04","S2_B03","S2_B02"])
        
        if b02 is not None and b03 is not None and b04 is not None:
            brightness = (b02 + b03 + b04) / 3.0
            br_vals = brightness[finite_mask]
            # if everything is ~0, p2=0; raise to 0.01 to avoid marking "clean"
            thr = max(0.01, np.percentile(br_vals, 2) if br_vals.size > 100 else 0.01)
            bright_ok = brightness > thr
        else:
            bright_ok = nonzero_any

        valid = valid_inter & finite_mask & bright_ok

        with rasterio.open(r["mask"]) as msrc:
            m = msrc.read(1)
            mask = reproject_to_profile(m, msrc, target_profile, Resampling.nearest).astype(np.int64)
        
        # Quality dict
        quality = {
            "has_s2": r["s2"] is not None,
            "has_s1": r["s1"] is not None,
            "clean_percent": float(valid.mean() * 100.0),
            "clean_union_percent": float(valid_union.mean() * 100.0),
            "finite_percent": float(finite_mask.mean() * 100.0),
            "s2_clean_percent": float(valid_s2.mean() * 100.0) if valid_s2 is not None else None,
            "s1_clean_percent": float(valid_s1.mean() * 100.0) if valid_s1 is not None else None
        }

        return {
            "image": torch.from_numpy(image),
            "mask": torch.from_numpy(mask),
            "valid": torch.from_numpy(valid),
            "valid_union": torch.from_numpy(valid_union),
            "valid_s2": torch.from_numpy(valid_s2) if valid_s2 is not None else None,
            "valid_s1": torch.from_numpy(valid_s1) if valid_s1 is not None else None,
            "channels": names,
            "alert_id": r["alert_id"],
            "dates": dates,
            "quality": quality,
        }

    def subset(self, indices):
        self.records = [self.records[i] for i in indices]
        return self 