from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import json
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window

from config import S2Cfg
from stats import StatsCollector
from raster import (
    get_band_index, parse_datetime, dilate_mask,
    reproject_to_profile, build_target_profile, find_json_pair,
)

@dataclass
class Product:
    tif_path: Path
    bands_json_path: Path
    metadata_path: Path
    provider: str
    date: np.datetime64

@dataclass
class ReadResult:
    channels: np.ndarray
    names: List[str]
    valid: np.ndarray
    date_iso: str

class Sentinel2Adapter:
    def __init__(self, cfg: S2Cfg, stats: StatsCollector):
        self.cfg = cfg; self.stats = stats

    @staticmethod
    def discover(alert_dir: Path) -> List[Product]:
        s2_dir = alert_dir / "sentinel2"
        # print(f"Looking for S2 in {s2_dir}")
        if not s2_dir.exists(): return []
        products: List[Product] = []
        for tif in s2_dir.glob("*.tif"):
            name = tif.stem.lower()
            if any(k in name for k in ["_ndvi","_gndvi","_ndwi","_ndre","_cri"]):
                continue
            bands_json, meta_json = find_json_pair(tif)
            if not bands_json or not meta_json: continue
            try:
                meta = json.loads(meta_json.read_text())
                dt = parse_datetime(meta.get("datetime", "1970-01-01T00:00:00"))
                base = tif.stem
                if "S2-16D" in base: provider = "BDC_S2-16D"
                elif any(x in base for x in ["S2A_MSIL2A","S2B_MSIL2A","S2_L2A"]): provider = "BDC_S2_L2A"
                else: provider = "AWS"
                products.append(Product(tif, bands_json, meta_json, provider, dt))
            except Exception:
                continue
        products.sort(key=lambda p: p.date)
        return products

    def _normalize(self, arr: np.ndarray, provider: str, ch: str) -> np.ndarray:
        key = f"S2_{provider}"; st = self.stats.get(key, ch)
        if st is None:
            if arr.dtype==np.uint16 or arr.max()>1.0: return np.clip(arr.astype(np.float32)/10000.0,0,1)
            return arr.astype(np.float32)
        p1,p99 = st["p1"], st["p99"]
        if p99<=p1: return np.clip(arr.astype(np.float32)/10000.0,0,1)
        out = (arr.astype(np.float32)-p1)/(p99-p1)
        return np.clip(out,0,1)

    def _ensure_stats(self, src, bands: List[str], provider: str):
        key = f"S2_{provider}"
        for b in self.cfg.bands:
            bi = get_band_index(bands, b)
            if bi is None: continue
            if not self.stats.has(key, b):
                self.stats.add(key, b, self.stats.compute_channel_stats(src, bi))

    def _scl_valid(self, src: rasterio.DatasetReader, bands: List[str], provider: str) -> Optional[np.ndarray]:
        scl_idx = get_band_index(bands, "SCL")
        scl_cfg = self.cfg.scl_cloud_mask
        if scl_idx is None or not scl_cfg["use"]:
            return None

        scl = src.read(scl_idx + 1)
        uniq = np.unique(scl)

        if provider == "AWS":
            mask_vals = scl_cfg.get("classes_mask_aws", {})
        elif provider == "BDC_S2_L2A":
            mask_vals = scl_cfg.get("classes_mask_bdc_l2a", {})
        elif provider == "BDC_S2-16D":
            mask_vals = scl_cfg.get("classes_mask_bdc_l6d", {})
        else:
            mask_vals = {}

        # print(f"\tSCL unique values: {uniq}, using mask values: {sorted(mask_vals)}")

        bad = np.isin(scl, np.array(sorted(mask_vals), dtype=scl.dtype))
        if scl_cfg["dilate_px"] > 0:
            bad = dilate_mask(bad, scl_cfg["dilate_px"])
        return ~bad

    def read(self, product: Product, target_profile: dict, window: Optional[Window]) -> ReadResult:
        with rasterio.open(product.tif_path) as src:
            bands = json.loads(product.bands_json_path.read_text()).get("bands", [])
            if self.cfg.provider_norm: self._ensure_stats(src, bands, product.provider)
            if window is None: window = Window(0, 0, src.width, src.height)

            # print(f"\t\tProvider: {product.provider}")

            valid = self._scl_valid(src, bands, product.provider)
            if valid is None: valid = np.ones((src.height, src.width), bool)

            arrays, names = [], []
            for b in self.cfg.bands:
                bi = get_band_index(bands, b)
                if bi is None: continue
                a = src.read(bi+1, window=window)
                a = self._normalize(a, product.provider, b)
                a = reproject_to_profile(a, src, target_profile, Resampling.bilinear)
                arrays.append(a); names.append(f"S2_{b}")

            for idx in self.cfg.indices:
                idx_tif = product.tif_path.parent / f"{product.tif_path.stem}_{idx.lower()}.tif"
                if idx_tif.exists():
                    with rasterio.open(idx_tif) as sidx:
                        arr = np.clip(sidx.read(1).astype(np.float32), -1, 1)
                        arrays.append(reproject_to_profile(arr, sidx, target_profile, Resampling.bilinear))
                        names.append(f"S2_{idx}")
                else:
                    need = {"NDVI": ("B08","B04"), "NDWI": ("B03","B08")}[idx]
                    biA, biB = get_band_index(bands, need[0]), get_band_index(bands, need[1])
                    if biA is None or biB is None: continue
                    A = self._normalize(src.read(biA+1, window=window), product.provider, need[0])
                    B = self._normalize(src.read(biB+1, window=window), product.provider, need[1])
                    A = reproject_to_profile(A, src, target_profile, Resampling.bilinear)
                    B = reproject_to_profile(B, src, target_profile, Resampling.bilinear)
                    val = (A-B)/(A+B+1e-6)
                    arrays.append(np.clip(val,-1,1).astype(np.float32)); names.append(f"S2_{idx}")

            if not arrays:
                H,W = target_profile["height"], target_profile["width"]
                return ReadResult(np.zeros((0,H,W),np.float32), [], np.zeros((H,W),bool), str(product.date))

            ch = np.stack(arrays, axis=0).astype(np.float32)
            valid = reproject_to_profile(valid.astype(np.uint8), src, target_profile, Resampling.nearest).astype(bool)
            return ReadResult(ch, names, valid, str(product.date))

def report_scl_values(product_tif: Path, bands_json: Path, k=20):
    with rasterio.open(product_tif) as src:
        bands = json.loads(bands_json.read_text()).get("bands", [])
        i = get_band_index(bands, "SCL")
        a = src.read(i + 1)
    u, c = np.unique(a, return_counts=True)
    order = np.argsort(-c)
    print("Top SCL values:", list(zip(u[order][:k].tolist(), (c[order][:k]/c.sum()).round(3).tolist())))