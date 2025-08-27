from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List
import json
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window

from config import S1Cfg
from raster import (
    get_band_index, parse_datetime, reproject_to_profile, find_json_pair,
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

class Sentinel1Adapter:
    def __init__(self, cfg: S1Cfg):
        self.cfg = cfg

    @staticmethod
    def discover(alert_dir: Path) -> List[Product]:
        s1_dir = alert_dir / "sentinel1"
        # print(f"Looking for S1 in {s1_dir}")
        if not s1_dir.exists(): return []
        products: List[Product] = []
        for tif in s1_dir.glob("*.tif"):
            name = tif.stem.lower()
            if any(k in name for k in ["_vv_vh_ratio","_rvi","_rfdi","_nrpb"]):
                continue
            bands_json, meta_json = find_json_pair(tif)
            if not bands_json or not meta_json: continue
            try:
                meta = json.loads(meta_json.read_text())
                dt = parse_datetime(meta.get("datetime", "1970-01-01T00:00:00"))
                products.append(Product(tif, bands_json, meta_json, provider="S1_GRD", date=dt))
            except Exception:
                continue
        products.sort(key=lambda p: p.date)
        return products

    @staticmethod
    def _to_db(a: np.ndarray) -> np.ndarray:
        a = a.astype(np.float32)
        a[a<=0] = 1e-6
        return 10.0*np.log10(a)

    def read(self, product: Product, target_profile: dict, window: Window|None) -> ReadResult:
        with rasterio.open(product.tif_path) as src:
            bands = json.loads(product.bands_json_path.read_text()).get("bands", [])
            if window is None: window = Window(0,0,src.width,src.height)

            arrays, names = [], []
            vv_i, vh_i = get_band_index(bands, "VV"), get_band_index(bands, "VH")
            vv = src.read(vv_i+1, window=window) if vv_i is not None else None
            vh = src.read(vh_i+1, window=window) if vh_i is not None else None

            if vv is not None:
                vv = self._to_db(vv) if self.cfg.to_db else vv.astype(np.float32)
                vv = reproject_to_profile(vv, src, target_profile, Resampling.bilinear)
                arrays.append(vv); names.append("S1_VV")
            if vh is not None:
                vh = self._to_db(vh) if self.cfg.to_db else vh.astype(np.float32)
                vh = reproject_to_profile(vh, src, target_profile, Resampling.bilinear)
                arrays.append(vh); names.append("S1_VH")

            # Ratio (dB space difference)
            if self.cfg.add_ratio and "S1_VV" in names and "S1_VH" in names:
                ratio = arrays[names.index("S1_VV")] - arrays[names.index("S1_VH")]
                arrays.append(ratio.astype(np.float32)); names.append("S1_VV_VH_RATIO")

            # RFDI and RVI (linear domain)
            if (self.cfg.add_rfdi or self.cfg.add_rvi) and "S1_VV" in names and "S1_VH" in names:
                vv_r, vh_r = arrays[names.index("S1_VV")], arrays[names.index("S1_VH")]
                lin = lambda dB: 10 ** (dB/10.0) if self.cfg.to_db else dB
                vv_lin, vh_lin = lin(vv_r), lin(vh_r)
                if self.cfg.add_rfdi:
                    rfdi = (vv_lin - vh_lin) / (vv_lin + vh_lin + 1e-6)
                    arrays.append(rfdi.astype(np.float32)); names.append("S1_RFDI")
                if self.cfg.add_rvi:
                    rvi = 4*vh_lin / (vv_lin + vh_lin + 1e-6)
                    arrays.append(rvi.astype(np.float32)); names.append("S1_RVI")

            if not arrays:
                H,W = target_profile["height"], target_profile["width"]
                return ReadResult(np.zeros((0,H,W),np.float32), [], np.zeros((H,W),bool), str(product.date))

            ch = np.stack(arrays, axis=0).astype(np.float32)
            valid = np.isfinite(ch[0])
            return ReadResult(ch, names, valid, str(product.date))