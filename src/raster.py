from __future__ import annotations
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
from rasterio.warp import reproject

def parse_datetime(dt: str) -> np.datetime64:
    try:
        return np.datetime64(dt)
    except Exception:
        return np.datetime64("1970-01-01T00:00:00")

def get_band_index(bands_info: List[str], name: str) -> Optional[int]:
    try:
        return bands_info.index(name)
    except ValueError:
        try:
            return bands_info.index(name.upper())
        except ValueError:
            try:
                return bands_info.index(name.lower())
            except ValueError:
                return None

def find_json_pair(path: Path) -> Tuple[Optional[Path], Optional[Path]]:
    # print(f"\tFinding JSON pairs for: {path.stem}")
    cand_bands = [path.parent / f"{path.stem}_bands.json"]
    # print(f"Looking for band files in: {cand_bands}")
    cand_meta  = [path.parent / f"{path.stem}_metadata.json"]
    # print(f"Looking for metadata files in: {cand_meta}")
    bands = next((p for p in cand_bands if p.exists()), None)
    meta = next((p for p in cand_meta if p.exists()), None)
    return bands, meta

def dilate_mask(mask: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return mask
    from scipy.ndimage import maximum_filter
    return maximum_filter(mask.astype(np.uint8), size=(2*k+1)).astype(bool)

def build_target_profile(src: rasterio.DatasetReader, win: Optional[Window]) -> dict:
    prof = src.profile.copy()
    if win is None:
        return prof
    transform = rasterio.windows.transform(win, src.transform)
    prof.update({"width": int(win.width), "height": int(win.height), "transform": transform})
    return prof

def reproject_to_profile(arr: np.ndarray, src: rasterio.DatasetReader, target_profile: dict,
                         resampling: Resampling) -> np.ndarray:
    dst = np.zeros((target_profile["height"], target_profile["width"]), dtype=arr.dtype)
    reproject(
        source=arr,
        destination=dst,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=target_profile["transform"],
        dst_crs=target_profile["crs"],
        resampling=resampling,
        num_threads=1,
    )
    return dst

def window_from_center(h: int, w: int, crop: int) -> Window:
    y = max(0, (h - crop) // 2); x = max(0, (w - crop) // 2)
    return Window(x, y, min(crop, w - x), min(crop, h - y))