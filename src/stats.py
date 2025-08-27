from __future__ import annotations
from pathlib import Path
from typing import Dict
import json
import numpy as np
from rasterio.windows import Window

class StatsCollector:
    def __init__(self, cache_path: str, p_low=1.0, p_high=99.0):
        self.cache_path = Path(cache_path)
        self.p_low, self.p_high = p_low, p_high
        self.stats: Dict[str, Dict[str, Dict[str, float]]] = {}
        if self.cache_path.exists():
            try:
                self.stats = json.loads(self.cache_path.read_text())
            except Exception:
                self.stats = {}

    def has(self, key: str, ch: str) -> bool:
        return key in self.stats and ch in self.stats[key]

    def get(self, key: str, ch: str):
        return self.stats.get(key, {}).get(ch)

    def add(self, key: str, ch: str, vals: Dict[str, float]):
        self.stats.setdefault(key, {})[ch] = vals

    def save(self):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.stats))

    def compute_channel_stats(self, src, band_idx: int, sample_windows=5):
        rng = np.random.default_rng(0)
        vals = []
        for _ in range(sample_windows):
            hh, ww = src.height, src.width
            size = min(256, hh, ww)
            if size <= 0: continue
            y = int(rng.integers(0, max(1, hh - size)))
            x = int(rng.integers(0, max(1, ww - size)))
            w = Window(x, y, size, size)
            a = src.read(band_idx + 1, window=w)
            a = a[np.isfinite(a) & (a > 0)]
            if a.size > 100: vals.append(a)
        if not vals: return {"p1":0.0, "p99":1.0, "mean":0.5, "std":0.2}
        v = np.concatenate(vals)
        p1, p99 = np.percentile(v, [self.p_low, self.p_high])
        v = v[(v>=p1)&(v<=p99)]
        return {"p1":float(p1), "p99":float(p99), "mean":float(v.mean()), "std":float(v.std()+1e-6)}
