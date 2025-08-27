from typing import List, Dict, Tuple
import random
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from config import Cfg
from dataset import MultiSensorAlertDataset
from pl_utils import rgb_from_sample
import numpy as np

def get_channels_for_experiment(exp: str) -> List[str]:
    S2  = ["S2_B02","S2_B03","S2_B04","S2_B08","S2_B11","S2_B12"]
    S2_IDX = ["S2_NDVI","S2_NDWI"]
    S1  = ["S1_VV", "S1_VH"]
    S1_IDX = ["S1_RVI", "S1_RFDI"] # , "S1_VV_VH_RATIO"
    if exp == "s2": return S2
    if exp == "s2idx": return S2 + S2_IDX
    if exp == "s2idxs1": return S2 + S2_IDX + S1
    if exp == "s1": return S1
    if exp == "s1idx": return S1 + S1_IDX
    if exp == "s2s1": return S2 + S1
    if exp == "s2idxs1idx": return S2 + S2_IDX + S1 + S1_IDX
    raise ValueError(f"Unknown experiment: {exp}")

class ChannelSelector(Dataset):
    """Returns only the desired channels + metadata for visual panel."""
    def __init__(self, base_ds: MultiSensorAlertDataset, wanted: List[str], min_clean: float, cfg: None):
        self.base = base_ds
        self.wanted = wanted
        self.cfg = cfg
        self.idxs = []
        for i in range(len(base_ds)):
            s = base_ds[i]
            if s["quality"]["clean_percent"] >= min_clean*100.0 and all(w in s["channels"] for w in wanted):
                self.idxs.append(i)
        if not self.idxs:
            raise RuntimeError("There are no eligible samples with the required channels.")

    def __len__(self): return len(self.idxs)

    def __getitem__(self, j: int):
        s = self.base[self.idxs[j]]
        x_all, chs = s["image"], s["channels"]
        xs = [ x_all[chs.index(w)] for w in self.wanted ]
        x = torch.stack(xs, 0).float()                 # (C,H,W)

        # remap mask according to config
        mask_np = s["mask"].numpy()
        valid_np = s["valid"].numpy().astype(bool)

        pos_val = int(self.cfg.training.mask_positive_value)
        ignore_vals = set(int(v) for v in self.cfg.training.mask_ignore_values)

        y_np = (mask_np == pos_val).astype(np.float32)
        eval_np = (~np.isin(mask_np, list(ignore_vals))) & valid_np

        y = torch.from_numpy(y_np).unsqueeze(0)          # (1,H,W)
        m = torch.from_numpy(eval_np.astype(np.float32)).unsqueeze(0)

        # preview RGB
        from pl_utils import rgb_from_sample
        rgb = rgb_from_sample(s)

        return {"x": x, "y": y, "m": m, "alert_id": s["alert_id"], "rgb": rgb}

def collate(batch):
    x = torch.stack([b["x"] for b in batch], 0)
    y = torch.stack([b["y"] for b in batch], 0)
    m = torch.stack([b["m"] for b in batch], 0)
    ids = [b["alert_id"] for b in batch]
    rgbs = [b["rgb"] for b in batch]
    return x, y, m, ids, rgbs

class MSDataModule(torch.nn.Module):  # simple container; could be LightningDataModule
    def __init__(self, cfg: Cfg):
        super().__init__()
        self.cfg = cfg
        self.base = MultiSensorAlertDataset(cfg)
        if cfg.training.channels:
            wanted = cfg.training.channels
        else:
            wanted = get_channels_for_experiment(cfg.training.experiment)
            present = set(self.base[0]["channels"])
            wanted = [w for w in wanted if w in present]
        self.wanted = wanted
        self.ds = ChannelSelector(self.base, wanted, cfg.training.min_clean, cfg)

        # split by alert
        rng = random.Random(cfg.seed)
        groups: Dict[str, List[int]] = {}
        for i in range(len(self.ds)):
            aid = self.ds[i]["alert_id"]
            groups.setdefault(aid, []).append(i)
        keys = list(groups.keys())
        rng.shuffle(keys)
        n_val = max(1, int(len(keys)*cfg.training.val_frac))
        self.val_ids = set(keys[:n_val])
        self.tr_idx, self.va_idx = [], []
        for aid, idxs in groups.items():
            (self.va_idx if aid in self.val_ids else self.tr_idx).extend(idxs)

    def train_dataloader(self):
        return DataLoader(Subset(self.ds, self.tr_idx), batch_size=self.cfg.training.batch_size,
                          shuffle=True, num_workers=0, collate_fn=collate)

    def val_dataloader(self):
        return DataLoader(Subset(self.ds, self.va_idx), batch_size=self.cfg.training.batch_size,
                          shuffle=False, num_workers=0, collate_fn=collate)
