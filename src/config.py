"""
Configuration module for the capacity assignment project.
This module defines a set of dataclasses representing configuration options for various components
of the project, including training, logging, sensor data, harmonization, statistics, sampling, 
data packing, and integration with Weights & Biases (wandb).
"""
from dataclasses import dataclass, field
from importlib.resources import path
from typing import Dict, List, Optional, Tuple
from rasterio.enums import Resampling
import yaml

@dataclass
class WandbCfg:
    enabled: bool = False
    project: str = "ms-defor"
    name: str = "experiment"
    entity: Optional[str] = None
    mode: str = "online"     # online|offline|disabled
    tags: List[str] = field(default_factory=list)

@dataclass
class ReduceOnPlateauCfg:
    factor: float = 0.5
    patience: int = 3

@dataclass
class TrainCfg:
    experiment: str = "s2"
    channels: Optional[List[str]] = None
    min_clean: float = 0.6
    batch_size: int = 4
    epochs: int = 25
    val_frac: float = 0.2
    encoder: str = "resnet34"
    encoder_weights: Optional[str] = "imagenet"
    lr: float = 1e-3
    threshold: float = 0.5
    early_stopping_patience: int = 6
    reduce_on_plateau: ReduceOnPlateauCfg = ReduceOnPlateauCfg()
    mask_positive_value: int = 255
    mask_ignore_values: List[int] = field(default_factory=list)

@dataclass
class LogCfg:
    out_dir: str = "runs"
    save_every: int = 1
    save_best: bool = True
    csv_name: str = "metrics.csv"
    jsonl_name: str = "events.jsonl"

@dataclass
class S2MaskCfg:
    use: bool = True
    encoding: str = "auto"
    classes_to_mask: Tuple[int, ...] = (3, 8, 9, 10, 11)
    values_to_mask: Optional[Tuple[int, ...]] = None
    dilate_px: int = 1

@dataclass
class S1Cfg:
    enabled: bool = True
    to_db: bool = True
    bands: Tuple[str, ...] = ("VV", "VH")
    indices: Tuple[str, ...] = () # ("RVI","RFDI")
    add_ratio: bool = True
    add_rfdi: bool = False
    add_rvi: bool = False

@dataclass
class S2Cfg:
    enabled: bool = True
    bands: Tuple[str, ...] = ("B02", "B03", "B04", "B08", "B11", "B12")
    indices: Tuple[str, ...] = ("NDVI",)
    provider_norm: bool = True
    scl_cloud_mask: S2MaskCfg = S2MaskCfg()

@dataclass
class TemporalCfg:
    mode: str = "single" # single|pair|delta (we use single)
    pre_days: int = 30
    post_days: int = 14
    selector: str = "closest_post"
    pairing: str = "closest_pair" # closest_pair|to_alert_then_pair|none
    max_pair_delta_days: int = 8
    expand_pairs: bool = False # if True → one dataset record per S2+S1 pair

@dataclass
class HarmonizeCfg:
    reference: str = "sentinel2"
    resampling_cont: Resampling = Resampling.bilinear
    resampling_cat: Resampling = Resampling.nearest

@dataclass
class StatsCfg:
    compute_if_missing: bool = True
    sample_per_type: int = 50
    p_low: float = 1.0
    p_high: float = 99.0
    cache_path: str = "stats/ms_stats.json"

@dataclass
class SamplingCfg:
    crop_size: int = 448
    min_valid_fraction: float = 0.6
    center_crops: bool = True

@dataclass
class PackCfg:
    fusion: str = "early"
    channel_order: Optional[List[str]] = None

@dataclass
class Cfg:
    run_name: str = "experiment"
    seed: int = 42
    root_dir: str = "dataset"
    sensors: Dict[str, dict] = None
    temporal: TemporalCfg = TemporalCfg()
    harmonize: HarmonizeCfg = HarmonizeCfg()
    stats: StatsCfg = StatsCfg()
    sampling: SamplingCfg = SamplingCfg()
    pack: PackCfg = PackCfg()
    training: TrainCfg = TrainCfg()
    logging: LogCfg = LogCfg()
    wandb: WandbCfg = WandbCfg()
    
    @staticmethod
    def load(path: str) -> "Cfg":
        raw = yaml.safe_load(open(path, "r"))
        s2 = raw.get("sensors", {}).get("sentinel2", {})
        s1 = raw.get("sensors", {}).get("sentinel1", {})
        return Cfg(
            run_name=raw.get("run_name", "experiment"),
            seed=raw.get("seed", 42),
            root_dir=raw.get("root_dir", "dataset"),
            sensors={"sentinel2": s2, "sentinel1": s1},
            temporal=TemporalCfg(**raw.get("temporal", {})),
            harmonize=HarmonizeCfg(**raw.get("harmonize", {})),
            stats=StatsCfg(**raw.get("stats", {})),
            sampling=SamplingCfg(**raw.get("sampling", {})),
            pack=PackCfg(**raw.get("pack", {})),
            training=TrainCfg(**raw.get("training", {})),
            logging=LogCfg(**raw.get("logging", {})),
            wandb=WandbCfg(**raw.get("wandb", {})),
        )
