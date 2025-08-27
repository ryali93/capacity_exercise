# Sentinel-1 SAR Dataset

## Overview
This archive contains **C-band SAR imagery with dual polarization (VV/VH)** from the Amazon Deforestation Multi-Sensor Dataset.

### Specifications
- **Resolution**: 10m
- **Provider**: ESA/Copernicus
- **Coverage**: 100 deforestation alerts across Brazilian Amazon
- **Temporal Range**: 2024-2025

### Available Data
- **Raw Imagery**: Multi-band GeoTIFF files
- **Derived Indices**: VV_VH_RATIO, RVI, NRPB, RFDI
- **Metadata**: JSON files with acquisition details and band information

### Key Use Cases
- All-weather monitoring (cloud-independent)
- Surface roughness analysis
- Forest structure assessment
- Change detection in any weather

## File Structure
```
sentinel1/
├── {alert_code}_sentinel1_{scene_id}_{date}.tif        # Raw imagery
├── {alert_code}_sentinel1_{scene_id}_{date}_metadata.json
├── {alert_code}_sentinel1_{scene_id}_{date}_band_info.json
└── {alert_code}_sentinel1_{scene_id}_{date}_{index}.tif  # Derived indices
```

## Loading Data
```python
import rasterio
from pathlib import Path

# Load raw imagery
sensor_file = Path("alert_dir/sentinel1/alert_123_sentinel1_scene_20250101.tif")
with rasterio.open(sensor_file) as src:
    data = src.read()  # Shape: (bands, height, width)
    
# Load derived index
ndvi_file = Path("alert_dir/sentinel1/alert_123_sentinel1_scene_20250101_ndvi.tif")
with rasterio.open(ndvi_file) as src:
    ndvi = src.read(1)  # Single band
```

## Citation
Please cite this dataset in your research:
```
Amazon Deforestation Multi-Sensor Dataset v1.0 (Sentinel-1 SAR)
Generated: 2025-07-29
```

For questions or issues, please refer to the main DATASET_OVERVIEW.md file.
