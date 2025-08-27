# Landsat Optical Dataset

## Overview
This archive contains **Long-term optical imagery with thermal bands** from the Amazon Deforestation Multi-Sensor Dataset.

### Specifications
- **Resolution**: 30m
- **Provider**: USGS/NASA
- **Coverage**: 100 deforestation alerts across Brazilian Amazon
- **Temporal Range**: 2024-2025

### Available Data
- **Raw Imagery**: Multi-band GeoTIFF files
- **Derived Indices**: NDVI, NDWI, NBR
- **Metadata**: JSON files with acquisition details and band information

### Key Use Cases
- Long-term change analysis (50+ year archive)
- Thermal anomaly detection
- Burn severity assessment
- Historical baseline establishment

## File Structure
```
landsat/
├── {alert_code}_landsat_{scene_id}_{date}.tif        # Raw imagery
├── {alert_code}_landsat_{scene_id}_{date}_metadata.json
├── {alert_code}_landsat_{scene_id}_{date}_band_info.json
└── {alert_code}_landsat_{scene_id}_{date}_{index}.tif  # Derived indices
```

## Loading Data
```python
import rasterio
from pathlib import Path

# Load raw imagery
sensor_file = Path("alert_dir/landsat/alert_123_landsat_scene_20250101.tif")
with rasterio.open(sensor_file) as src:
    data = src.read()  # Shape: (bands, height, width)
    
# Load derived index
ndvi_file = Path("alert_dir/landsat/alert_123_landsat_scene_20250101_ndvi.tif")
with rasterio.open(ndvi_file) as src:
    ndvi = src.read(1)  # Single band
```

## Citation
Please cite this dataset in your research:
```
Amazon Deforestation Multi-Sensor Dataset v1.0 (Landsat Optical)
Generated: 2025-07-29
```

For questions or issues, please refer to the main DATASET_OVERVIEW.md file.
