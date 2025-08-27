# CBERS-4A Multi-Resolution Dataset

## Overview
This archive contains **Brazilian satellite with 2m RGB composites and multispectral** from the Amazon Deforestation Multi-Sensor Dataset.

### Specifications
- **Resolution**: 2m (WPM TCI), 16m (MUX)
- **Provider**: INPE/CRESDA
- **Coverage**: 100 deforestation alerts across Brazilian Amazon
- **Temporal Range**: 2024-2025

### Available Data
- **Raw Imagery**: Multi-band GeoTIFF files
- **Derived Indices**: NDVI
- **Metadata**: JSON files with acquisition details and band information

### Key Use Cases
- High-resolution boundary mapping (2m RGB)
- Multi-scale analysis (2m to 64m)
- Detailed deforestation validation
- Brazilian satellite constellation

## File Structure
```
cbers4a/
├── {alert_code}_cbers4a_{scene_id}_{date}.tif        # Raw imagery
├── {alert_code}_cbers4a_{scene_id}_{date}_metadata.json
├── {alert_code}_cbers4a_{scene_id}_{date}_band_info.json
└── {alert_code}_cbers4a_{scene_id}_{date}_{index}.tif  # Derived indices
```

## Loading Data
```python
import rasterio
from pathlib import Path

# Load raw imagery
sensor_file = Path("alert_dir/cbers4a/alert_123_cbers4a_scene_20250101.tif")
with rasterio.open(sensor_file) as src:
    data = src.read()  # Shape: (bands, height, width)
    
# Load derived index
ndvi_file = Path("alert_dir/cbers4a/alert_123_cbers4a_scene_20250101_ndvi.tif")
with rasterio.open(ndvi_file) as src:
    ndvi = src.read(1)  # Single band
```

## Citation
Please cite this dataset in your research:
```
Amazon Deforestation Multi-Sensor Dataset v1.0 (CBERS-4A Multi-Resolution)
Generated: 2025-07-29
```

For questions or issues, please refer to the main DATASET_OVERVIEW.md file.
