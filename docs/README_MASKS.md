# Ground Truth Deforestation Masks

## Overview
This archive contains binary deforestation masks derived from MapBiomas Alert Platform for 100 alerts across the Brazilian Amazon.

### Mask Specifications
- **Format**: GeoTIFF (binary masks)
- **Values**: 
  - `0` = Forest/Non-deforested area
  - `1` = Deforested area  
  - `255` = No data/masked area
- **Resolution**: 10m pixel spacing
- **CRS**: EPSG:4326 (WGS84 Geographic)
- **Extent**: 4km × 4km around each alert centroid

### Data Source
- **Source**: MapBiomas Alert Platform (Brazil's operational monitoring)
- **Validation**: Expert-validated alerts with quality scores >0.8
- **Minimum Area**: 0.5 hectares
- **Temporal Precision**: ±1-2 weeks around deforestation event
- **Spatial Precision**: ±1-2 pixels (10-20m boundary accuracy)

### File Structure
```
masks/
├── {alert_code}_mask.tif              # Binary deforestation mask
└── {alert_code}_mask_metadata.json    # Mask creation metadata
```

### Loading Masks
```python
import rasterio
import numpy as np

# Load deforestation mask
with rasterio.open("1390023_mask.tif") as src:
    mask = src.read(1)
    
# Calculate deforested area
deforested_pixels = np.sum(mask == 1)
total_pixels = np.sum((mask == 0) | (mask == 1))  # Exclude nodata
deforestation_percentage = (deforested_pixels / total_pixels) * 100

print(f"Deforested area: {deforestation_percentage:.1f}%")
```

### Usage in ML
```python
# For binary classification
y_true = (mask == 1).astype(int)  # Convert to 0/1

# For masking valid areas only
valid_mask = mask != 255
features = satellite_data[valid_mask]
labels = mask[valid_mask]
```
