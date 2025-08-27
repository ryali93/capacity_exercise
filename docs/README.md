# Satellite Imagery Downloader - Fixed Geographic Bounds

A comprehensive satellite imagery downloader system that ensures **consistent geographic coverage** across all satellite platforms for machine learning applications.

## 🎯 Key Innovation: Fixed Geographic Bounds

Unlike traditional pixel-based approaches, this system uses **fixed geographic distances** around alert centroids, ensuring all images cover the exact same area regardless of satellite resolution.

### Before vs After

**Before (Pixel-based):**
- Sentinel-2: 512×512 pixels = 5.12km × 5.12km at 10m resolution
- Landsat: 512×512 pixels = 15.36km × 15.36km at 30m resolution
- ❌ Different geographic coverage = inconsistent context

**After (Fixed geographic bounds):**
- Sentinel-2: ~400×400 pixels = 4km × 4km
- Landsat: ~133×133 pixels = 4km × 4km  
- ✅ Same geographic coverage = consistent context

## 🛰️ Supported Satellites

| Satellite | Resolution | Typical Pixels (2km buffer) | Data Source |
|-----------|------------|------------------------------|-------------|
| **Sentinel-2** | 10m | ~400×400 | AWS, Brazil Data Cube |
| **Landsat** | 30m | ~133×133 | AWS, Brazil Data Cube |
| **CBERS-4** | 16m (MUX), 64m (WFI) | ~250×250, ~62×62 | Brazil Data Cube |
| **CBERS-4A** | 2m (WPM), 16m (MUX) | ~2000×2000, ~250×250 | Brazil Data Cube |
| **Sentinel-1** | 10m | ~400×400 | Microsoft Planetary Computer |

## 📁 File Structure

```
download_base_fixed_bounds.py          # Base class with fixed bounds logic
download_sentinel2_fixed_bounds.py     # Sentinel-2 downloader
download_landsat_fixed_bounds.py       # Landsat downloader  
download_cbers4_fixed_bounds.py        # CBERS-4 downloader
download_cbers4a_fixed_bounds.py       # CBERS-4A downloader
download_sentinel1_fixed_bounds.py     # Sentinel-1 downloader
download_satellites_fixed_bounds_example.py  # Usage examples
```

## 🚀 Quick Start

### Basic Usage

```python
from download_sentinel2_fixed_bounds import Sentinel2Downloader

# Create downloader with 2km buffer (4km × 4km total area)
downloader = Sentinel2Downloader(buffer_distance_m=2000)

# Process alerts from GeoJSON file
downloader.process_alerts("mapbiomas_alerts.geojson", limit=10)
```

### Multi-Satellite Processing

```python
from download_sentinel2_fixed_bounds import Sentinel2Downloader
from download_landsat_fixed_bounds import LandsatDownloader
from download_cbers4_fixed_bounds import CBERS4Downloader

# All satellites will cover the same 4km × 4km area
buffer_2km = 2000

s2 = Sentinel2Downloader(buffer_distance_m=buffer_2km)
landsat = LandsatDownloader(buffer_distance_m=buffer_2km)
cbers4 = CBERS4Downloader(buffer_distance_m=buffer_2km)

# Process same alerts with all satellites
for downloader in [s2, landsat, cbers4]:
    downloader.process_alerts("alerts.geojson", limit=5)
```

### Dynamic Buffer Sizes

```python
# Different analysis scales
downloader = Sentinel2Downloader()

# Fine-scale analysis (1km × 1km)
downloader.process_alerts("alerts.geojson", limit=5, buffer_distance_m=500)

# Standard analysis (4km × 4km) 
downloader.process_alerts("alerts.geojson", limit=5, buffer_distance_m=2000)

# Regional analysis (10km × 10km)
downloader.process_alerts("alerts.geojson", limit=5, buffer_distance_m=5000)
```

## 🔧 Configuration Options

### Constructor Parameters

```python
downloader = Sentinel2Downloader(
    output_base_dir="alert_imagery",     # Output directory
    buffer_distance_m=2000,              # Buffer distance in meters
    target_pixels=512                    # Legacy parameter (not used)
)
```

### Buffer Distance Examples

| Buffer (m) | Total Area | Use Case |
|------------|------------|----------|
| 500 | 1km × 1km | Tree-level analysis |
| 1000 | 2km × 2km | Local deforestation |
| 2000 | 4km × 4km | Standard analysis |
| 5000 | 10km × 10km | Regional context |
| 10000 | 20km × 20km | Landscape analysis |

## 📊 Output Structure

```
alert_imagery/
├── [alert_code]/
│   ├── [alert_code]_[date]_mask.tif                    # Alert polygon mask
│   ├── sentinel2/
│   │   ├── [alert]_sentinel2_[id]_[date].tif          # All bands
│   │   ├── [alert]_sentinel2_[id]_[date]_ndvi.tif     # NDVI
│   │   ├── [alert]_sentinel2_[id]_[date]_bands.json   # Band info
│   │   └── [alert]_sentinel2_[id]_[date]_metadata.json
│   ├── landsat/
│   ├── cbers4/
│   ├── cbers4a/
│   └── sentinel1/
```

## 📋 Metadata Information

Each image includes comprehensive metadata:

```json
{
  "alert_code": "12345",
  "satellite": "sentinel2",
  "resolution_m": 10,
  "buffer_distance_m": 2000,
  "geographic_extent_km": 4.0,
  "center_coordinates": {"lon": -60.123, "lat": -10.456},
  "dimensions": {"width": 400, "height": 400},
  "bands": ["blue", "green", "red", "nir"],
  "cloud_cover": 15.2,
  "datetime": "2023-06-15T10:30:00Z"
}
```

## 🎯 Machine Learning Benefits

### 1. Consistent Spatial Context
- All satellites cover the same geographic area
- Same landscape features visible in all images
- Direct comparison between sensors possible

### 2. Multi-Sensor Fusion
```python
# Easy to create multi-sensor datasets
buffer_2km = 2000
sensors = [
    Sentinel2Downloader(buffer_distance_m=buffer_2km),
    LandsatDownloader(buffer_distance_m=buffer_2km),
    Sentinel1Downloader(buffer_distance_m=buffer_2km)
]

for sensor in sensors:
    sensor.process_alerts("training_alerts.geojson")
    
# Result: Perfect pixel-aligned multi-sensor training data
```

### 3. Scale-Aware Analysis
```python
# Train models at different scales
scales = [500, 1000, 2000, 5000]  # meters

for scale in scales:
    downloader = Sentinel2Downloader(buffer_distance_m=scale)
    downloader.process_alerts(f"alerts_scale_{scale}m.geojson")
```

## 🔍 Advanced Features

### NDVI Generation
Automatically creates NDVI (Normalized Difference Vegetation Index) for optical satellites:
- Sentinel-2: Uses red/NIR bands
- Landsat: Uses red/NIR bands  
- CBERS-4/4A: Uses red/NIR bands

### Multiple Data Sources
- **AWS Open Data**: Sentinel-2, Landsat
- **Brazil Data Cube**: Sentinel-2, Landsat, CBERS-4, CBERS-4A
- **Microsoft Planetary Computer**: Sentinel-1

### Quality Filtering
- Cloud cover < 30% for optical satellites
- Automatic invalid geometry fixing
- Duplicate removal by date/sensor

## 🛠️ Installation Requirements

```bash
pip install rasterio geopandas shapely pystac-client numpy
pip install planetary-computer  # For Sentinel-1
```

## 📈 Performance Comparison

| Metric | Fixed Bounds | Pixel-Based |
|--------|--------------|-------------|
| **Geographic Consistency** | ✅ Perfect | ❌ Variable |
| **Multi-sensor Fusion** | ✅ Easy | ❌ Complex |
| **ML Training** | ✅ Optimal | ❌ Suboptimal |
| **Spatial Analysis** | ✅ Comparable | ❌ Incomparable |

## 🎓 Use Cases

### 1. Deforestation Detection
```python
# Download 2km context around each deforestation alert
downloader = Sentinel2Downloader(buffer_distance_m=2000)
downloader.process_alerts("deforestation_alerts.geojson")
```

### 2. Multi-Temporal Analysis
```python
# Same geographic area across different dates
for year in [2020, 2021, 2022, 2023]:
    downloader.process_alerts(f"alerts_{year}.geojson")
```

### 3. Sensor Comparison Studies
```python
# Compare how different sensors see the same area
sensors = [Sentinel2Downloader, LandsatDownloader, CBERS4Downloader]
for SensorClass in sensors:
    sensor = SensorClass(buffer_distance_m=2000)
    sensor.process_alerts("comparison_alerts.geojson")
```

## 📞 Support

For questions or issues:
1. Check the example scripts in `download_satellites_fixed_bounds_example.py`
2. Review metadata files for debugging
3. Verify GeoJSON format for input alerts

## 🔄 Migration from Original System

To migrate from the original pixel-based system:

1. Replace imports:
```python
# Old
from download_sentinel2 import Sentinel2Downloader

# New  
from download_sentinel2_fixed_bounds import Sentinel2Downloader
```

2. Update initialization:
```python
# Old
downloader = Sentinel2Downloader(target_size=512)

# New
downloader = Sentinel2Downloader(buffer_distance_m=2000)
```

3. The new system is backward compatible with existing method calls.

---

**Result: Consistent, comparable, and ML-ready satellite imagery datasets! 🎉**