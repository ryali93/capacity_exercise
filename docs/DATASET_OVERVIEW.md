# Amazon Deforestation Multi-Sensor Dataset

*Generated on 2025-07-29 21:30:13*

## Dataset Overview

This dataset contains **100 deforestation alerts** from the Brazilian Amazon with multi-sensor satellite imagery and ground truth masks. The dataset is designed for multi-modal machine learning research in environmental monitoring and change detection.

### Key Statistics
- **Total Alerts**: 100
- **Multi-sensor Coverage**: 100 alerts (100.0%)
- **Ground Truth Masks**: 100 alerts (100.0%)
- **Total Satellite Images**: 1943
- **Derived Indices**: 5802
- **Total Dataset Size**: 4.4 GB

## Sensor Coverage

The dataset includes imagery from **4 satellite sensors**:

### Sentinel-1
- **Type**: SAR 
- **Provider**: ESA/Copernicus
- **Coverage**: 100 alerts (100.0%)
- **Total Images**: 162
- **Resolution**: 10m
- **File Size**: 0.3 GB
- **Spectral Bands**: VV, VH
- **Indices Available**: NRPB (162), RFDI (162), RVI (162), VV_VH_RATIO (162)

### Sentinel-2
- **Type**: Optical 
- **Provider**: ESA/Copernicus
- **Coverage**: 100 alerts (100.0%)
- **Total Images**: 1021
- **Resolution**: 10m
- **File Size**: 3.4 GB
- **Spectral Bands**: B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12
- **Indices Available**: GNDVI (1021), NDVI (1021), NDWI (1021)

### Landsat
- **Type**: Optical 
- **Provider**: USGS/NASA
- **Coverage**: 100 alerts (100.0%)
- **Total Images**: 680
- **Resolution**: 30m
- **File Size**: 0.3 GB
- **Spectral Bands**: B2, B3, B4, B5, B6, B7
- **Indices Available**: NBR (680), NDVI (680), NDWI (680)

### CBERS-4A
- **Type**: Optical 
- **Provider**: INPE/CRESDA
- **Coverage**: 100 alerts (100.0%)
- **Total Images**: 80
- **Resolution**: 2m (WPM), 16m (MUX), 64m (WFI)
- **File Size**: 0.5 GB
- **Sensor Breakdown**:
  - **MUX** (16m): 102 images - Multispectral
  - **WPM** (2m): 58 images - RGB Composite + Multispectral
- **High-Resolution RGB Composites (TCI)**: 29 files at 2m resolution
- **Spectral Bands**: WPM-TCI, BAND5, BAND6, BAND7, BAND8, BAND13, BAND14, BAND15, BAND16
- **Indices Available**: NDVI (51)

## Temporal Coverage

- **Date Range**: 2024-09-29 to 2025-07-11
- **Time Span**: 285 days
- **Total Acquisitions**: 999

### Temporal Distribution by Sensor
- **Sentinel-2**: 239 acquisitions (2024-10-31 to 2025-06-10)
- **Landsat**: 680 acquisitions (2024-09-29 to 2025-07-11)
- **CBERS-4A**: 80 acquisitions (2024-11-05 to 2025-06-18)

## Multi-Sensor Combinations

The following sensor combinations are available:

- **CBERS-4A + Landsat + Sentinel-1 + Sentinel-2**: 100 alerts (100.0%)

## Sentinel-1 Detailed Coverage

Sentinel-1 provides SAR imagery from ESA/Copernicus:

- **Alert Coverage**: 100 alerts
- **Data Volume**: 0.3 GB
- **Total Images**: 162
- **Resolution**: 10m
- **Polarizations**: VV, VH (dual-pol)
- **Product Type**: Ground Range Detected (GRD)
- **Orbit**: Descending pass optimization for Amazon
- **Available Bands**: VV, VH
- **Derived Indices**: NRPB (162), RFDI (162), RVI (162), VV_VH_RATIO (162)


## Sentinel-2 Detailed Coverage

Sentinel-2 provides optical imagery from ESA/Copernicus:

- **Alert Coverage**: 100 alerts
- **Data Volume**: 3.4 GB
- **Total Images**: 1021
- **Resolution**: 10m
- **Spectral Bands**: 13 bands (443nm - 2190nm)
- **Key Capabilities**: Red-edge bands for vegetation analysis
- **Revisit Time**: 5 days (twin satellite constellation)
- **Available Bands**: B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12
- **Derived Indices**: GNDVI (1021), NDVI (1021), NDWI (1021)


## Landsat Detailed Coverage

Landsat provides optical imagery from USGS/NASA:

- **Alert Coverage**: 100 alerts
- **Data Volume**: 0.3 GB
- **Total Images**: 680
- **Resolution**: 30m
- **Spectral Bands**: 11 bands including thermal infrared
- **Heritage**: 50+ years of consistent observations
- **Key Strength**: Long-term change detection
- **Available Bands**: B2, B3, B4, B5, B6, B7
- **Derived Indices**: NBR (680), NDVI (680), NDWI (680)


## CBERS-4A Detailed Coverage

CBERS-4A provides optical imagery from INPE/CRESDA:

- **Alert Coverage**: 100 alerts
- **Data Volume**: 0.5 GB
- **Total Images**: 80
- **Resolution**: 2m (WPM), 16m (MUX), 64m (WFI)
- **High-Resolution RGB**: 29 TCI composites at 2m resolution

### Sensor Distribution
- **MUX**: 102 images at 16m (Multispectral)
- **WPM**: 58 images at 2m (RGB Composite + Multispectral)
- **Available Bands**: WPM-TCI, BAND5, BAND6, BAND7, BAND8, BAND13, BAND14, BAND15, BAND16
- **Derived Indices**: NDVI (51)


## Dataset Structure

```
alert_imagery/
├── {alert_code}/                    # One directory per alert
│   ├── {alert_code}_mask.tif       # Ground truth deforestation mask
│   ├── {alert_code}_mask_metadata.json
│   ├── sentinel1/                    # SAR imagery (if available)
│   │   ├── {alert_code}_sentinel1_{scene_id}_{date}.tif
│   │   ├── {alert_code}_sentinel1_{scene_id}_{date}_band_info.json
│   │   ├── {alert_code}_sentinel1_{scene_id}_{date}_metadata.json
│   │   └── {alert_code}_sentinel1_{scene_id}_{date}_{index}.tif  # vv_vh_ratio, rvi, etc.
│   ├── sentinel2/                    # Optical imagery (if available)
│   │   ├── {alert_code}_sentinel2_{scene_id}_{date}.tif
│   │   ├── {alert_code}_sentinel2_{scene_id}_{date}_band_info.json
│   │   ├── {alert_code}_sentinel2_{scene_id}_{date}_metadata.json
│   │   └── {alert_code}_sentinel2_{scene_id}_{date}_{index}.tif  # ndvi, ndwi, ndre, etc.
│   ├── landsat/                      # Landsat imagery (if available)
│   │   ├── {alert_code}_landsat_{scene_id}_{date}.tif
│   │   ├── {alert_code}_landsat_{scene_id}_{date}_band_info.json
│   │   ├── {alert_code}_landsat_{scene_id}_{date}_metadata.json
│   │   └── {alert_code}_landsat_{scene_id}_{date}_{index}.tif    # ndvi, nbr, evi, etc.
│   └── cbers4a/                      # CBERS-4A imagery (if available)
│       ├── {alert_code}_cbers4a_WPM_{scene_id}_{date}.tif    # 2m RGB composite
│       ├── {alert_code}_cbers4a_MUX_{scene_id}_{date}.tif    # 16m multispectral
│       ├── {alert_code}_cbers4a_{sensor}_{scene_id}_{date}_metadata.json
│       └── {alert_code}_cbers4a_{sensor}_{scene_id}_{date}_ndvi.tif
└── mapbiomas_alerts.geojson          # Alert metadata and geometries
```

## Available Vegetation/SAR Indices

The dataset includes pre-computed indices optimized for deforestation detection:

### Optical Indices (Sentinel-2, Landsat, CBERS)
- **GNDVI**: Green NDVI - chlorophyll content and plant stress
- **NBR**: Normalized Burn Ratio - fire/burn detection and forest disturbance
- **NDVI**: Normalized Difference Vegetation Index - vegetation health and biomass
- **NDWI**: Normalized Difference Water Index - water/moisture detection

### SAR Indices (Sentinel-1)
- **NRPB**: Normalized Radar Backscatter - surface scattering characteristics
- **RFDI**: Radar Forest Degradation Index - forest change and degradation detection
- **RVI**: Radar Vegetation Index - vegetation structure and biomass
- **VV_VH_RATIO**: VV/VH Polarization Ratio - surface roughness and structure

## Technical Specifications

### Coordinate System
- **CRS**: EPSG:4326 (WGS84 Geographic)
- **Spatial Extent**: 4km × 4km around each alert centroid
- **Ground Truth**: Binary masks (0=forest, 1=deforestation, 255=nodata)

### File Formats
- **Imagery**: GeoTIFF with optimized compression (DEFLATE/ZSTD)
- **Metadata**: JSON format with comprehensive band and acquisition information
- **Ground Truth**: GeoTIFF masks co-registered with imagery
- **Alert Geometries**: GeoJSON with MapBiomas alert polygons

### Data Quality
- **Average File Size**: 0.4 MB per file
- **Space Optimization**: Indices stored as uint16 with optimal scaling
- **Compression**: DEFLATE compression with predictor=2 for optimal performance
- **Resolution Range**: 2m (CBERS-4A WPM) to 64m (CBERS-4A WFI)

## Sentinel-1 Technical Details

Sentinel-1 SAR specifications:

### SAR Configuration
- **Frequency**: C-band (5.405 GHz)
- **Polarizations**: VV + VH (dual-pol)
- **Product**: Ground Range Detected (GRD)
- **Pixel Spacing**: 10m x 10m
- **Swath Width**: 250 km (IW mode)
- **Orbit**: Descending (optimized for Amazon coverage)

### Processing Level
- **Radiometric Calibration**: Sigma naught (σ°) backscatter
- **Geometric Correction**: Terrain corrected
- **Speckle Filtering**: Multi-temporal Lee filter applied

### Data Volume & Coverage
- **Alert Coverage**: 100 alerts
- **Total Images**: 162  
- **Data Volume**: 0.3 GB
- **Average File Size**: 0.2 MB
- **Derived Indices**: 4 types (648 files)


## Sentinel-2 Technical Details

Sentinel-2 optical specifications:

### Optical Configuration  
- **Instrument**: MultiSpectral Instrument (MSI)
- **Spectral Bands**: 13 bands (443-2190 nm)
- **Spatial Resolution**: 10m (4 bands), 20m (6 bands), 60m (3 bands)
- **Swath Width**: 290 km
- **Revisit Time**: 5 days (twin satellite constellation)

### Key Spectral Bands
- **Visible**: Blue (490nm), Green (560nm), Red (665nm) @ 10m
- **Near-Infrared**: NIR (842nm) @ 10m  
- **Red Edge**: 705nm, 740nm, 783nm @ 20m (unique for vegetation analysis)
- **SWIR**: 1610nm, 2190nm @ 20m

### Data Volume & Coverage
- **Alert Coverage**: 100 alerts
- **Total Images**: 1021  
- **Data Volume**: 3.4 GB
- **Average File Size**: 0.6 MB
- **Derived Indices**: 3 types (3063 files)


## Landsat Technical Details

Landsat optical specifications:

### Optical Configuration
- **Mission**: Landsat 8/9 Operational Land Imager (OLI)
- **Spectral Bands**: 11 bands including thermal infrared
- **Spatial Resolution**: 30m (multispectral), 15m (panchromatic)
- **Swath Width**: 185 km
- **Revisit Time**: 16 days (8-day with Landsat 8+9)

### Heritage & Strengths
- **Timeline**: 50+ years of consistent global observations
- **Calibration**: Radiometrically stable for long-term analysis
- **Thermal**: Thermal infrared bands for fire/heat detection
- **Archive**: Extensive historical data for change detection

### Data Volume & Coverage
- **Alert Coverage**: 100 alerts
- **Total Images**: 680  
- **Data Volume**: 0.3 GB
- **Average File Size**: 0.1 MB
- **Derived Indices**: 3 types (2040 files)


## CBERS-4A Technical Details

CBERS-4A optical specifications:

### Multi-Sensor Configuration
CBERS-4A carries three complementary sensors:

#### WPM Sensor (2m resolution)
- **Product**: True Color Image (TCI) RGB composite  
- **Processing**: Principal Component Analysis (PCA) fusion
- **Bands**: 3-band GeoTIFF (Red, Green, Blue)
- **Values**: 1-255 (8-bit), 0=NoData
- **Use Case**: High-resolution visual analysis, detailed boundary mapping

#### MUX Sensor (16m resolution)
- **Bands**: BAND5 (Blue), BAND6 (Green), BAND7 (Red), BAND8 (NIR)
- **Spectral Range**: 450-890 nm  
- **Quantization**: 10-bit → 16-bit output
- **Use Case**: Spectral analysis, vegetation indices

#### WFI Sensor (64m resolution)  
- **Bands**: BAND13 (Blue), BAND14 (Green), BAND15 (Red), BAND16 (NIR)
- **Swath**: 866 km (wide coverage)
- **Use Case**: Regional monitoring, large-scale analysis

### Data Volume & Coverage
- **Alert Coverage**: 100 alerts
- **Total Images**: 80  
- **Data Volume**: 0.5 GB
- **Average File Size**: 2.5 MB
- **Derived Indices**: 1 types (51 files)


## Ground Truth Methodology

Ground truth masks are derived from **MapBiomas Alert Platform**, Brazil's operational deforestation monitoring system:

- **Source**: MapBiomas validated deforestation polygons
- **Validation**: Expert-validated alerts with quality scores >0.8
- **Minimum Area**: 0.5 hectares (detectable at 10m resolution)
- **Temporal Precision**: ±1-2 weeks around actual deforestation timing
- **Spatial Precision**: ±1-2 pixels (10-20m) boundary accuracy

### Data Loading Example
```python
# Load multi-sensor data including CBERS-4A
alert_dir = Path("alert_imagery/1390023")

# Load ground truth
mask = rasterio.open(alert_dir / "1390023_mask.tif").read(1)

# Load CBERS-4A high-resolution RGB (if available)
cbers4a_dir = alert_dir / "cbers4a"
if cbers4a_dir.exists():
    # Find WPM TCI files (2m RGB composite)
    wpm_files = list(cbers4a_dir.glob("*_WPM_*.tif"))
    if wpm_files:
        rgb_2m = rasterio.open(wpm_files[0]).read()  # 3-band RGB
    
    # Find MUX multispectral files (16m)
    mux_files = list(cbers4a_dir.glob("*_MUX_*.tif"))
    if mux_files:
        multispectral_16m = rasterio.open(mux_files[0]).read()  # 4-band multispectral

# Load other sensors
sar_dir = alert_dir / "sentinel1"
optical_dir = alert_dir / "sentinel2"
# ... (load as before)
```

---

*This dataset was created for research in multi-sensor deforestation detection. The data combines imagery from multiple satellite missions including Brazil's CBERS-4A constellation with validated ground truth from Brazil's operational monitoring systems.*

**Dataset Version**: 1.0  
**Last Updated**: 2025-07-29  
**Total Size**: 4.4 GB