# Data Documentation

This directory contains the dataset and data-related scripts for the Solar PV Forecasting Benchmark project.

## Dataset Overview

### Source
**NASA POWER (Prediction Of Worldwide Energy Resources)** database
- Website: https://power.larc.nasa.gov/
- Community: Renewable Energy (RE)
- Temporal Resolution: Hourly
- Spatial Resolution: 0.5° × 0.5° grid

### Study Location
**Hengsha Island, Shanghai, China**
- Latitude: 31.3403°N
- Longitude: 121.8389°E
- Elevation: 1.06 m above sea level
- Climate: Subtropical monsoon (Köppen: Cfa)

### Time Period
- **Start**: January 1, 2020, 00:00 UTC
- **End**: December 31, 2024, 23:00 UTC
- **Total hours**: 43,824 (5 years)
- **Daylight hours** (GHI > 20 W/m²): 20,637 hours

## Variables

| Variable Code | Description | Unit | Min | Max | Mean |
|---------------|-------------|------|-----|-----|------|
| `ALLSKY_SFC_SW_DWN` | Global Horizontal Irradiance | W/m² | 0 | 1036.35 | 338.96 |
| `ALLSKY_SFC_SW_DNI` | Direct Normal Irradiance | W/m² | 0 | 982.45 | 156.73 |
| `T2M` | Air Temperature at 2m | °C | -4.28 | 35.70 | 19.20 |
| `RH2M` | Relative Humidity at 2m | % | 36.33 | 100.00 | 76.89 |
| `WS10M` | Wind Speed at 10m | m/s | 0.04 | 25.23 | 5.69 |
| `PS` | Surface Pressure | kPa | 99.82 | 103.56 | 101.52 |

### Derived Variables

| Variable | Description | Formula |
|----------|-------------|---------|
| `solar_zenith` | Solar zenith angle | NREL SPA algorithm |
| `solar_azimuth` | Solar azimuth angle | NREL SPA algorithm |
| `G0h` | Extraterrestrial horizontal irradiance | $G_{sc} \cos(\theta_z)$ |
| `kt` | Clearness index | GHI / G0h |
| `PV_pu` | Normalized PV power | See equation below |

**Normalized PV Power Calculation**:
```
T_c = T_a + GHI * (T_NOCT - 20) / 800
PV_pu = η₀ * (1 - α(T_c - 25)) * (GHI / 1000)

where:
  η₀ = 0.18 (reference efficiency)
  α = 0.005 (temperature coefficient, /°C)
  T_NOCT = 45°C (Nominal Operating Cell Temperature)
```

## Data Files

### Main Dataset
- **File**: `hengsha_hourly_2020_2024.csv`
- **Size**: ~50 MB
- **Format**: CSV with headers
- **Encoding**: UTF-8
- **Missing values**: < 0.1% (handled via linear interpolation)

### Summary Statistics
- **File**: `../figures/feature_summary_stats.csv`
- **Purpose**: Quick reference for descriptive statistics

## Data Quality

### Quality Control Checks

1. **Range validation**: All values within physically plausible bounds
2. **Temporal consistency**: No gaps in hourly sequence
3. **Solar geometry**: Nighttime GHI correctly at zero
4. **Missing data**: < 0.1% of records, handled via interpolation
5. **Outlier detection**: Checked for instrument errors

### Known Issues

- **None** - NASA POWER data is pre-validated and quality-controlled

## Data Access and Download

### Option 1: Direct Download from NASA POWER

```python
import requests
import pandas as pd
from datetime import datetime

def download_nasa_power_data(lat, lon, start_date, end_date):
    """
    Download hourly data from NASA POWER API.
    
    Args:
        lat: Latitude in decimal degrees
        lon: Longitude in decimal degrees
        start_date: Start date as 'YYYYMMDD'
        end_date: End date as 'YYYYMMDD'
    
    Returns:
        pandas DataFrame with hourly data
    """
    base_url = "https://power.larc.nasa.gov/api/temporal/hourly/point"
    
    parameters = [
        "ALLSKY_SFC_SW_DWN",  # GHI
        "ALLSKY_SFC_SW_DNI",  # DNI
        "T2M",                 # Temperature
        "RH2M",                # Humidity
        "WS10M",               # Wind Speed
        "PS"                   # Pressure
    ]
    
    params = {
        "parameters": ",".join(parameters),
        "community": "RE",
        "longitude": lon,
        "latitude": lat,
        "start": start_date,
        "end": end_date,
        "format": "JSON"
    }
    
    response = requests.get(base_url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        # Process data['properties']['parameter']
        # Convert to DataFrame
        return process_nasa_data(data)
    else:
        raise Exception(f"API request failed: {response.status_code}")

# Example usage
df = download_nasa_power_data(
    lat=31.3403,
    lon=121.8389,
    start_date="20200101",
    end_date="20241231"
)
```

### Option 2: Use Existing Dataset

If the CSV file is available:

```python
import pandas as pd

# Load data
df = pd.read_csv('hengsha_hourly_2020_2024.csv', 
                 parse_dates=['datetime'])

# Basic info
print(df.info())
print(df.describe())
```

## Data Preprocessing Pipeline

### Step 1: Load Raw Data
```python
df = pd.read_csv('hengsha_hourly_2020_2024.csv')
```

### Step 2: Filter Daylight Hours
```python
# Keep only hours with meaningful solar radiation
df_day = df[df['GHI'] > 20].copy()
```

### Step 3: Calculate Solar Position
```python
from pvlib import solarposition

solar_position = solarposition.get_solarposition(
    time=df.index,
    latitude=31.3403,
    longitude=121.8389,
    altitude=1.06
)

df['solar_zenith'] = solar_position['zenith']
df['solar_azimuth'] = solar_position['azimuth']
```

### Step 4: Calculate Clearness Index
```python
# Solar constant
G_sc = 1367  # W/m²

# Extraterrestrial horizontal irradiance
df['G0h'] = G_sc * np.cos(np.radians(df['solar_zenith']))

# Clearness index (only for G0h > 10)
df['kt'] = np.where(
    df['G0h'] > 10,
    df['GHI'] / df['G0h'],
    np.nan
)
```

### Step 5: Calculate Normalized PV Power
```python
# Constants
eta_0 = 0.18
alpha = 0.005
T_NOCT = 45
T_ref = 25

# Cell temperature
df['T_cell'] = df['T2M'] + df['GHI'] * (T_NOCT - 20) / 800

# Normalized PV power
df['PV_pu'] = eta_0 * (1 - alpha * (df['T_cell'] - T_ref)) * (df['GHI'] / 1000)
df['PV_pu'] = df['PV_pu'].clip(lower=0, upper=1)
```

### Step 6: Train/Val/Test Split
```python
# Chronological split (no shuffling!)
train = df[(df['datetime'] >= '2020-01-01') & 
           (df['datetime'] <= '2022-12-31')]

val = df[(df['datetime'] >= '2023-01-01') & 
         (df['datetime'] <= '2023-12-31')]

test = df[(df['datetime'] >= '2024-01-01') & 
          (df['datetime'] <= '2024-12-31')]
```

## Data Citation

If you use this dataset, please cite both:

1. **NASA POWER Project**:
   ```
   NASA/POWER CERES/MERRA2 Native Resolution Hourly Data
   Dates: 01/01/2020 through 12/31/2024
   Location: Latitude 31.3403, Longitude 121.8389
   Elevation from MERRA-2: Average for 0.5 x 0.625 degree lat/lon region = 1.06 meters
   The value for missing source data that cannot be computed or is outside of the sources availability range: -999
   Parameter(s): ALLSKY_SFC_SW_DWN T2M RH2M WS10M PS
   ```

2. **This Work**:
   ```
   [Your paper citation]
   ```

## License and Terms of Use

NASA POWER data is freely available for research and commercial use. Please review:
- NASA POWER Data Access Viewer: https://power.larc.nasa.gov/data-access-viewer/
- Terms of use: Public domain, no restrictions

## Contact

For questions about the data:
- **Email**: your.email@institution.edu
- **Issues**: [GitHub Issues](https://github.com/yourusername/Solar_PV_Forecasting_Benchmark/issues)

## Additional Resources

- [NASA POWER Documentation](https://power.larc.nasa.gov/docs/)
- [PVLib Python](https://pvlib-python.readthedocs.io/) - For solar position calculations
- [NREL SPA Calculator](https://midcdmz.nrel.gov/spa/) - Solar position algorithm reference
