# NYC Taxi Fare Prediction

Machine learning project for analyzing and predicting NYC Yellow Taxi fares using 2023 trip data.

![NYC Traffic Congestion](./image.png)

## Project Overview

This project performs comprehensive analysis of 34 million NYC Yellow Taxi trips from 2023, with the goal of building a fare prediction model. The pipeline includes data cleaning, exploratory analysis, feature engineering, and preparation for machine learning.

## Dataset

**Source**: [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)

- **Period**: 2023 (full year)
- **Records**: 34,009,543 trips (after outlier removal)
- **Raw Size**: 3.7GB CSV
- **Processed Size**: 469MB Parquet
- **Target Variable**: `fare_amount` (metered base fare)

### Key Statistics
- **Mean fare**: $16.78
- **Median fare**: $12.80
- **Fare-distance correlation**: 0.954
- **Manhattan pickups**: 90% of all trips

## Project Structure

```
NYC-Taxi/
├── data/
│   ├── raw/                          # Original CSV (3.7GB)
│   │   └── Yellow_Taxi_Trip_Data.csv
│   ├── processed/                    # Cleaned Parquet (469MB)
│   │   └── cleaned_data.parquet
│   └── lookups/                      # Reference tables (taxi zones)
├── scripts/
│   ├── data_processing_pipeline.py   # ETL pipeline (DuckDB + Polars)
│   ├── EDA.ipynb                     # Exploratory data analysis
│   ├── query_db.py                   # Database query utility
│   └── test.sql                      # SQL test queries
├── taxi_data.db                      # DuckDB database (420MB)
├── data_quality_report.csv           # Validation results
├── zones_statistics.csv              # Zone-level statistics
├── image.png                         # NYC congestion visualization
└── main.py                           # Main entry point
```

## Quick Start

### Prerequisites

```bash
python 3.10+
```

### Installation

```bash
# Clone repository
git clone <repository-url>
cd NYC-Taxi

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install polars pandas duckdb seaborn matplotlib jupyter
```

### Running the Pipeline

```bash
# 1. Process raw data (ETL pipeline)
python scripts/data_processing_pipeline.py

# 2. Explore the data
jupyter notebook scripts/EDA.ipynb
```

## Data Processing Pipeline

The ETL pipeline (`data_processing_pipeline.py`) performs:

1. **Data Loading**: Reads 3.7GB CSV into DuckDB
2. **Feature Engineering**: Creates temporal features (hour, weekday, month)
3. **Outlier Removal**: Removes extreme values (0.1% and 99.9% quantiles)
4. **Export**: Saves cleaned data as Parquet with Snappy compression
5. **Quality Reporting**: Generates data validation reports

**Tech Stack**: DuckDB (SQL engine) + Polars (data manipulation)

## Key Findings

### Temporal Patterns
- **Rush hours** (7-9 AM, 5-7 PM): 31% of trips, but lower average fares ($16.28 vs $17.23)
- **Early morning premium**: 4-6 AM has highest fares (~$17.50), likely airport trips
- **Thursday peak**: Highest trip volume (5.38M trips)
- **Midday duration peak**: 2-5 PM shows longest trip durations (~16 minutes)

### Spatial Patterns
- **Manhattan dominance**: 90% of pickups (30.6M trips)
- **Airport effect**: Queens ($43.34 avg) and EWR ($37.91 avg) have 2-3x higher fares
- **Manhattan paradox**: Highest volume, lowest average fare ($14.81)

### Fare Insights
- **Primary driver**: Trip distance (r=0.954 correlation)
- **NYC meter structure**: $2.50 per mile base rate
- **Secondary factors**: Time of day, location, traffic conditions

## Feature Engineering

### Temporal Features
- `travel_hour`: Hour of day (0-23)
- `travel_weekday`: Day of week (0=Sunday)
- `travel_month`: Month (1-12)
- `is_rush_hour`: Boolean for 7-9 AM, 5-7 PM
- `is_weekend`: Boolean for Saturday/Sunday

### Trip Features
- `trip_duration`: Duration in minutes
- `avg_speed`: Miles per minute
- `fare_per_mile`: $/mile rate
- `fare_per_minute`: $/minute rate

### Route Features
- `route_mean_distance`: Average distance for pickup-dropoff pair
- `distance_ratio`: Actual distance / route average

## Tech Stack

- **Data Processing**: DuckDB, Polars
- **Analysis**: Pandas (for visualization)
- **Visualization**: Seaborn, Matplotlib
- **Storage**: Parquet (Snappy compression)
- **Notebook**: Jupyter

## Files Description

- **data_processing_pipeline.py**: Complete ETL pipeline
- **EDA.ipynb**: Comprehensive exploratory analysis with visualizations
- **taxi_data.db**: DuckDB database with indexed tables
- **data_quality_report.csv**: Validation metrics
- **zones_statistics.csv**: Aggregated zone-level statistics

## Data Quality

### Outlier Removal Strategy
- **Method**: Quantile-based filtering (0.1% - 99.9%)
- **Before**: Max fare $386,983, max distance 17,456 miles
- **After**: Max fare $96.10, max distance 21.7 miles
- **Records removed**: 150,624 trips (0.4%)

### Validation
- Missing values handled in preprocessing
- Temporal features validated (no future dates)
- Location IDs cross-referenced with taxi zone lookup

## Future Work

### Machine Learning Models
- [ ] Baseline: Linear Regression
- [ ] Tree-based: XGBoost, LightGBM
- [ ] Deep Learning: Neural networks with embeddings

### Additional Features
- [ ] Cyclical encoding (sin/cos for temporal features)
- [ ] Airport indicators
- [ ] Borough-to-borough features
- [ ] Weather data integration

### Deployment
- [ ] FastAPI endpoint for predictions
- [ ] Streamlit dashboard
- [ ] Docker containerization

## Performance Notes

- **Memory-efficient**: Uses lazy evaluation with Polars
- **Fast queries**: DuckDB in-memory engine
- **Compressed storage**: Parquet reduces size by 87%
- **Scalable**: Pipeline can handle years of data

## References

- [NYC TLC Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
- [Data Dictionary](data_dictionary_trip_records_yellow.pdf)
- [Taxi Zone Lookup](data/lookups/taxi_zone_lookup.csv)

## License

MIT

## Author

Sam Alavi
