# NYC Taxi Fare Prediction

Machine learning system for predicting NYC Yellow Taxi fares using historical trip data.

## Overview

This project builds an end-to-end ML pipeline including data processing, exploratory analysis, model training, and deployment via FastAPI.

## Project Structure

```
nyc-taxi-prediction/
├── data/
│   ├── raw/                    # Original data files
│   ├── processed/              # Parquet files (partitioned by year/month)
│   └── metadata/               # Lookup tables
├── notebooks/                  # Jupyter notebooks for EDA
├── scripts/                    # Data processing and training scripts
├── models/                     # Saved model artifacts
├── app/                        # FastAPI application
├── dashboard/                  # Streamlit dashboard
└── docs/                       # Documentation
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/nyc-taxi-prediction.git
cd nyc-taxi-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Data Pipeline

```bash
# 1. Download data (2023)
python scripts/download_data.py --year 2023

# 2. Preprocess into Parquet partitions
python scripts/preprocess.py

# 3. Explore data
jupyter notebook notebooks/01_data_exploration.ipynb
```

### Model Training

```bash
# Train XGBoost model
python scripts/train_xgboost.py

# Train TensorFlow model
python scripts/train_tensorflow.py
```

### Run API

```bash
# Start FastAPI server
uvicorn app.main:app --reload

# API docs: http://localhost:8000/docs
```

### Launch Dashboard

```bash
# Start Streamlit dashboard
streamlit run dashboard/app.py

# Dashboard: http://localhost:8501
```

## Dataset

**Source**: [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)

- ~38 million trips (2023 Yellow Taxi data)
- Features: pickup/dropoff times, locations, distances, fares
- Size: ~3.8GB (CSV) → ~800MB (Parquet)

## Tech Stack

- **Data**: DuckDB, Pandas, Parquet
- **ML**: XGBoost, TensorFlow, Scikit-learn
- **Visualization**: Matplotlib, Seaborn, Plotly, Folium
- **API**: FastAPI, Pydantic, Uvicorn
- **Dashboard**: Streamlit
- **Cloud**: AWS S3 / Google Cloud Storage

## API Usage

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "pickup_location_id": 237,
    "dropoff_location_id": 161,
    "trip_distance": 5.2,
    "passenger_count": 2,
    "pickup_hour": 14,
    "day_of_week": 3
  }'
```

**Response:**
```json
{
  "predicted_fare": 23.50,
  "model_used": "xgboost"
}
```

## Documentation

- [Claude Instructions](docs/CLAUDE.md) - Instructions for Claude Code
- [Tech Stack](docs/TECH_STACK.md) - Technology details
- [Data Quality](docs/DATA_QUALITY.md) - Validation rules

## License

MIT License

## Acknowledgments

Data provided by NYC Taxi & Limousine Commission