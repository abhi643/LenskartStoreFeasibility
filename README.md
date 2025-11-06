# Store Feasibility Predictor

Predict the feasibility of a retail location using AI and geospatial data. This project uses a machine learning pipeline to analyze location features and estimate the suitability of a site for opening a store.

## Features

- **Web Interface**: User-friendly frontend for inputting addresses or coordinates.
- **Flask API**: Backend server for handling prediction requests.
- **Geospatial Analysis**: Uses OpenStreetMap and geopy for location-based features.
- **Machine Learning**: Random Forest regression model trained on real-world data.
- **Data Preprocessing**: Handles missing values, feature engineering, and data quality scoring.

## Project Structure

```
.
├── app.py
├── lenskart_site_pipeline_with_ml.py
├── train_feasibility_model.py
├── expanded_lenskart_site_feasibility_500.csv
├── feasibility_model.joblib
├── requirements.txt
├── runtime.txt
├── index.html
├── sampleAddresses.txt
```

## Getting Started

### Prerequisites

- Python 3.10 (see [`runtime.txt`](runtime.txt))
- pip

### Installation

1. **Clone the repository**
    ```sh
    git clone https://github.com/yourusername/store-feasibility-predictor.git
    cd store-feasibility-predictor
    ```

2. **Install dependencies**
    ```sh
    pip install -r requirements.txt
    ```

3. **Train the model (optional)**
    If you want to retrain the model:
    ```sh
    python train_feasibility_model.py
    ```
    This will generate `feasibility_model.joblib`.

4. **Run the Flask server**
    ```sh
    python app.py
    ```
    The API will be available at `http://127.0.0.1:5000`.

5. **Open the frontend**
    Open [`index.html`](index.html) in your browser.

## Usage

- Enter an address or latitude/longitude, along with property details, in the web interface.
- Click "Predict Feasibility" to get a score and interpretation.

## API

### POST `/predict`

**Request Body (JSON):**
```json
{
  "address": "Connaught Place, New Delhi, Delhi 110001",
  "lat": 28.6315,
  "lon": 77.2167,
  "property_type": "commercial",
  "site_area_sqft": 1200,
  "asking_rent_per_sqft_inr": 45,
  "floor_level": "Ground"
}
```
*Either `address` or both `lat` and `lon` must be provided.*

**Response:**
```json
{
  "feasibility_score": 0.678
}
```

## Data

- Example addresses: [`sampleAddresses.txt`](sampleAddresses.txt)
- Training data: [`expanded_lenskart_site_feasibility_500.csv`](expanded_lenskart_site_feasibility_500.csv)

## License

MIT License

---

**Note:** This project uses OpenStreetMap APIs and may be subject to rate limits. For production use, consider deploying with [gunicorn](https://gunicorn.org/) and a reverse proxy.
