from flask import Flask, request, jsonify
import pandas as pd
import logging
import joblib
import os
import subprocess  # Import the subprocess module
from lenskart_site_pipeline_with_ml import lenskart_site_pipeline_with_feasibility

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

from flask_cors import CORS
CORS(app)

# --- REVISED MODEL LOADING LOGIC ---
model_file = 'feasibility_model.joblib'
model = None

# Check if the model file exists
if os.path.exists(model_file):
    logger.info(f"Found existing model. Loading from {model_file}")
    try:
        model = joblib.load(model_file)
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading existing model file: {e}")
else:
    # If the model file does not exist, run the training script
    logger.warning(f"Model file not found. Training a new model by running train_feasibility_model.py...")
    try:
        # We run the script as a separate process to ensure a clean run
        subprocess.run(["python", "train_feasibility_model.py"], check=True)
        logger.info("Training script finished. Reloading the newly created model.")
        model = joblib.load(model_file)
        logger.info("Newly trained model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to train or load a new model: {e}")

# --- END OF REVISED LOGIC ---

@app.route('/predict', methods=['POST'])
def predict():
    # This part of your code can remain exactly the same
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        address = data.get('address')
        lat = data.get('lat')
        lon = data.get('lon')
        property_type = data.get('property_type', 'commercial')
        site_area_sqft = data.get('site_area_sqft', 1200)
        asking_rent_per_sqft_inr = data.get('asking_rent_per_sqft_inr', 45)
        floor_level = data.get('floor_level', 'Ground')

        if not address and (lat is None or lon is None):
            return jsonify({"error": "Address or lat/lon must be provided"}), 400

        if model is None:
            return jsonify({"error": "Model not loaded. Please check server logs."}), 500

        result = lenskart_site_pipeline_with_feasibility(
            address=address,
            lat=float(lat) if lat is not None else None,
            lon=float(lon) if lon is not None else None,
            property_type=property_type,
            site_area_sqft=site_area_sqft,
            asking_rent_per_sqft_inr=asking_rent_per_sqft_inr,
            floor_level=floor_level,
            model=model
        )

        if 'error' in result:
            return jsonify({"error": result['error']}), 400

        return jsonify({"feasibility_score": result['feasibility_score']}), 200

    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)