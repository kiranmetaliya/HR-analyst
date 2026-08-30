import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
import shap
from flask import Flask, request, jsonify, render_template, send_from_directory

# Configure system paths for clean module imports
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import preprocessing classes to prevent pickle deserialization errors
try:
    from src.preprocess import BookingPreprocessor, SmoothedTargetEncoder
except ImportError:
    from preprocess import BookingPreprocessor, SmoothedTargetEncoder

import __main__
__main__.BookingPreprocessor = BookingPreprocessor
__main__.SmoothedTargetEncoder = SmoothedTargetEncoder

app = Flask(__name__, 
            static_folder='static', 
            template_folder='templates')

# Load models and resources at startup
MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models'))
OUTPUTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../outputs'))

print("Loading preprocessor and LightGBM model...")
preprocessor = joblib.load(os.path.join(MODELS_DIR, 'preprocessor.joblib'))
lgb_model = joblib.load(os.path.join(MODELS_DIR, 'lightgbm.joblib'))

print("Initializing SHAP TreeExplainer...")
explainer = shap.TreeExplainer(lgb_model)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/outputs/<path:filename>')
def serve_outputs(filename):
    """Serves files from the outputs/ directory (e.g., shap_summary.png)."""
    return send_from_directory(OUTPUTS_DIR, filename)

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """Returns the evaluation metrics from train.py."""
    metrics_path = os.path.join(OUTPUTS_DIR, 'evaluation_metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            return jsonify(json.load(f))
    else:
        return jsonify({"error": "Metrics file not found. Run model training first."}), 404

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Accepts booking details, runs preprocessing, calculates LightGBM prediction,
    and returns probability along with local SHAP feature contributions.
    """
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No input data provided."}), 400

        # Fill in form inputs and use defaults for non-form fields
        row = {
            'hotel': data.get('hotel', 'City Hotel'),
            'lead_time': int(data.get('lead_time', 100)),
            'arrival_date_year': int(data.get('arrival_date_year', 2016)),
            'arrival_date_month': data.get('arrival_date_month', 'August'),
            'arrival_date_week_number': int(data.get('arrival_date_week_number', 33)),
            'arrival_date_day_of_month': int(data.get('arrival_date_day_of_month', 15)),
            'stays_in_weekend_nights': int(data.get('stays_in_weekend_nights', 1)),
            'stays_in_week_nights': int(data.get('stays_in_week_nights', 2)),
            'adults': int(data.get('adults', 2)),
            'children': float(data.get('children', 0.0)),
            'babies': int(data.get('babies', 0)),
            'meal': data.get('meal', 'BB'),
            'country': data.get('country', 'PRT'),
            'market_segment': data.get('market_segment', 'Online TA'),
            'distribution_channel': data.get('distribution_channel', 'TA/TO'),
            'is_repeated_guest': int(data.get('is_repeated_guest', 0)),
            'previous_cancellations': int(data.get('previous_cancellations', 0)),
            'previous_bookings_not_canceled': int(data.get('previous_bookings_not_canceled', 0)),
            'reserved_room_type': data.get('reserved_room_type', 'A'),
            'assigned_room_type': data.get('assigned_room_type', 'A'),
            'booking_changes': int(data.get('booking_changes', 0)),
            'deposit_type': data.get('deposit_type', 'No Deposit'),
            'agent': float(data.get('agent', 9.0)),
            'company': float(data.get('company', 0.0)),
            'days_in_waiting_list': int(data.get('days_in_waiting_list', 0)),
            'customer_type': data.get('customer_type', 'Transient'),
            'adr': float(data.get('adr', 100.0)),
            'required_car_parking_spaces': int(data.get('required_car_parking_spaces', 0)),
            'total_of_special_requests': int(data.get('total_of_special_requests', 0))
        }

        # Convert to DataFrame
        df = pd.DataFrame([row])

        # Preprocess features
        df_preprocessed = preprocessor.transform(df)

        # Get prediction probability
        prob = float(lgb_model.predict_proba(df_preprocessed)[0, 1])
        prediction = 1 if prob >= 0.5 else 0

        # Calculate SHAP values
        shap_values_raw = explainer.shap_values(df_preprocessed)

        # Standardize extraction of SHAP values
        if isinstance(shap_values_raw, list):
            shap_vals = shap_values_raw[1][0]
        elif isinstance(shap_values_raw, np.ndarray) and len(shap_values_raw.shape) == 3:
            shap_vals = shap_values_raw[0, :, 1]
        else:
            if hasattr(shap_values_raw, 'values'):
                vals = shap_values_raw.values
                if len(vals.shape) == 3:
                    shap_vals = vals[0, :, 1]
                else:
                    shap_vals = vals[0]
            else:
                shap_vals = shap_values_raw[0]

        # Extract features and their SHAP contributions
        contributions = []
        for name, val, val_raw in zip(preprocessor.feature_names_, df_preprocessed.values[0], shap_vals):
            if abs(val_raw) > 0.005:  # filter out negligible features
                contributions.append({
                    "feature": name,
                    "scaled_value": float(val),
                    "shap_value": float(val_raw)
                })

        # Sort features by absolute SHAP contribution
        contributions = sorted(contributions, key=lambda x: abs(x['shap_value']), reverse=True)

        return jsonify({
            "success": True,
            "probability": prob,
            "prediction": prediction,
            "contributions": contributions[:10]  # Return top 10 features driving prediction
        })

    except Exception as e:
        print(f"Error in prediction: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    # Run server locally on port 5000
    app.run(host='127.0.0.1', port=5000, debug=True)
