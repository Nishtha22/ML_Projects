from flask import Flask, request, jsonify
import joblib
import numpy as np
import json
import os

app = Flask(__name__)

# Load model and preprocessors
MODEL_PATH = 'models/heart_disease_model.pkl'
SCALER_PATH = 'models/scaler.pkl'
LABEL_ENCODERS_PATH = 'models/label_encoders.pkl'
TARGET_ENCODER_PATH = 'models/target_encoder.pkl'
FEATURE_NAMES_PATH = 'models/feature_names.json'

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
label_encoders = joblib.load(LABEL_ENCODERS_PATH)
target_encoder = joblib.load(TARGET_ENCODER_PATH)

with open(FEATURE_NAMES_PATH, 'r') as f:
    feature_names = json.load(f)

@app.route('/')
def home():
    return jsonify({
        'message': 'Heart Disease Prediction API',
        'endpoints': {
            '/predict': 'POST - Make a prediction',
            '/health': 'GET - Check API health'
        }
    })

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Prepare input data
        input_data = {}
        
        # Process each feature
        for feature in feature_names:
            if feature not in data:
                return jsonify({'error': f'Missing feature: {feature}'}), 400
            
            value = data[feature]
            
            # Encode categorical features
            if feature in label_encoders:
                try:
                    value = label_encoders[feature].transform([str(value)])[0]
                except ValueError:
                    return jsonify({
                        'error': f'Invalid value for {feature}. Expected one of: {list(label_encoders[feature].classes_)}'
                    }), 400
            
            input_data[feature] = value
        
        # Create feature array in correct order
        feature_array = np.array([[input_data[f] for f in feature_names]])
        
        # Scale features
        feature_scaled = scaler.transform(feature_array)
        
        # Make prediction
        prediction = model.predict(feature_scaled)[0]
        prediction_proba = model.predict_proba(feature_scaled)[0]
        
        # Decode prediction
        predicted_class = target_encoder.inverse_transform([prediction])[0]
        
        # Prepare response
        response = {
            'prediction': predicted_class,
            'probability': {
                target_encoder.classes_[0]: float(prediction_proba[0]),
                target_encoder.classes_[1]: float(prediction_proba[1])
            },
            'confidence': float(max(prediction_proba))
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/features', methods=['GET'])
def get_features():
    """Get expected feature names and their types"""
    feature_info = {}
    for feature in feature_names:
        if feature in label_encoders:
            feature_info[feature] = {
                'type': 'categorical',
                'valid_values': list(label_encoders[feature].classes_)
            }
        else:
            feature_info[feature] = {
                'type': 'numeric'
            }
    
    return jsonify(feature_info)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)