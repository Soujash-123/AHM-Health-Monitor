from flask import Flask, request, jsonify
import pandas as pd
import joblib
import time

app = Flask(__name__)

# Load the models
model_names = [
    'temperature_model', 
    'vibration_model', 
    'magnetic_flux_model',
    'audible_sound_model',  # New model
    'ultra_sound_model'     # New model
]
models = {name: joblib.load(f"{name}.pkl") for name in model_names}

# Define the feature sets used by each model
feature_sets = {
    'temperature_model': ['temperature_one', 'temperature_two'],
    'vibration_model': ['vibration_x', 'vibration_y', 'vibration_z'],
    'magnetic_flux_model': ['magnetic_flux_x', 'magnetic_flux_y', 'magnetic_flux_z'],
    'audible_sound_model': ['vibration_x', 'vibration_y', 'vibration_z', 'audible_sound'],  # New features
    'ultra_sound_model': ['vibration_x', 'vibration_y', 'vibration_z', 'ultra_sound']       # New features
}

def predict_from_models(input_data):
    # Create a DataFrame from the input data
    df_input = pd.DataFrame([input_data])
    
    # Dictionary to store predictions
    predictions = {}
    
    # Predict using each model
    for model_name in model_names:
        features = feature_sets[model_name]
        model = models[model_name]
        
        # Select relevant features from the input data
        X_input = df_input[features]
        
        # Predict and store the result in the dictionary
        prediction = model.predict(X_input)[0]
        predictions[model_name.replace('_model', '')] = prediction
    
    return predictions

def evaluate_state(predictions):
    # Determine state based on predictions
    all_healthy = all(p == 'healthy' for p in predictions.values())
    all_unhealthy = all(p == 'unhealthy' for p in predictions.values())
    
    if all_healthy:
        overall_health = "Healthy state"
    elif all_unhealthy:
        overall_health = "Unhealthy state"
    else:
        unhealthy_components = [key for key, value in predictions.items() if value == 'unhealthy']
        overall_health = f"Warning. Unhealthy components detected: {unhealthy_components}"
    
    return overall_health

def calculate_expected_timestamp(input_data):
    temperature = sum(input_data[key] for key in feature_sets['temperature_model'])
    magnetic_flux = sum(input_data[key] for key in feature_sets['magnetic_flux_model'])
    vibration = sum(input_data[key] for key in feature_sets['vibration_model'])
    
    # Assuming that 'audible_sound' and 'ultra_sound' fields also affect timestamp calculation
    audible_sound = input_data.get('audible_sound', 0)
    ultra_sound = input_data.get('ultra_sound', 0)
    
    expected_seconds = 3 * temperature + 5 * magnetic_flux + 2 * vibration + audible_sound + ultra_sound
    
    return expected_seconds

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.json
    
    # Get predictions from models
    predictions = predict_from_models(input_data)
    
    # Evaluate overall health state
    overall_health = evaluate_state(predictions)
    
    # Calculate expected timestamp
    expected_timestamp = calculate_expected_timestamp(input_data) if "Warning" in overall_health else 0
    
    # Prepare the response
    response = {
        "Component Temperature": predictions['temperature'],
        "Component Vibration": predictions['vibration'],
        "Component Magnetic Flux": predictions['magnetic_flux'],
        "Component Audible Sound": predictions['audible_sound'],  # New field
        "Component Ultra Sound": predictions['ultra_sound'],      # New field
        "Overall health": overall_health,
        "Unhealthy TimeStamp": expected_timestamp
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
