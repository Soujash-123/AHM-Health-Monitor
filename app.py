from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load the models
model_names = [
    'temperature_model', 
    'vibration_model', 
    'magnetic_flux_model',
    'audible_sound_model',
    'ultra_sound_model'
]
models = {name: joblib.load(f"{name}.pkl") for name in model_names}

# Define the feature sets used by each model
feature_sets = {
    'temperature_model': ['temperature_one', 'temperature_two'],
    'vibration_model': ['vibration_x', 'vibration_y', 'vibration_z'],
    'magnetic_flux_model': ['magnetic_flux_x', 'magnetic_flux_y', 'magnetic_flux_z'],
    'audible_sound_model': ['vibration_x', 'vibration_y', 'vibration_z', 'audible_sound'],
    'ultra_sound_model': ['vibration_x', 'vibration_y', 'vibration_z', 'ultra_sound']
}

def predict_from_models(input_data):
    df_input = pd.DataFrame(input_data)
    predictions = {model_name.replace('_model', ''): [] for model_name in model_names}
    
    for model_name in model_names:
        features = feature_sets[model_name]
        model = models[model_name]
        X_input = df_input[features]
        model_predictions = model.predict(X_input)
        predictions[model_name.replace('_model', '')] = model_predictions.tolist()
    
    return predictions

def evaluate_machine_condition(temperature, vibration):
    if temperature < 80 and vibration < 1.8:
        return "Safe Condition"
    elif temperature < 100 and vibration < 2.8:
        return "Maintain Condition"
    else:
        return "Repair Condition"

def detect_temperature_anomaly(temperature):
    if temperature < 80:
        return "No significant temperature anomaly detected"
    elif 80 <= temperature < 100:
        return "Moderate Overheating - Check Lubrication"
    elif 100 <= temperature < 120:
        return "Significant Overheating - Possible Misalignment or Bearing Wear"
    else:
        return "Critical Overheating - Immediate Repair Needed"

def detect_vibration_anomaly(vibration):
    if vibration < 1.8:
        return "No significant vibration anomaly detected"
    elif 1.8 <= vibration < 2.8:
        return "Unbalance Fault"
    elif 2.8 <= vibration < 4.5:
        return "Misalignment Fault"
    elif 4.5 <= vibration < 7.1:
        return "Looseness Fault"
    else:
        return "Bearing Fault or Gear Mesh Fault"

def determine_overall_health(predictions):
    # If all predictions are 0 (healthy), return "Healthy". Otherwise, return "Unhealthy".
    if all(status == 0 for status in predictions.values()) or all(status == "Healthy" for status in predictions.values()):
        return "Healthy"
    else:
        return "Unhealthy"

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.json
    if not isinstance(input_data, list) or len(input_data) > 200:
        return jsonify({"error": "Input should be a list of up to 200 entries."}), 400

    predictions = predict_from_models(input_data)
    
    results = []
    for idx, entry in enumerate(input_data):
        # Calculate average temperature and maximum vibration for each entry
        temperature = (entry['temperature_one'] + entry['temperature_two']) / 2
        vibration = max(entry['vibration_x'], entry['vibration_y'], entry['vibration_z'])
        
        machine_condition = evaluate_machine_condition(temperature, vibration)
        temperature_anomaly = detect_temperature_anomaly(temperature)
        vibration_anomaly = detect_vibration_anomaly(vibration)
        
        entry_result = {
            "Component Temperature": predictions['temperature'][idx],
            "Component Vibration": predictions['vibration'][idx],
            "Component Magnetic Flux": predictions['magnetic_flux'][idx],
            "Component Audible Sound": predictions['audible_sound'][idx],
            "Component Ultra Sound": predictions['ultra_sound'][idx],
            "Machine Condition": machine_condition,
            "Temperature Anomaly": temperature_anomaly,
            "Vibration Anomaly": vibration_anomaly,
            "Average Temperature": temperature,
            "Maximum Vibration": vibration,
            "Overall Health": determine_overall_health({
                'temperature': predictions['temperature'][idx],
                'vibration': predictions['vibration'][idx],
                'magnetic_flux': predictions['magnetic_flux'][idx],
                'audible_sound': predictions['audible_sound'][idx],
                'ultra_sound': predictions['ultra_sound'][idx]
            })
        }
        
        results.append(entry_result)
    
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
