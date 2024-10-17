from flask import Flask, request, jsonify
import pandas as pd
import joblib
from collections import Counter

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
    df_input = pd.DataFrame([input_data])
    predictions = {}
    
    for model_name in model_names:
        features = feature_sets[model_name]
        model = models[model_name]
        X_input = df_input[features]
        prediction = model.predict(X_input)[0]
        predictions[model_name.replace('_model', '')] = prediction
    
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
    predictions = predict_from_models(input_data)
    
    # Calculate average temperature and maximum vibration
    temperature = (input_data['temperature_one'] + input_data['temperature_two']) / 2
    vibration = max(input_data['vibration_x'], input_data['vibration_y'], input_data['vibration_z'])
    
    machine_condition = evaluate_machine_condition(temperature, vibration)
    temperature_anomaly = detect_temperature_anomaly(temperature)
    vibration_anomaly = detect_vibration_anomaly(vibration)
    
    overall_health = determine_overall_health(predictions)
    
    response = {
        "Component Temperature": predictions['temperature'],
        "Component Vibration": predictions['vibration'],
        "Component Magnetic Flux": predictions['magnetic_flux'],
        "Component Audible Sound": predictions['audible_sound'],
        "Component Ultra Sound": predictions['ultra_sound'],
        "Machine Condition": machine_condition,
        "Temperature Anomaly": temperature_anomaly,
        "Vibration Anomaly": vibration_anomaly,
        "Average Temperature": temperature,
        "Maximum Vibration": vibration,
        "Overall Health": overall_health
    }
    
    return jsonify(response)

# New route to handle 200 entries at once
@app.route('/anomaly20010', methods=['POST'])
def anomaly20010():
    input_data_list = request.json  # Expecting a list of 200 entries
    
    if not isinstance(input_data_list, list) or len(input_data_list) > 200:
        return jsonify({"error": "Input must be a list of up to 200 entries"}), 400

    # Lists to collect results for majority voting or averaging
    temperatures = []
    vibrations = []
    magnetic_fluxes = []
    audible_sounds = []
    ultra_sounds = []
    
    machine_conditions = []
    temperature_anomalies = []
    vibration_anomalies = []
    overall_healths = []
    
    # Lists to collect average values for temperature and maximum vibration
    avg_temperatures = []
    max_vibrations = []

    for input_data in input_data_list:
        predictions = predict_from_models(input_data)
        
        # Calculate average temperature and maximum vibration
        temperature = (input_data['temperature_one'] + input_data['temperature_two']) / 2
        vibration = max(input_data['vibration_x'], input_data['vibration_y'], input_data['vibration_z'])
        
        machine_condition = evaluate_machine_condition(temperature, vibration)
        temperature_anomaly = detect_temperature_anomaly(temperature)
        vibration_anomaly = detect_vibration_anomaly(vibration)
        overall_health = determine_overall_health(predictions)
        
        # Collect results for each key
        temperatures.append(predictions['temperature'])
        vibrations.append(predictions['vibration'])
        magnetic_fluxes.append(predictions['magnetic_flux'])
        audible_sounds.append(predictions['audible_sound'])
        ultra_sounds.append(predictions['ultra_sound'])
        
        machine_conditions.append(machine_condition)
        temperature_anomalies.append(temperature_anomaly)
        vibration_anomalies.append(vibration_anomaly)
        overall_healths.append(overall_health)
        
        avg_temperatures.append(temperature)
        max_vibrations.append(vibration)

    # Calculate majority results for categorical keys
    majority_machine_condition = Counter(machine_conditions).most_common(1)[0][0]
    majority_temperature_anomaly = Counter(temperature_anomalies).most_common(1)[0][0]
    majority_vibration_anomaly = Counter(vibration_anomalies).most_common(1)[0][0]
    majority_overall_health = Counter(overall_healths).most_common(1)[0][0]
    
    # Calculate average results for numeric keys
    avg_component_temperature = sum(temperatures) / len(temperatures)
    avg_component_vibration = sum(vibrations) / len(vibrations)
    avg_component_magnetic_flux = sum(magnetic_fluxes) / len(magnetic_fluxes)
    avg_component_audible_sound = sum(audible_sounds) / len(audible_sounds)
    avg_component_ultra_sound = sum(ultra_sounds) / len(ultra_sounds)
    
    # Calculate overall average temperature and max vibration across all 200 entries
    overall_avg_temperature = sum(avg_temperatures) / len(avg_temperatures)
    overall_max_vibration = max(max_vibrations)

    # Prepare the single majority-based response
    response = {
        "Component Temperature": avg_component_temperature,
        "Component Vibration": avg_component_vibration,
        "Component Magnetic Flux": avg_component_magnetic_flux,
        "Component Audible Sound": avg_component_audible_sound,
        "Component Ultra Sound": avg_component_ultra_sound,
        "Machine Condition": majority_machine_condition,
        "Temperature Anomaly": majority_temperature_anomaly,
        "Vibration Anomaly": majority_vibration_anomaly,
        "Average Temperature": overall_avg_temperature,
        "Maximum Vibration": overall_max_vibration,
        "Overall Health": majority_overall_health
    }

    return jsonify(response)
if __name__ == '__main__':
    app.run(debug=True)
