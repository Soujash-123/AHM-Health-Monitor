import pandas as pd
import joblib

# Load the models
model_names = ['temperature_model', 'vibration_model', 'magnetic_flux_model']
models = {name: joblib.load(f"{name}.pkl") for name in model_names}

# Define the feature sets used by each model
feature_sets = {
    'temperature_model': ['temperature_one', 'temperature_two'],
    'vibration_model': ['vibration_x', 'vibration_y', 'vibration_z'],
    'magnetic_flux_model': ['magnetic_flux_x', 'magnetic_flux_y', 'magnetic_flux_z']
}

def predict_from_models(input_data):
    # Create a DataFrame from the input data
    df_input = pd.DataFrame([input_data])
    
    # List to store predictions
    predictions = []
    
    # Predict using each model
    for model_name in model_names:
        features = feature_sets[model_name]
        model = models[model_name]
        
        # Select relevant features from the input data
        X_input = df_input[features]
        
        # Predict and append the result to the list
        prediction = model.predict(X_input)[0]
        predictions.append(prediction)
    
    return predictions
def evaluate_state(predictions):
    # Determine state based on predictions
    all_healthy = all(p == 'healthy' for p in predictions)
    all_unhealthy = all(p == 'unhealthy' for p in predictions)
    
    if all_healthy:
        print("Healthy state")
    elif all_unhealthy:
        print("Unhealthy state")
    else:
        unhealthy_indexes = [model_names[i] for i, p in enumerate(predictions) if p == 'unhealthy']
        print(f"Warning. Unhealthy component detected at: {unhealthy_indexes}")
# Example input data (replace with actual data)
#22.24	18.69	0.01	0.04	0.13	0.27	0.093	0.115


input_data = {
    'temperature_one': 2,
    'temperature_two': 0,
    'vibration_x': 0.01,
    'vibration_y': 0.04,
    'vibration_z': 0.13,
    'magnetic_flux_x': 0,
    'magnetic_flux_y': 0,
    'magnetic_flux_z': 0
}

# Get predictions
predictions = predict_from_models(input_data)
print("Predictions from each model:")
print(predictions)
evaluate_state(predictions)

