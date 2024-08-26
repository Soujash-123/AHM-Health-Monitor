import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset
df = pd.read_csv('dataset2.csv')

# Define feature subsets
feature_sets = {
    'temperature_model': ['temperature_one', 'temperature_two'],
    'vibration_model': ['vibration_x', 'vibration_y', 'vibration_z'],
    'magnetic_flux_model': ['magnetic_flux_x', 'magnetic_flux_y', 'magnetic_flux_z'],
    'audible_sound_model': ['vibration_x', 'vibration_y', 'vibration_z', 'audible_sound'],  # New model
    'ultra_sound_model': ['vibration_x', 'vibration_y', 'vibration_z', 'ultra_sound']       # New model
}

# Initialize a dictionary to store models and their accuracy
models = {}
accuracies = {}

# Train and evaluate models
for model_name, features in feature_sets.items():
    X = df[features]
    y = df['health_status']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train the Random Forest Classifier
    rfc = RandomForestClassifier(random_state=42)
    rfc.fit(X_train, y_train)
    
    # Predict and evaluate the model
    y_pred = rfc.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Store the model and accuracy
    models[model_name] = rfc
    accuracies[model_name] = accuracy
    
    # Save the model to a file
    model_filename = f"{model_name}.pkl"
    joblib.dump(rfc, model_filename)
    print(f"{model_name} model saved as {model_filename}")
    print(f"{model_name} Accuracy: {accuracy:.2f}")

# Optional: Print the models' accuracies
print("\nModel Accuracies:")
for model_name, accuracy in accuracies.items():
    print(f"{model_name}: {accuracy:.2f}")
