import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Function to train, predict health_status, and save the model
def predict_health_status(csv_file, model_save_path='trained_model.pkl'):
    # Define the columns to use for prediction
    features = [
        'temperature_one', 'temperature_two', 'vibration_x', 'vibration_y', 'vibration_z',
        'magnetic_flux_x', 'magnetic_flux_y', 'magnetic_flux_z'
    ]
    
    # Load the dataset
    df = pd.read_csv(csv_file)
    
    # Separate features and target variable
    X = df[features]
    y = df['health_status']
    
    # Split the data into training and temporary sets (60-40)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    
    # Split the temporary set into validation and test sets (50-50)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Initialize the Random Forest Classifier with regularization
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=4,
        random_state=42
    )
    
    # Train the model
    clf.fit(X_train, y_train)
    
    # Save the trained model
    joblib.dump(clf, model_save_path)
    print(f"Model saved to {model_save_path}")
    
    # Evaluate the model on the training set
    y_train_pred = clf.predict(X_train)
    print("Training Accuracy Score:", accuracy_score(y_train, y_train_pred))
    print("Training Classification Report:")
    print(classification_report(y_train, y_train_pred))
    
    # Evaluate the model on the validation set
    y_val_pred = clf.predict(X_val)
    print("Validation Accuracy Score:", accuracy_score(y_val, y_val_pred))
    print("Validation Classification Report:")
    print(classification_report(y_val, y_val_pred))
    
    # Evaluate the model on the test set
    y_test_pred = clf.predict(X_test)
    print("Test Accuracy Score:", accuracy_score(y_test, y_test_pred))
    print("Test Classification Report:")
    print(classification_report(y_test, y_test_pred))
    
    # Perform k-fold cross-validation
    cross_validate_model(csv_file)

# Function to perform k-fold cross-validation
def cross_validate_model(csv_file, k=5):
    # Define the columns to use for prediction
    features = [
        'temperature_one', 'temperature_two', 'vibration_x', 'vibration_y', 'vibration_z',
        'magnetic_flux_x', 'magnetic_flux_y', 'magnetic_flux_z'
    ]
    
    # Load the dataset
    df = pd.read_csv(csv_file)
    
    # Separate features and target variable
    X = df[features]
    y = df['health_status']
    
    # Initialize the Random Forest Classifier
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=4,
        random_state=42
    )
    
    # Perform k-fold cross-validation
    scores = cross_val_score(clf, X, y, cv=k, scoring='accuracy')
    
    print(f"{k}-Fold Cross-Validation Accuracy Scores: {scores}")
    print(f"Mean Accuracy: {scores.mean()}")
    print(f"Standard Deviation: {scores.std()}")

# Function for hyperparameter tuning
def tune_hyperparameters(csv_file):
    # Define the columns to use for prediction
    features = [
        'temperature_one', 'temperature_two', 'vibration_x', 'vibration_y', 'vibration_z',
        'magnetic_flux_x', 'magnetic_flux_y', 'magnetic_flux_z'
    ]
    
    # Load the dataset
    df = pd.read_csv(csv_file)
    
    # Separate features and target variable
    X = df[features]
    y = df['health_status']
    
    # Define the parameter grid for Grid Search
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 4]
    }
    
    # Initialize the Random Forest Classifier
    clf = RandomForestClassifier(random_state=42)
    
    # Perform Grid Search
    grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X, y)
    
    print("Best Parameters:", grid_search.best_params_)
    print("Best Cross-Validation Score:", grid_search.best_score_)

# Example usage
csv_file = 'dataset2.csv'
predict_health_status(csv_file, 'health_status_model.pkl')
tune_hyperparameters(csv_file)
