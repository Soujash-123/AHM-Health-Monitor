Health Status Prediction and Analysis
Overview

This project aims to predict health status using machine learning models and perform an in-depth analysis of the dataset. It includes:

    Model Training and Saving: Training multiple Random Forest models based on different feature sets and saving them.
    Prediction and Evaluation: Predicting health status based on input data and evaluating the overall state.
    Data Visualization: Generating various plots to analyze feature distributions, correlations, and model performance.

Project Structure

    model.py: Contains code to train and save Random Forest models based on different feature sets.
    Test.py: Used to load the saved models, make predictions on new data, and evaluate the state of the system based on those predictions.
    dataVis.py: Handles data visualization including box plots, violin plots, correlation matrices, feature importance, and decision trees.

Requirements

Ensure you have the following libraries installed:

    pandas
    numpy
    scikit-learn
    seaborn
    matplotlib
    joblib

You can install the necessary packages using pip:

bash

pip install pandas numpy scikit-learn seaborn matplotlib joblib

Dataset

The dataset used in this project is dataset2.csv, which should be placed in the same directory as the scripts. The dataset should contain the following columns:

    Features: temperature_one, temperature_two, vibration_x, vibration_y, vibration_z, magnetic_flux_x, magnetic_flux_y, magnetic_flux_z
    Target Variable: health_status (categorical labels: 'healthy', 'unhealthy')

Usage
Training Models

Run model.py to train models based on different feature subsets and save them as .pkl files. The models will be saved in the current directory.

bash

python model.py

Testing and Evaluation

Use Test.py to load the saved models, make predictions on new data, and evaluate the system state.

bash

python Test.py

Update the input_data dictionary in Test.py with the actual data you want to test.
Data Visualization

Run dataVis.py to generate various visualizations and analyze the dataset.

bash

python dataVis.py

The generated plots and statistics will be saved in the health_status_analysis directory.
Outputs

    model.py: Saves trained models as .pkl files.
    Test.py: Prints predictions and evaluation results.
    dataVis.py: Saves plots and descriptive statistics in the health_status_analysis directory.

