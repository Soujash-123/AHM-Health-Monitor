import pandas as pd
import numpy as np
import os

# Function to read CSV files and randomly distribute rows
def distribute_rows(csv_file1, csv_file2, output_file):
    # Define the columns to keep
    columns_to_keep = [
        'temperature_one', 'temperature_two', 'vibration_x', 'vibration_y', 'vibration_z',
        'magnetic_flux_x', 'magnetic_flux_y', 'magnetic_flux_z', 'health_status'
    ]
    
    # Read the CSV files into DataFrames, selecting only the specified columns
    df1 = pd.read_csv(csv_file1, usecols=columns_to_keep)
    df2 = pd.read_csv(csv_file2, usecols=columns_to_keep)
    
    # Combine the rows from both DataFrames
    combined_df = pd.concat([df1, df2], ignore_index=True)
    
    # Shuffle the rows
    shuffled_df = combined_df.sample(frac=1).reset_index(drop=True)
    
    # Write the shuffled DataFrame to a new CSV file
    shuffled_df.to_csv(output_file, index=False)

# Example usage
csv_file1 = 'DE_BOF_15438.csv'
csv_file2 = 'NDE_BOF_15438.csv'
output_file = 'dataset.csv'

distribute_rows(csv_file1, csv_file2, output_file)
