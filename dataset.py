import pandas as pd
import numpy as np
import os

# Function to read CSV files and randomly distribute rows
def distribute_rows(csv_file1, csv_file2, output_file):
    # Read the CSV files into DataFrames
    df1 = pd.read_csv(csv_file1)
    df2 = pd.read_csv(csv_file2)
    
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

