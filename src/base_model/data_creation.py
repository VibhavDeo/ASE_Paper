import pandas as pd
import os

# Define the directory containing the CSV files
input_directory = '/Users/priyaandurkar/Documents/Academic Terms/Spring 2024/ASE/aa24-main/data/flash'
output_directory = 'data/'

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Function to process each CSV file
def process_csv(file_path, file_name):
    # Read the original CSV file
    df = pd.read_csv(file_path)

    # Identify columns with trailing '-' or '+' in their headers
    x_columns = [col for col in df.columns if not (col.endswith('-') or col.endswith('+'))]
    y_columns = [col for col in df.columns if col.endswith('-') or col.endswith('+')]

    # Create two separate dataframes
    df_x = df[x_columns]
    df_y = df[y_columns]

    # Generate filenames for the output CSVs
    base_name = os.path.basename(file_name)
    x_output_file = os.path.join(output_directory, f'x_{file_name}')
    y_output_file = os.path.join(output_directory, f'y_{file_name}')

    # Save the dataframes to separate CSV files
    df_x.to_csv(x_output_file, index=False)
    df_y.to_csv(y_output_file, index=False)

# Iterate through all CSV files in the input directory
for file_name in os.listdir(input_directory):
    if file_name.endswith('.csv'):
        file_path = os.path.join(input_directory, file_name)
        process_csv(file_path, file_name)

print("CSV files have been processed and split successfully.")
