import json
import pandas as pd
from rich import print
from sklearn.model_selection import train_test_split

# Load local Parquet dataset
parquet_file_path = './dataset/instruction_dataset.parquet'
dataset = pd.read_parquet(parquet_file_path)

# Print the columns of the dataset to verify
print("Dataset Columns:", dataset.columns)

# Convert dataset to list of dictionaries
json_data = dataset.to_dict(orient='list')

# Print the first row to verify
print("First Row:", {key: json_data[key][0] for key in json_data})

# Initialize lists to store Alpaca format data
alpaca_data = []

# Process each entry in the dataset
for i in range(len(json_data['instruction'])):
    instruction = json_data['instruction'][i]
    input_text = json_data['input'][i]
    original_output = json_data['output'][i]

    # Create Alpaca format entry
    alpaca_entry = {
        "instruction": instruction,
        "input": input_text,
        "output": original_output
    }
    alpaca_data.append(alpaca_entry)

# Split the dataset into training and testing sets (80% training, 20% testing)
train_data, test_data = train_test_split(alpaca_data, test_size=0.2, random_state=42)

# Save training set in Alpaca format
with open("../data/alpaca_train_dataset.json", "w") as file:
    json.dump(train_data, file, indent=4)

# Save testing set in Alpaca format
with open("../data/alpaca_test_dataset.json", "w") as file:
    json.dump(test_data, file, indent=4)

print("Alpaca format training dataset saved as 'alpaca_train_dataset.json'")
print("Alpaca format testing dataset saved as 'alpaca_test_dataset.json'")
