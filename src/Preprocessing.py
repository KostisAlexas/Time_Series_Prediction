import pandas as pd
import numpy as np

# Load the CSV file
df = pd.read_csv('test')

# Step 1: Remove rows with missing or non-numeric values in critical columns
# Critical columns: 'timestamp', 'node_cpu_usage', 'node_memory_usage'
cols = ['timestamp', 'node_cpu_usage', 'node_memory_usage']
for col in cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df.dropna(subset=cols, inplace=True)

# Ensure cpu and memory columns are floats
df['node_cpu_usage'] = df['node_cpu_usage'].astype(float)
df['node_memory_usage'] = df['node_memory_usage'].astype(float)

# Step 2: Process the timestamp.
# There are 1440 intervals in 12 hours with a 30-second interval.
# Represent each timestamp as an integer index: 0 for 0 ms, 1 for 30000 ms, 2 for 60000 ms, etc.
df['timestamp'] = (df['timestamp'] / 30000).astype(int)

# Step 3: Round node_cpu_usage and node_memory_usage to 4 decimal places using standard rounding
df['node_cpu_usage'] = df['node_cpu_usage'].round(4)
df['node_memory_usage'] = df['node_memory_usage'].round(4)

# Step 4: Remove the first unnamed column if it exists
if df.columns[0].startswith('Unnamed'):
    df.drop(df.columns[0], axis=1, inplace=True)

# Sort by nodeid and then by timestamp (so each node's time series is in order)
df.sort_values(by=['nodeid', 'timestamp'], inplace=True)
df.reset_index(drop=True, inplace=True)

# Replace nodeid with numeric mapping (first appearance becomes 1, next becomes 2, etc.)
unique_nodes = df['nodeid'].unique()
node_mapping = {node: idx+1 for idx, node in enumerate(unique_nodes)}
df['nodeid'] = df['nodeid'].map(node_mapping)

# Save the cleaned data to a new CSV file
df.to_csv('/home/ohyeah/Documents/Alibaba_Dataset_Node_0/cleaned_data.csv', index=False)
