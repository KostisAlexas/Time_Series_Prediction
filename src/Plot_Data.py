import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random

# Path to your large CSV file
csv_file = 'test'

# Define column names based on the dataset description.
# The CSV has 5 columns:
# - 1st column: unnamed event (ignored in analysis)
# - 2nd column: timestamp (numeric)
# - 3rd column: nodeid (unique identifier)
# - 4th column: node_cpu_usage (float, may contain non-numeric values)
# - 5th column: node_memory_usage (float, may contain non-numeric values)
columns = ['event', 'timestamp', 'nodeid', 'node_cpu_usage', 'node_memory_usage']

# Read the CSV in chunks to handle the large file size
chunksize = 10**5  # Adjust the chunksize based on your system's memory capacity
data_chunks = []

for chunk in pd.read_csv(csv_file, header=None, names=columns, chunksize=chunksize):
    # Convert the 'timestamp' to numeric (in case there are errors)
    chunk['timestamp'] = pd.to_numeric(chunk['timestamp'], errors='coerce')
    # Convert cpu and memory usage to numeric, setting errors as NaN for non-numeric values
    chunk['node_cpu_usage'] = pd.to_numeric(chunk['node_cpu_usage'], errors='coerce')
    chunk['node_memory_usage'] = pd.to_numeric(chunk['node_memory_usage'], errors='coerce')
    data_chunks.append(chunk)

# Concatenate all chunks into one DataFrame
df = pd.concat(data_chunks, ignore_index=True)

# Drop rows with missing timestamps
df = df.dropna(subset=['timestamp'])

# Ensure the DataFrame is sorted by timestamp for time series plots
df = df.sort_values('timestamp')

# -------------------------------
# Plot 1: CPU Usage Over Time for a Random Node
# -------------------------------
# Select a random node from the available node ids
unique_nodes = df['nodeid'].unique()
random_node = random.choice(unique_nodes)

# Filter data for the chosen node and sort by timestamp
df_node = df[df['nodeid'] == random_node].sort_values('timestamp')

plt.figure(figsize=(12, 6))
plt.plot(df_node['timestamp'], df_node['node_cpu_usage'], marker='.', linestyle='-', label='CPU Usage')
plt.title(f'CPU Usage over Time for Node {random_node}')
plt.xlabel('Timestamp')
plt.ylabel('CPU Usage')
plt.legend()
plt.tight_layout()
plt.show()

# -------------------------------
# Plot 2: Histogram of CPU Usage Distribution
# -------------------------------
plt.figure(figsize=(12, 6))
# Drop NaN values from CPU usage
plt.hist(df['node_cpu_usage'].dropna(), bins=50, edgecolor='black')
plt.title('Histogram of CPU Usage')
plt.xlabel('CPU Usage')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# -------------------------------
# Plot 3: Histogram of Memory Usage Distribution
# -------------------------------
plt.figure(figsize=(12, 6))
plt.hist(df['node_memory_usage'].dropna(), bins=50, edgecolor='black')
plt.title('Histogram of Memory Usage')
plt.xlabel('Memory Usage')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# -------------------------------
# Plot 4: Frequency of CPU Usage in Bins
# -------------------------------
# Create bins from 0 to 1 (assuming CPU usage is in the range [0,1])
bins = np.linspace(0, 1, 21)  # 20 bins
df['cpu_usage_bin'] = pd.cut(df['node_cpu_usage'], bins=bins)
cpu_counts = df['cpu_usage_bin'].value_counts().sort_index()

plt.figure(figsize=(12, 6))
cpu_counts.plot(kind='bar', edgecolor='black')
plt.title('Frequency of CPU Usage in Bins')
plt.xlabel('CPU Usage Range')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

