import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.metrics import MeanAbsoluteError
from sklearn.model_selection import train_test_split
import os

# Check for GPU availability
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPU is available. Training on GPU.")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("GPU is not available. Training on CPU.")

# Parameters
csv_file = "test"  # CSV containing data for all nodes
node_id_start = 100   # Starting node id for training
node_id_end = 150    # Ending node id for training
sequence_length = 64
general_epochs = 5
finetune_epochs = 10
batch_size = 64

# Load the dataset (assumes dataset contains data for all nodes with column 'nodeid')
df = pd.read_csv(csv_file)

# Filter nodes in the specified id range
df = df[(df['nodeid'] >= node_id_start) & (df['nodeid'] <= node_id_end)].copy()

# Function to create sequences for a given node's data
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

# Prepare lists to collect general training data across nodes
X_list = []
y_list = []
node_ids = df['nodeid'].unique()

# For each node, sort by timestamp and create sequences
for node in node_ids:
    df_node = df[df['nodeid'] == node].sort_values(by='timestamp')  # assumes there is a 'timestamp' column
    # Select only the relevant columns
    node_data = df_node[['node_cpu_usage', 'node_memory_usage']].values
    if len(node_data) <= sequence_length:
        continue
    X_node, y_node = create_sequences(node_data, sequence_length)
    X_list.append(X_node)
    y_list.append(y_node)

# Combine sequences from all nodes
X_all = np.concatenate(X_list, axis=0)
y_all = np.concatenate(y_list, axis=0)

# Split combined data into training and testing sets (80/20 split)
X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, shuffle=False
)

# Convert to TensorFlow tensors
X_train_all = tf.convert_to_tensor(X_train_all, dtype=tf.float32)
y_train_all = tf.convert_to_tensor(y_train_all, dtype=tf.float32)
X_test_all = tf.convert_to_tensor(X_test_all, dtype=tf.float32)
y_test_all = tf.convert_to_tensor(y_test_all, dtype=tf.float32)

# Define the general LSTM model
def build_model():
    model = Sequential([
        LSTM(64, activation='relu', return_sequences=True, input_shape=(sequence_length, 2)),
        LSTM(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(2)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Train the general model on all nodes data
general_model = build_model()
print("Training general model on all nodes data...")
general_model.fit(X_train_all, y_train_all, epochs=general_epochs, batch_size=batch_size,
                  validation_data=(X_test_all, y_test_all), verbose=1)

# Evaluate the general model on the aggregated test set
general_loss = general_model.evaluate(X_test_all, y_test_all, verbose=0)
print(f"General Model Test Loss: {general_loss:.4f}")

# Dictionary to store fine-tuning errors for each node
node_errors = {}

# Function to compute MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# For each node in the specified range, perform fine-tuning and evaluate error
for node in node_ids:
    df_node = df[df['nodeid'] == node].sort_values(by='timestamp')
    node_data = df_node[['node_cpu_usage', 'node_memory_usage']].values
    if len(node_data) <= sequence_length:
        continue
    X_node, y_node = create_sequences(node_data, sequence_length)

    # Split node data into train and test (80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X_node, y_node, test_size=0.2, random_state=42, shuffle=False
    )
    X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
    X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
    y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)

    # Clone the general model and load its weights
    node_model = build_model()
    node_model.set_weights(general_model.get_weights())

    # Fine-tune on this specific node's training data
    node_model.fit(X_train, y_train, epochs=finetune_epochs, batch_size=batch_size, verbose=0)

    # Evaluate on node test set
    loss = node_model.evaluate(X_test, y_test, verbose=0)
    y_pred = node_model.predict(X_test)
    mape = mean_absolute_percentage_error(y_test.numpy(), y_pred)
    node_errors[node] = {'loss': loss, 'MAPE': mape}
    print(f"Node {node}: Loss = {loss:.4f}, MAPE = {mape:.2f}%")

# Compute general error across all nodes test data (using the general model without fine-tuning)
y_pred_all = general_model.predict(X_test_all)
general_mape = mean_absolute_percentage_error(y_test_all.numpy(), y_pred_all)
print(f"\nGeneral Model Aggregated Test MAPE: {general_mape:.2f}%")

# Print node-specific errors
print("\nNode-specific errors (without charts):")
for node in sorted(node_errors.keys()):
    err = node_errors[node]
    print(f"Node {node}: Loss = {err['loss']:.4f}, MAPE = {err['MAPE']:.2f}%")

