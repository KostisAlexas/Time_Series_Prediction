import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, clone_model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Concatenate, Flatten
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

# ------------------------- PARAMETERS -------------------------
# CSV file containing data for all nodes
csv_file = "test"

# General model training node id range
general_node_start = 600   # inclusive
general_node_end = 700    # inclusive

# Fine-tuning node id range (can be different from general training range)
finetune_node_start = 100  # inclusive
finetune_node_end = 150    # inclusive

sequence_length = 32
general_epochs = 1
finetune_epochs = 5
batch_size = 64

# Embedding parameters
embedding_dim = 8
# Assume node ids range from 1 to max_node_id, adjust accordingly
max_node_id = 13116 # Change as needed

# ------------------------- DATA LOADING -------------------------
df = pd.read_csv(csv_file)

# ------------------------- SEQUENCE CREATION FUNCTION -------------------------
def create_sequences(data, seq_length):
    X_seq, y = [], []
    for i in range(len(data) - seq_length):
        X_seq.append(data[i:i + seq_length])
        # Only predict CPU usage (assumes CPU is first column)
        y.append(data[i + seq_length, 0])
    return np.array(X_seq), np.array(y)

# ------------------------- PREPARE GENERAL MODEL DATA -------------------------
# Filter nodes for general model training based on nodeid range
df_general = df[(df['nodeid'] >= general_node_start) & (df['nodeid'] <= general_node_end)].copy()
df_general.sort_values(by=['nodeid', 'timestamp'], inplace=True)

X_seq_list, X_nodeid_list, y_list = [], [], []
nodes_general = df_general['nodeid'].unique()

for node in nodes_general:
    df_node = df_general[df_general['nodeid'] == node].sort_values(by='timestamp')
    # Use both CPU and Memory as input features (columns: CPU, Memory)
    node_data = df_node[['node_cpu_usage', 'node_memory_usage']].values
    if len(node_data) <= sequence_length:
        continue
    X_seq, y = create_sequences(node_data, sequence_length)
    X_seq_list.append(X_seq)
    y_list.append(y)
    # Create a matching array for node id (same id for every sequence)
    node_ids = np.full((X_seq.shape[0],), node)
    X_nodeid_list.append(node_ids)

# Combine data from all nodes
X_seq_all = np.concatenate(X_seq_list, axis=0)
X_nodeid_all = np.concatenate(X_nodeid_list, axis=0)
y_all = np.concatenate(y_list, axis=0)

# Split into training and testing sets (80/20 split)
X_seq_train, X_seq_test, X_nodeid_train, X_nodeid_test, y_train, y_test = train_test_split(
    X_seq_all, X_nodeid_all, y_all, test_size=0.2, random_state=42, shuffle=False
)

# Convert to tensors
X_seq_train = tf.convert_to_tensor(X_seq_train, dtype=tf.float32)
X_seq_test = tf.convert_to_tensor(X_seq_test, dtype=tf.float32)
X_nodeid_train = tf.convert_to_tensor(X_nodeid_train, dtype=tf.int32)
X_nodeid_test = tf.convert_to_tensor(X_nodeid_test, dtype=tf.int32)
y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)

# ------------------------- BUILD MODEL FUNCTION -------------------------
def build_model():
    # Input for timeseries data
    ts_input = Input(shape=(sequence_length, 2), name="ts_input")
    # Process timeseries with LSTM layers
    x = LSTM(64, activation='relu', return_sequences=True)(ts_input)
    x = LSTM(64, activation='relu')(x)

    # Input for node id
    node_input = Input(shape=(), dtype=tf.int32, name="node_input")
    x_node = Embedding(input_dim=max_node_id+1, output_dim=embedding_dim)(node_input)
    x_node = Flatten()(x_node)

    # Concatenate LSTM output and node embedding
    combined = Concatenate()([x, x_node])
    combined = Dense(32, activation='relu')(combined)
    # Output: Predict CPU usage only
    output = Dense(1, name="cpu_output")(combined)

    model = Model(inputs=[ts_input, node_input], outputs=output)
    model.compile(optimizer='adam', loss='mse')
    return model

# ------------------------- TRAIN GENERAL MODEL -------------------------
general_model = build_model()
print("Training general model on nodes {} to {}...".format(general_node_start, general_node_end))
general_model.fit(
    [X_seq_train, X_nodeid_train], y_train,
    epochs=general_epochs,
    batch_size=batch_size,
    validation_data=([X_seq_test, X_nodeid_test], y_test),
    verbose=1
)

general_loss = general_model.evaluate([X_seq_test, X_nodeid_test], y_test, verbose=0)
print(f"General Model Test Loss: {general_loss:.4f}")

# Function to compute MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Get predictions for general model on test set
y_pred_all = general_model.predict([X_seq_test, X_nodeid_test])
general_mape = mean_absolute_percentage_error(y_test.numpy(), y_pred_all)
print(f"General Model Aggregated Test MAPE: {general_mape:.2f}%")

# ------------------------- FINE-TUNING PER NODE -------------------------
# Fine-tuning node range filter
df_finetune = df[(df['nodeid'] >= finetune_node_start) & (df['nodeid'] <= finetune_node_end)].copy()
df_finetune.sort_values(by=['nodeid', 'timestamp'], inplace=True)
nodes_finetune = df_finetune['nodeid'].unique()

node_errors = {}

for node in nodes_finetune:
    df_node = df_finetune[df_finetune['nodeid'] == node].sort_values(by='timestamp')
    node_data = df_node[['node_cpu_usage', 'node_memory_usage']].values
    if len(node_data) <= sequence_length:
        continue
    X_seq_node, y_node = create_sequences(node_data, sequence_length)

    # Split into train and test (80/20 split)
    X_seq_train_node, X_seq_test_node, y_train_node, y_test_node = train_test_split(
        X_seq_node, y_node, test_size=0.2, random_state=42, shuffle=False
    )

    # Create node id arrays for this node (same constant value)
    X_nodeid_train_node = np.full((X_seq_train_node.shape[0],), node)
    X_nodeid_test_node = np.full((X_seq_test_node.shape[0],), node)

    # Convert to tensors
    X_seq_train_node = tf.convert_to_tensor(X_seq_train_node, dtype=tf.float32)
    X_seq_test_node = tf.convert_to_tensor(X_seq_test_node, dtype=tf.float32)
    X_nodeid_train_node = tf.convert_to_tensor(X_nodeid_train_node, dtype=tf.int32)
    X_nodeid_test_node = tf.convert_to_tensor(X_nodeid_test_node, dtype=tf.int32)
    y_train_node = tf.convert_to_tensor(y_train_node, dtype=tf.float32)
    y_test_node = tf.convert_to_tensor(y_test_node, dtype=tf.float32)

    # Clone the general model and load its weights for fine-tuning
    node_model = build_model()
    node_model.set_weights(general_model.get_weights())

    # Fine-tune on this node's data
    node_model.fit(
        [X_seq_train_node, X_nodeid_train_node], y_train_node,
        epochs=finetune_epochs,
        batch_size=batch_size,
        verbose=0
    )

    # Evaluate the fine-tuned model on the node's test data
    loss = node_model.evaluate([X_seq_test_node, X_nodeid_test_node], y_test_node, verbose=0)
    y_pred_node = node_model.predict([X_seq_test_node, X_nodeid_test_node])
    mape = mean_absolute_percentage_error(y_test_node.numpy(), y_pred_node)

    # Print sample predictions vs. actual CPU values (first 5 samples)
    print(f"\nNode {node} Fine-tuning Results:")
    for i in range(min(5, len(y_test_node))):
        print(f"Sample {i+1}: Actual CPU = {y_test_node.numpy()[i]:.4f}, Predicted CPU = {y_pred_node[i,0]:.4f}")

    node_errors[node] = {'loss': loss, 'MAPE': mape}
    print(f"Node {node}: Loss = {loss:.4f}, MAPE = {mape:.2f}%")

# Print aggregated fine-tuning errors for all nodes
print("\nFine-tuning results for nodes {} to {}:".format(finetune_node_start, finetune_node_end))
for node in sorted(node_errors.keys()):
    err = node_errors[node]
    print(f"Node {node}: Loss = {err['loss']:.4f}, MAPE = {err['MAPE']:.2f}%")

