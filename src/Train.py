import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Reshape
from tensorflow.keras.callbacks import EarlyStopping

# Check and configure GPU usage
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU is available and configured.")
    except RuntimeError as e:
        print(e)
else:
    print("GPU not available, using CPU.")

# Load dataset
csv_file = "test"
df = pd.read_csv(csv_file)

# Unique node IDs
unique_nodes = df['nodeid'].unique()

# Select 30 random nodes (change to 30 to avoid too many computations)
random_nodes = random.sample(list(unique_nodes), 1)

# Define sequence length (lookback period)
sequence_length = 30  

# Define prediction horizons (5 and 15 timesteps ahead)
horizons = {"short": 5, "long": 15}  

# Function to create sequences
def create_sequences(data, seq_length, horizon):
    X, y = [], []
    for i in range(len(data) - seq_length - horizon):
        X.append(data[i:i + seq_length])  # Input sequence
        y.append(data[i + seq_length + horizon - 1])  # Predict the exact future step
    return np.array(X), np.array(y)

# Store all results
final_results = {"short": {"mse": [], "mae": [], "mape": []},
                 "long": {"mse": [], "mae": [], "mape": []}}

# Create the plots dictionary to store the predictions and actual values
plots_data = {"short": {"actual": [], "predicted": []},
              "long": {"actual": [], "predicted": []}}

for node_id in random_nodes:
    df_node = df[df['nodeid'] == node_id].copy()

    # Ensure enough data points
    if len(df_node) < sequence_length + max(horizons.values()):
        continue  

    # Keep only CPU usage column
    node_data = df_node[['node_cpu_usage']].values

    for horizon_type, horizon in horizons.items():
        # Create sequences
        X_node, y_node = create_sequences(node_data, sequence_length, horizon)

        # Ensure 80/20 split where last 20% is the test set
        split_index = int(len(X_node) * 0.8)
        X_train, X_test = X_node[:split_index], X_node[split_index:]
        y_train, y_test = y_node[:split_index], y_node[split_index:]

        # Convert to TensorFlow tensors
        X_train, y_train = tf.convert_to_tensor(X_train, dtype=tf.float32), tf.convert_to_tensor(y_train, dtype=tf.float32)
        X_test, y_test = tf.convert_to_tensor(X_test, dtype=tf.float32), tf.convert_to_tensor(y_test, dtype=tf.float32)

        # Define LSTM Model
        model = Sequential([
            tf.keras.layers.Input(shape=(sequence_length, 1)),  
            LSTM(64, activation='relu', return_sequences=True),
            LSTM(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(1)  # Predicts only CPU usage for a single future timestep
        ])

        # Compile Model
        model.compile(optimizer='adam', loss='mse')

        # Define early stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        # Train Model with early stopping (training will use GPU if available)
        model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test), callbacks=[early_stopping], verbose=0)

        # Get Predictions
        y_pred = model.predict(X_test)

        # Store metrics
        mse = np.mean((y_test.numpy() - y_pred) ** 2)
        mae = np.mean(np.abs(y_test.numpy() - y_pred))
        mape = np.mean(np.abs((y_test.numpy() - y_pred) / y_test.numpy())) * 100

        # Store metrics for evaluation
        final_results[horizon_type]["mse"].append(mse)
        final_results[horizon_type]["mae"].append(mae)
        final_results[horizon_type]["mape"].append(mape)

        # Store actual and predicted values for plotting
        plots_data[horizon_type]["actual"].extend(y_test.numpy().flatten())
        plots_data[horizon_type]["predicted"].extend(y_pred.flatten())

# Compute final averages
for horizon_type in horizons.keys():
    final_mse = np.mean(final_results[horizon_type]["mse"])
    final_mae = np.mean(final_results[horizon_type]["mae"])
    final_mape = np.mean(final_results[horizon_type]["mape"])

    print(f"\nFinal Accuracy ({horizon_type.capitalize()}-Term Prediction):")
    print(f"Test MSE: {final_mse:.6f}")
    print(f"Test MAE: {final_mae:.6f}")
    print(f"Test MAPE: {final_mape:.2f}%")

# Plot Actual vs Predicted values for the unknown (test) data
for horizon_type in horizons.keys():
    plt.figure(figsize=(10, 6))
    plt.plot(plots_data[horizon_type]["actual"], label="Actual")
    plt.plot(plots_data[horizon_type]["predicted"], label="Predicted", linestyle='--')
    plt.title(f"Actual vs Predicted CPU Usage - {horizon_type.capitalize()} Horizon")
    plt.xlabel("Time Step")
    plt.ylabel("CPU Usage")
    plt.legend()
    plt.show()

