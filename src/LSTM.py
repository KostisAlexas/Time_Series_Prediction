import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.metrics import MeanAbsoluteError, MeanSquaredError

# Load the dataset
csv_file = "C:/Users/tasis/Downloads/Node_0/cleaned_data.csv"
df = pd.read_csv(csv_file)

# Select a specific node
node_id_to_use = 500 # Change this to analyze a different node
df_node = df[df['nodeid'] == node_id_to_use].copy()

# Keep only relevant columns (CPU & Memory usage)
node_data = df_node[['node_cpu_usage', 'node_memory_usage']].values

# Define sequence length (Lookback period)
sequence_length = 30

# Function to create sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])  # Last 60 timestamps
        y.append(data[i + seq_length])    # Next step (CPU & Memory usage)
    return np.array(X), np.array(y)

# Create sequences
X_node, y_node = create_sequences(node_data, sequence_length)

# Split into train & test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_node, y_node, test_size=0.2, random_state=42, shuffle=False
)

# Convert to TensorFlow tensors
X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)

# Define LSTM Model
model = Sequential([
    LSTM(64, activation='relu', return_sequences=True, input_shape=(sequence_length, 2)),
    LSTM(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(2)  # Predicts next CPU & Memory Usage
])

# Compile Model
model.compile(optimizer='adam', loss='mse')

# Train Model with history tracking
history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test))

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")

# Generate predictions
y_pred = model.predict(X_test)
# Plot Training & Validation Loss

plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.show()

# Plot actual vs. predicted CPU usage
plt.figure(figsize=(10, 5))
plt.plot(y_test.numpy()[:, 0], label="Actual CPU Usage", color="blue", alpha=0.7)
plt.plot(y_pred[:, 0], label="Predicted CPU Usage", color="red", linestyle="dashed")
plt.xlabel("Time Steps")
plt.ylabel("CPU Usage")
plt.title("Actual vs. Predicted CPU Usage")
plt.legend()
plt.grid(True)
plt.show()

# Plot actual vs. predicted Memory usage
plt.figure(figsize=(10, 5))
plt.plot(y_test.numpy()[:, 1], label="Actual Memory Usage", color="green", alpha=0.7)
plt.plot(y_pred[:, 1], label="Predicted Memory Usage", color="orange", linestyle="dashed")
plt.xlabel("Time Steps")
plt.ylabel("Memory Usage")
plt.title("Actual vs. Predicted Memory Usage")
plt.legend()
plt.grid(True)
plt.show()

# Evaluate the model
mse = model.evaluate(X_test, y_test)
mae_metric = MeanAbsoluteError()
mae_metric.update_state(y_test, model.predict(X_test))
mae = mae_metric.result().numpy()

# Function to calculate MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Get Predictions
y_pred = model.predict(X_test)

# Compute MAPE
mape = mean_absolute_percentage_error(y_test.numpy(), y_pred)

# Print Metrics
print(f"Test MSE: {mse:.4f}")
print(f"Test MAE: {mae:.4f}")
print(f"Test MAPE: {mape:.2f}%")