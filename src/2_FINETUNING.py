import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
import gc
import os
import random

# Ορισμός όλων των παραμέτρων στην αρχή με περιγραφικά ονόματα
data_file_path = "/content/drive/MyDrive/Colab Notebooks/Good_For_Training_Only.csv"
weights_file_horizon_5 = "base_model_horizon_5.weights.h5"
weights_file_horizon_15 = "base_model_horizon_15.weights.h5"
sequence_length = 32  # Μήκος της χρονοσειράς εισόδου
prediction_horizons = [5, 15]  # Ορίζοντες πρόβλεψης
batch_size = 64  # Μέγεθος παρτίδας για εκπαίδευση
finetune_epochs = 10  # Αριθμός εποχών για fine-tuning
number_of_nodes_to_finetune = 10  # Αριθμός κόμβων για fine-tuning
lstm_units = 50  # Μονάδες LSTM στα επίπεδα
dropout_rate = 0.3  # Ποσοστό dropout για αποφυγή overfitting
learning_rate = 0.001  # Ρυθμός μάθησης του optimizer
early_stopping_patience = 2  # Υπομονή για early stopping
l2_regularization_strength = 0.001  # Ισχύς L2 regularization
random_seed = 42  # Σπόρος για αναπαραγωγιμότητα

# Επιλογή τυχαίων κόμβων
random.seed(random_seed)
df_ids = pd.read_csv(data_file_path, usecols=['nodeid'], dtype={'nodeid': int})
all_nodes = df_ids['nodeid'].unique()
filtered_nodes = [n for n in all_nodes if n > 1000]
selected_nodes = random.sample(filtered_nodes, number_of_nodes_to_finetune)
print(f"Επιλέχθηκαν οι κόμβοι: {selected_nodes}")

# Συνάρτηση δημιουργίας μοντέλου
def build_model(horizon):
    ts_input = Input(shape=(sequence_length, 1))
    x = LSTM(lstm_units, activation='relu', return_sequences=True,
             kernel_regularizer=tf.keras.regularizers.l2(l2_regularization_strength))(ts_input)
    x = Dropout(dropout_rate)(x)
    x = LSTM(lstm_units, activation='relu',
             kernel_regularizer=tf.keras.regularizers.l2(l2_regularization_strength))(x)
    x = Dropout(dropout_rate)(x)
    output = Dense(horizon)(x)  # Έξοδος απευθείας με το horizon
    model = Model(inputs=ts_input, outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
    return model

# Συνάρτηση δημιουργίας ακολουθιών
def create_sequences(data, seq_length, horizon):
    X_seq, y = [], []
    for i in range(len(data) - seq_length - horizon + 1):
        X_seq.append(data[i:i + seq_length])
        y.append(data[i + seq_length:i + seq_length + horizon, 0])
    return np.array(X_seq), np.array(y)

# Συνάρτηση υπολογισμού MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    y_true = np.where(y_true == 0, 1e-6, y_true)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Fine-tuning
node_errors = {horizon: {} for horizon in prediction_horizons}

for horizon in prediction_horizons:
    weights_file = weights_file_horizon_5 if horizon == 5 else weights_file_horizon_15
    if not os.path.exists(weights_file):
        print(f"Το αρχείο {weights_file} δεν βρέθηκε. Βεβαιώσου ότι έχεις εκπαιδεύσει το base model πρώτα.")
        continue
    
    print(f"\n=== Fine-Tuning για Horizon {horizon} ===")
    for node in selected_nodes:
        print(f"\nΕπεξεργασία Node {node}...")
        df_node = pd.read_csv(data_file_path)
        df_node = df_node[df_node['nodeid'] == node].copy()
        df_node.sort_values(by='timestamp', inplace=True)
        
        node_data = df_node[['node_cpu_usage']].values
        
        if len(node_data) < sequence_length + horizon:
            print(f"Node {node} δεν έχει αρκετά δεδομένα για horizon {horizon}")
            continue
        
        X_seq_node, y_node = create_sequences(node_data, sequence_length, horizon)
        
        if len(X_seq_node) == 0:
            print(f"Node {node} δεν έχει αρκετές ακολουθίες για horizon {horizon}")
            continue
        
        X_train_node, X_test_node, y_train_node, y_test_node = train_test_split(
            X_seq_node, y_node, test_size=0.2, random_state=42, shuffle=False
        )
        
        # Δημιουργία μοντέλου fine-tuning
        node_model = build_model(horizon)
        
        # Φόρτωση βαρών από base model
        node_model.load_weights(weights_file)
        print(f"Φορτώθηκαν τα βάρη από {weights_file} για horizon {horizon}")
        
        # Fine-tuning
        node_model.fit(
            X_train_node, y_train_node,
            epochs=finetune_epochs,
            batch_size=batch_size,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=early_stopping_patience, restore_best_weights=True)],
            verbose=0
        )
        
        # Υπολογισμός MAPE
        y_pred_node = node_model.predict(X_test_node)
        mape = mean_absolute_percentage_error(y_test_node, y_pred_node)
        node_errors[horizon][node] = mape
        print(f"Node {node}, Horizon {horizon}: MAPE = {mape:.2f}%")
        
        # Καθαρισμός μνήμης
        del X_seq_node, y_node, X_train_node, X_test_node, y_train_node, y_test_node, node_model
        tf.keras.backend.clear_session()
        gc.collect()
        print(f"Η μνήμη καθαρίστηκε για Node {node}, Horizon {horizon}")

# Εκτύπωση αποτελεσμάτων
for horizon in prediction_horizons:
    print(f"\n=== Τελικά MAPE για Horizon {horizon} ===")
    for node in selected_nodes:
        if node in node_errors[horizon]:
            print(f"Node {node}: MAPE = {node_errors[horizon][node]:.2f}%")
