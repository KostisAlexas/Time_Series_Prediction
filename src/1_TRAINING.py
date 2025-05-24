import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import gc
import os

# Ορισμός όλων των παραμέτρων στην αρχή με περιγραφικά ονόματα
data_file_path = "/content/drive/MyDrive/Colab Notebooks/Good_For_Training_Only.csv"
weights_file_horizon_5 = "base_model_horizon_5.weights.h5"
weights_file_horizon_15 = "base_model_horizon_15.weights.h5"
sequence_length = 32  # Μήκος της χρονοσειράς εισόδου
prediction_horizons = [5, 15]  # Ορίζοντες πρόβλεψης
batch_size = 64  # Μέγεθος παρτίδας για εκπαίδευση
training_epochs = 5  # Αριθμός εποχών εκπαίδευσης
node_group_size = 50  # Μέγεθος ομάδας κόμβων για διαχείριση μνήμης
total_nodes_to_process = 200  # Συνολικός αριθμός κόμβων για εκπαίδευση
lstm_units = 50  # Μονάδες LSTM στα επίπεδα
dropout_rate = 0.3  # Ποσοστό dropout για αποφυγή overfitting
learning_rate = 0.001  # Ρυθμός μάθησης του optimizer
early_stopping_patience = 2  # Υπομονή για early stopping
l2_regularization_strength = 0.001  # Ισχύς L2 regularization

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

# Φόρτωση δεδομένων
print("Φόρτωση δεδομένων...")
df = pd.read_csv(data_file_path)
df.sort_values(by=['nodeid', 'timestamp'], inplace=True)
all_nodes = df['nodeid'].unique()[:total_nodes_to_process]

# Ομαδοποίηση κόμβων
node_groups = [all_nodes[i:i + node_group_size] for i in range(0, len(all_nodes), node_group_size)]

# Εκπαίδευση base model για κάθε horizon
for horizon in prediction_horizons:
    print(f"\n=== Εκπαίδευση Base Model για Horizon {horizon} ===")
    weights_file = weights_file_horizon_5 if horizon == 5 else weights_file_horizon_15
    model = None
    early_stopping = EarlyStopping(monitor='val_loss', patience=early_stopping_patience, restore_best_weights=True)
    
    for group_idx, node_group in enumerate(node_groups):
        print(f"\nΕπεξεργασία ομάδας {group_idx + 1}/{len(node_groups)}")
        all_X_seq, all_y = [], []
        for node in node_group:
            df_node = df[df['nodeid'] == node]
            node_data = df_node[['node_cpu_usage']].values
            if len(node_data) >= sequence_length + horizon:
                X_seq, y = create_sequences(node_data, sequence_length, horizon)
                all_X_seq.append(X_seq)
                all_y.append(y)
        
        if not all_X_seq:
            print(f"Δεν υπάρχουν αρκετά δεδομένα για την ομάδα {group_idx + 1}")
            continue
        
        all_X_seq = np.vstack(all_X_seq)
        all_y = np.vstack(all_y)
        print(f"Δημιουργήθηκαν {len(all_X_seq)} ακολουθίες για την ομάδα {group_idx + 1}")
        
        X_train, X_test, y_train, y_test = train_test_split(all_X_seq, all_y, test_size=0.2, random_state=42)
        
        if model is None:
            model = build_model(horizon)
            if os.path.exists(weights_file):
                model.load_weights(weights_file)
                print(f"Φορτώθηκαν υπάρχοντα βάρη από {weights_file}")
        else:
            model.load_weights(weights_file)
            print(f"Φορτώθηκαν βάρη από {weights_file} για συνέχιση εκπαίδευσης")
        
        print(f"Ξεκινάει η εκπαίδευση για horizon {horizon}, ομάδα {group_idx + 1}...")
        history = model.fit(X_train, y_train,
                            epochs=training_epochs,
                            batch_size=batch_size,
                            validation_data=(X_test, y_test),
                            callbacks=[early_stopping],
                            verbose=1)
        
        print(f"Ολοκληρώθηκε η εκπαίδευση για ομάδα {group_idx + 1}. Τελευταία val_loss: {history.history['val_loss'][-1]:.4f}")
        model.save_weights(weights_file)
        print(f"Αποθηκεύτηκαν τα βάρη στο αρχείο: {weights_file}")
        
        del all_X_seq, all_y, X_train, X_test, y_train, y_test
        tf.keras.backend.clear_session()
        gc.collect()
        print(f"Η μνήμη καθαρίστηκε μετά την επεξεργασία της ομάδας {group_idx + 1}")
    
    print(f"\nΟλοκληρώθηκε η εκπαίδευση για horizon {horizon}")
