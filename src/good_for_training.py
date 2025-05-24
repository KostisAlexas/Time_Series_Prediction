# Βρίσκει κόμβους με χαμηλή διακύμανση και βάζει τα id τους σε ενα txt. Η διακύμανση αξιολογείται με ενα τύπο που μετα ελεγχεται απο ενα threshold που ρυθμίζει ο χρήστης.

import pandas as pd
import numpy as np

csv_file = "/content/drive/MyDrive/Colab_Notebooks/cleaned_data.csv"
output_file = "good_for_training.txt"
epsilon = 0.05
quantile_level = 0.25

df = pd.read_csv(csv_file)
df.sort_values(by=['nodeid', 'timestamp'], inplace=True)

metrics = []

for node, group in df.groupby('nodeid'):
    vals = group['node_cpu_usage'].values
    diffs = np.abs(np.diff(vals))
    abruptness = np.quantile(diffs, 0.95)
    magnitude = vals.max() - vals.min()
    freq = np.sum(diffs > epsilon) / len(diffs) if len(diffs) > 0 else 0
    metrics.append((node, abruptness, magnitude, freq))

metrics_df = pd.DataFrame(metrics, columns=['nodeid','abruptness','magnitude','freq']).set_index('nodeid')

thresholds = metrics_df.quantile(quantile_level)

good_nodes = metrics_df[
    (metrics_df['abruptness'] < thresholds['abruptness']) &
    (metrics_df['magnitude']  < thresholds['magnitude'])  &
    (metrics_df['freq']       < thresholds['freq'])
].index.tolist()

with open(output_file, 'w') as f:
    for node in good_nodes:
        f.write(f"{node}\n")

