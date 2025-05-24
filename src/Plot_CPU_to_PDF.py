import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# File paths
csv_file = "/content/drive/MyDrive/Colab Notebooks/Good_For_Training_Only.csv"
output_pdf = "cpu_usage_plots.pdf"

# Φόρτωση δεδομένων
df = pd.read_csv(csv_file)
df.sort_values(by=['nodeid', 'timestamp'], inplace=True)

# Δημιουργία PDF με ένα γράφημα ανά σελίδα
with PdfPages(output_pdf) as pdf:
    for node, group in df.groupby('nodeid'):
        plt.figure()
        plt.plot(group['timestamp'], group['node_cpu_usage'] * 100)
        plt.title(f"Node {node} CPU Usage Over Time")
        plt.xlabel("Timestamp")
        plt.ylabel("CPU Usage (%)")
        plt.tight_layout()
        pdf.savefig()
        plt.close()

