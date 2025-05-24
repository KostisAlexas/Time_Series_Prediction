import pandas as pd

csv_file = "/content/drive/MyDrive/Colab_Notebooks/cleaned_data.csv"
good_nodes_txt = "good_for_training.txt"
output_csv = "Good_For_Training_Only.csv"

with open(good_nodes_txt, 'r') as f:
    good_nodes = set(line.strip() for line in f)

df = pd.read_csv(csv_file)
filtered_df = df[df['nodeid'].astype(str).isin(good_nodes)].copy()

unique_ids = filtered_df['nodeid'].astype(str).unique()
id_map = {old_id: str(new_id) for new_id, old_id in enumerate(unique_ids, start=1)}

filtered_df['nodeid'] = filtered_df['nodeid'].astype(str).map(id_map)
filtered_df.to_csv(output_csv, index=False)

