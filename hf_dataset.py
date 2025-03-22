from datasets import *
import pandas as pd
from torch.utils.data import DataLoader

dataset = load_dataset("lerobot/droid_100", split="train")
dataset.set_format(type="torch", columns=["episode_index", "action"])

# Step 2: Convert the dataset to a pandas DataFrame.
df = dataset.to_pandas()

# Assume your DataFrame has columns: 'ts_id', 'timestamp', 'value', etc.

# Step 3: Group the data by the time series id. Here we aggregate all values in each column into lists.
# You can customize the aggregation function for each column if needed.
grouped_df = df.groupby("episode_index").agg(lambda x: list(x)).reset_index()

# The resulting DataFrame now has one row per time series,
# with columns (other than ts_id) being lists of values from the original rows.

# Step 4: Convert the grouped DataFrame back into a Hugging Face Dataset.
grouped_dataset = Dataset.from_pandas(grouped_df)

# (Optional) Verify the transformation by looking at a sample
print(len(grouped_dataset))
