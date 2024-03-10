import os 
import pandas as pd

data_dir = "./../../data/"
data = pd.read_parquet(os.path.join(data_dir,"finance_full.parquet"), engine='pyarrow')
data = data[data["input"] == ""]
data = data[data["instruction"] != ""]
data = data[data["output"] != ""]
data = data.drop("input", axis=1)
data = data[:1000]
data.to_json(os.path.join(data_dir,"finance_1000.json"), orient='records')

