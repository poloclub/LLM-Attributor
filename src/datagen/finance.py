import os 
import pandas as pd
import re 

data_dir = "./../../data/"
data = pd.read_parquet(os.path.join(data_dir,"finance_full.parquet"), engine='pyarrow')
data = data[data["input"] == ""]
data = data[data["instruction"] != ""]
data = data[data["output"] != ""]
data = data.drop("input", axis=1)

new_data = []
i = -1
while len(new_data) < 1000:
    i += 1
    if len(data.iloc[i]["output"].split(" ")) < 10: continue
    
    instruction = data.iloc[i]["instruction"]
    instruction = re.sub(' +', ' ', instruction)
    output = data.iloc[i]["output"]
    output = re.sub(' +', ' ', output)
    data_dict = {"instruction": instruction, "output": output}
    new_data.append(data_dict)
new_data = pd.DataFrame(new_data)
# data = data[:1000]
new_data.to_json(os.path.join(data_dir,"finance_1000_mar13.json"), orient='records')

