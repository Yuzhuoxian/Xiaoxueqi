# scripts/utility.py
import os
import json
import pandas as pd

def dict_to_json(data, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as json_file:
        json.dump(data, json_file, indent=4)
    return

def dict_to_table(data, filename):
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename, index=False)
    return