import os
import pandas as pd
import json

def init(filename):
    data_folder = os.path.join(os.getcwd(), 'data')  # path to data folder as a string
    os.makedirs(data_folder, exist_ok=True)
    path = os.path.join(data_folder, filename)
    return path

# Read

def read_csv(filename: str) -> pd.DataFrame:

    path = init(filename)

    df = pd.read_csv(path)
    try:
        df['Date'] = pd.to_datetime(df['Date'])  
        df.set_index('Date', inplace=True)
    except Exception as e:
        print(f'Something happened while reading csv: {e}')
    return df

def read_json(filename):

    path = init(filename)

    with open(path, "r") as file:
        data = json.load(file)
    return data

# Write (save)

def save_csv(df: pd.DataFrame, filename: str) -> str:

    path = init(filename)

    df.to_csv(
    path_or_buf=path
    )

    return path

def save_json(data, filename: str):

    path = init(filename)

    with open(path, "w") as file:
        json.dump(data, file)