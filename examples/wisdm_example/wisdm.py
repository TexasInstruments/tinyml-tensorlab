import os
import requests as re

import pandas as pd

url = "https://www.cis.fordham.edu/wisdm/includes/datasets/latest/WISDM_ar_latest.tar.gz"

DOWNLOADED_ZIP_FILE_NAME = "WISDM_ar_latest.tar.gz"
EXTRACTED_FOLDER_NAME = "WISDM_ar_v1.1"
RAW_DATA_FILE_NAME = "WISDM_ar_v1.1_raw.txt"
NUM_SPLITS = 4

FILE_NAME = "wisdm_dataset.csv"
CREATED_ZIP_FILE_NAME = "wisdm_dataset.zip"

def download_data():
    print("Downloading dataset")
    data = re.get(url)
    with open(DOWNLOADED_ZIP_FILE_NAME, "wb") as f:
        f.write(data.content)
    return None

def extract_data():
    print("Extracting dataset")
    os.system(f"tar -xvzf {DOWNLOADED_ZIP_FILE_NAME} > /dev/null 2>&1")
    files = os.listdir(EXTRACTED_FOLDER_NAME)
    for file in files:
        if file != RAW_DATA_FILE_NAME:
            os.remove(f"{EXTRACTED_FOLDER_NAME}/{file}" )
    os.rename(f"{EXTRACTED_FOLDER_NAME}/{RAW_DATA_FILE_NAME}", f"{EXTRACTED_FOLDER_NAME}/{FILE_NAME}")
    return None
    
def clean_data(df):
    df = df[df['user'] == 8]
    df = df.drop(['user'], axis=1)
    df = df[['timestamp', 'x', 'y', 'z', 'activity']]
    rename_columns = {'x': 'Acc_X', 'y': 'Acc_Y', 'z': 'Acc_Z', 'activity': 'Target', 'timestamp': 'Time'}
    df.rename(columns=rename_columns, inplace=True)
    return df

def load_data():
    columns = ["user", "activity", "timestamp", "x", "y", "z"]
    file = open(f"{EXTRACTED_FOLDER_NAME}/{FILE_NAME}", "r")
    data = file.read()
    file.close()
    data = data.replace(";\n", "\n").replace(";", "\n").replace("\n\n", "\n").replace(",\n", "\n")
    file = open(f"{EXTRACTED_FOLDER_NAME}/{FILE_NAME}", "w")
    file.write(data)
    file.close()
    data = pd.read_csv(f"{EXTRACTED_FOLDER_NAME}/{FILE_NAME}", header=None, names=columns)
    data = clean_data(data)
    return data


def store_datafiles(df):
    unique_targets = df['Target'].unique()
    if not os.path.exists("classes"):
        os.mkdir("classes")

    for idx, target in enumerate(unique_targets):
        if not os.path.exists(f"classes/class_{idx}_{target.lower()}"):
            os.mkdir(f"classes/class_{idx}_{target.lower()}")
        current_target_df = df[df['Target'] == target]
        num_splits = NUM_SPLITS
        chunk_size = len(current_target_df) // num_splits
        current_target_dfs = [current_target_df[i*chunk_size:(i+1)*chunk_size] for i in range(num_splits)]
        for df_idx, current_target_df in enumerate(current_target_dfs):
            current_target_df = current_target_df.drop(['Target'], axis=1)
            current_target_df.to_csv(f"classes/class_{idx}_{target.lower()}/{target.lower()}_{df_idx}.csv", index=False)
        print("Created classes/class_{}_{}".format(idx, target.lower()))

def cleanup():
    os.system(f"rm -rf {EXTRACTED_FOLDER_NAME} > /dev/null 2>&1")
    os.system(f"rm {DOWNLOADED_ZIP_FILE_NAME} > /dev/null 2>&1")

def zip_files():
    print(f"Zipping the classes into {CREATED_ZIP_FILE_NAME}")
    os.system(f"zip -r {CREATED_ZIP_FILE_NAME} classes > /dev/null 2>&1")
    os.system("rm -rf classes > /dev/null 2>&1")

if __name__ == '__main__':

    download_data()
    extract_data()

    df = load_data()

    store_datafiles(df)
    zip_files()

    cleanup()
