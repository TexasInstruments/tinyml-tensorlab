import os
import requests as re
import zipfile
import pandas as pd
import shutil

# Project Definition
DATASET_NAME = "electrical_fault_6class"

# Dataset definition
DATASET_URL = "https://software-dl.ti.com/C2000/esd/mcu_ai/01_04_00/datasets/electrical_fault_raw.zip"
DATASET_FILE_NAME = "classData.csv"
independent_variables = ["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]
dependent_variables = ["G", "C", "B", "A"]  # Ground, Node C, Node B, Node A

# Dataset Processing
NUM_SPLITS = 5
class_names = [
    "no_fault",
    "bc_fault",
    "abc_fault",
    "ag_fault",
    "abg_fault",
    "abcg_fault",
]

def download_data():
    print("Downloading dataset")
    data = re.get(DATASET_URL)
    if data.status_code != 200:
        raise Exception(f"Failed to download dataset. HTTP status code: {data.status_code}")
    with open(f"{DATASET_NAME}.zip", "wb") as f:
        f.write(data.content)
    return None

def extract_data():
    print("Extracting dataset")
    with zipfile.ZipFile(f"{DATASET_NAME}.zip", 'r') as zip_ref:
        zip_ref.extractall(DATASET_NAME)
    return None

def load_data():
    data = pd.read_csv(f"{DATASET_NAME}/{DATASET_FILE_NAME}", index_col=False)
    columns = independent_variables + dependent_variables
    data = data[columns]
    # Map the four binary columns to a single target class
    def map_to_class(row):
        bits = (row['G'], row['C'], row['B'], row['A'])
        # Mapping: (G,C,B,A) -> class index
        if bits == (0,0,0,0):
            return 0  # NO_FAULT
        elif bits == (0,1,1,0):
            return 1  # BC_FAULT
        elif bits == (0,1,1,1):
            return 2  # ABC_FAULT
        elif bits == (1,0,0,1):
            return 3  # AG_FAULT
        elif bits == (1,0,1,1):
            return 4  # ABG_FAULT
        elif bits == (1,1,1,1):
            return 5  # ABCG_FAULT
        else:
            # Unexpected combination, assign -1 (will become "unknown")
            return -1
    data['Target'] = data.apply(map_to_class, axis=1)
    # Convert to string labels for folder naming
    data['Target'] = data['Target'].apply(lambda x: class_names[x] if 0 <= x < len(class_names) else "unknown")
    return data

def store_datafiles(df):
    if not os.path.exists("classes"):
        os.mkdir("classes")

    for idx, target in enumerate(class_names):
        if not os.path.exists(f"classes/class_{idx}_{target}"):
            os.mkdir(f"classes/class_{idx}_{target}")
        current_target_df = df[df['Target'] == target]
        num_splits = NUM_SPLITS
        chunk_size = len(current_target_df) // num_splits
        current_target_dfs = [current_target_df[i*chunk_size:(i+1)*chunk_size] for i in range(num_splits)]
        for df_idx, current_target_df in enumerate(current_target_dfs):
            current_target_df = current_target_df.drop(['Target'], axis=1)
            current_target_df.to_csv(f"classes/class_{idx}_{target}/{target}_{df_idx}.csv", index=False)
        print(f"Created classes/class_{idx}_{target}")

def cleanup():
    if os.path.exists(DATASET_NAME):
        shutil.rmtree(DATASET_NAME)
    zip_path = f"{DATASET_NAME}.zip"
    if os.path.exists(zip_path):
        os.remove(zip_path)

def zip_files():
    print(f"Zipping the classes into {DATASET_NAME}_dataset.zip")    
    shutil.make_archive(f"{DATASET_NAME}_dataset", 'zip', root_dir='.', base_dir='classes')
    shutil.rmtree('classes', ignore_errors=True)

if __name__ == '__main__':
    download_data()
    extract_data()
    df = load_data()
    store_datafiles(df)
    zip_files()
    cleanup()
