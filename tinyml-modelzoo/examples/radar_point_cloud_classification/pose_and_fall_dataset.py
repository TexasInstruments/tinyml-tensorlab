"""
This script is intented to be run as an optional preprocessing step for data filtering
and organization for the radar pose and fall model. This code may be needed only if 
the user is not using the publically available dataset at: 
https://software-dl.ti.com/C2000/esd/mcu_ai/01_04_00/datasets/radar_human_pose_detection.zip

"""

import os
from pathlib import Path
import shutil
import torch
import math
import pandas as pd
import numpy as np
import random

MIN_POINTS = 5  # Minimum number of point cloud points for a row to be considered
# Data Filtering
FILTER = True
MAX_HEIGHT = 3
MIN_HEIGHT = -4
MAX_DISTANCE = 5
MIN_DISTANCE = -4
LOWEST_POINTS_INCLUDED = False
TRAIN_SPLIT = 0.8
DATASET_DIR = "Path/to/local/pose_and_fall_detection/dataset"


# Setup path to data folder
data_path = Path(os.getcwd() + "/dataset")
print(data_path)
class_path = Path(f"{data_path}/classes/")

# Find the class names (according to folder structure)
class_names = [f for f in os.listdir(class_path)]
file_names = [f for f in list(class_path.rglob("*.csv"))]
#print(f"File names: {file_names}")
print(f"# of Files: {len(file_names)}")

class_output_path = data_path / "classes_cleaned"
annotations_path = data_path / "annotations"

# Enumeration for each Category, \/\/\/\/\/ add new categories here along with data CSV's in folder of same name 
class_data = {0: 'STANDING', 1: 'SITTING', 2: 'LYING', 3: 'FALLING', 4: 'WALKING'}

def ensure_dirs():
    """Create output directories."""
    class_output_path.mkdir(parents=True, exist_ok=True)
    annotations_path.mkdir(parents=True, exist_ok=True)
    for class_name in class_data.values():
       (class_output_path / class_name.lower()).mkdir(parents=True, exist_ok=True)

def setup_real_dataset(dataset_dir: str) -> str:
    """
    Scan dataset_dir for class subfolders and CSV files, then write
    instances_train_list.txt and instances_test_list.txt into the
    sibling annotations/ folder (creating it if needed).

    Returns dataset_dir unchanged (callers use it as-is).
    """
    annot_dir = os.path.join(os.path.dirname(dataset_dir), "annotations")
    os.makedirs(annot_dir, exist_ok=True)

    train_lines, test_lines = [], []
    rng = random.Random(42)

    for cls in sorted(os.listdir(dataset_dir)):
        cls_path = os.path.join(dataset_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        files = sorted(f for f in os.listdir(cls_path) if f.endswith(".csv"))
        rng.shuffle(files)
        split = max(1, int(len(files) * TRAIN_SPLIT))
        for f in files[:split]:
            train_lines.append(f"{cls}/{f}")
        for f in files[split:]:
            test_lines.append(f"{cls}/{f}")

    with open(os.path.join(annot_dir, "instances_train_list.txt"), "w") as f:
        f.write("\n".join(train_lines))
    with open(os.path.join(annot_dir, "instances_test_list.txt"), "w") as f:
        f.write("\n".join(test_lines))

    print(f"    annotation dir : {annot_dir}")
    print(f"    train files    : {len(train_lines)}")
    print(f"    test files     : {len(test_lines)}")
    return dataset_dir

def main():

    df_data = {}
    #annotations = []

    """
    Read in the data from the CSV files and concatenate into single dataframe
    """

    ## Read in the CSV files and concatenate into one large dataset. 
    ## Remove collection information columns and add classification column
    ## Remove any rows with invalid data (not enough points or invalid track ID)
    df_csv_append = pd.DataFrame()
    for file in file_names:
        df = pd.read_csv(file, encoding='utf-8', engine='python')

        # Add tracker columns to datasets that may not have recorded it
        required = [
            "posx", "posy", "posz",
            "velx", "vely", "velz",
            "accx", "accy", "accz"
        ]

        # Add any missing ones with a default of 0
        for col in required:
            if col not in df.columns:
                df[col] = 0   # pandas automatically broadcasts the scalar to all rows
        classification = os.path.basename(os.path.dirname(file)).upper()
        value = -1
        for key, val in class_data.items():
            if val == classification:
                value = key
        if 'Session_ID' in df.columns:
            df.drop(['Session_ID', 'Recording_Number'], axis=1, inplace=True)
        df.insert(0, 'classification', value)
        print(f'{file.name}: {df.shape}')
        dropped = 0
        for index, row in df.iterrows():
                # First check if either column exists in the DataFrame
            if 'Track ID' in df.columns:
                track_id_col = 'Track ID'
            elif 'tid' in df.columns:
                track_id_col = 'tid'
            else:
                # Handle the case where neither column exists
                print(f"Warning: Neither 'Track ID' nor 'tid' found in file {file.name}")
                df.drop(index, inplace=True)
                dropped += 1
                continue
                
            # Now check the condition using the identified column
            if (row.notna().sum() < (9 + 3 * MIN_POINTS)) or not str(float(row[track_id_col])).endswith('.0'):
                df.drop(index, inplace=True)
                dropped += 1

        print(f"Dropped {dropped} rows for invalid data")

        #df_csv_append = pd.concat([df_csv_append, df], ignore_index=True)

        # sort the rows by classification column. 
    
        # Format of final training dataFrame (0,1,2 : highest, 3,4,5 : Lowest) 
        # Number of high/low points dependant on min number of points available
        static_columns = ['classification', 'posz', 'velx', 'vely', 'velz', 'accx', 'accy', 'accz']
        dynamic_columns = [f"{var}{i}" for i in range(MIN_POINTS) for var in ['y', 'z', 'snr']]
        output_columns = static_columns + dynamic_columns
        df_sorting_append = pd.DataFrame(columns=output_columns)

        # Extract lowest and highest points from the point cloud data
        # Form new dataframe using only these points and the target info
        filtered = 0
        for index, row in df.iterrows():
            points = []
            for col in df.columns:
                if col.startswith('pointy') and not np.isnan(row[col]) and not np.isnan(
                        row[col.replace('pointy', 'pointz')]) and not np.isnan(row[col.replace('pointy', 'snr')]):
                    points.append([row[col] - row['posy'], row[col.replace('pointy', 'pointz')],
                                row[col.replace('pointy', 'snr')]])

            df_points = pd.DataFrame(points, columns=['pointy', 'pointz', 'snr'])
            df_points = df_points.sort_values(by='pointz')

            if FILTER:
                df_points = df_points[
                    (df_points['pointy'] <= MAX_DISTANCE) &
                    (df_points['pointy'] >= MIN_DISTANCE) &
                    (df_points['pointz'] <= MAX_HEIGHT) &
                    (df_points['pointz'] >= MIN_HEIGHT)
                    ]
                if (df_points.shape[0] < MIN_POINTS):
                    filtered += 1
                    continue
            
            if LOWEST_POINTS_INCLUDED:
                # Use half points from top and half from bottom
                highestPoints = df_points.tail(math.floor(MIN_POINTS / 2))
                lowestPoints = df_points.head(math.floor(MIN_POINTS / 2))
                
                # Create a copy and populate it
                df_copy = df_sorting_append.iloc[:0, :].copy()
                df_copy.loc[0] = [
                    row['classification'],
                    row['posz'],
                    row['velx'],
                    row['vely'],
                    row['velz'],
                    row['accx'],
                    row['accy'],
                    row['accz'],
                    *highestPoints.values.flatten(),
                    *lowestPoints.values.flatten()
                ]
            else:
                # Use all MIN_POINTS from top points
                allHighPoints = df_points.tail(MIN_POINTS)
                
                # Create a copy and populate it
                df_copy = df_sorting_append.iloc[:0, :].copy()
                df_copy.loc[0] = [
                    row['classification'],
                    row['posz'],
                    row['velx'],
                    row['vely'],
                    row['velz'],
                    row['accx'],
                    row['accy'],
                    row['accz'],
                    *allHighPoints.values.flatten()
                ]
            
            df_sorting_append = pd.concat([df_sorting_append, df_copy], ignore_index=True)
            #print(f"I got to here for file {file} ")
            
            

        if (FILTER):
            df_sorting_append = df_sorting_append[
                (df_sorting_append['posz'] <= MAX_HEIGHT) &
                (df_sorting_append['posz'] >= MIN_HEIGHT)]
            print(f'Filtered Out {filtered} Items')

        if classification not in df_data:
            df_data[classification]= []
            df_data[classification].append(df_sorting_append)
            #df_data = df_sorting_append.sort_values(by='classification')
        # Save individual CSV
        output_file = class_output_path / classification.lower() / file.name
        df_sorting_append.iloc[:, 1:].to_csv(output_file, index=False, header=False)
        # annotations.append(str(output_file.relative_to(data_path)))

if __name__ == "__main__":
    main()
    setup_real_dataset(str(class_output_path))
