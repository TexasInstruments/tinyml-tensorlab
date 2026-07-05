import math
from typing import Dict, List, Any
import os
import pandas as pd
import glob
import pickle
import numpy as np
from pathlib import Path

class StatisticalAnalysisEngine:
    def __init__(self, data_path, task_family):
        self.data_path = data_path
        self.task_family = task_family
        self.supported_formats = {'.csv', '.txt', '.npy', '.pkl'}

    @staticmethod
    def _load_data_file(file_path: str) -> pd.DataFrame:
        """Load a data file in any supported format to DataFrame."""
        ext = Path(file_path).suffix.lower()

        if ext == '.csv':
            return pd.read_csv(file_path)
        elif ext == '.txt':
            return pd.read_csv(file_path, sep=r'\s+|,|\t', engine='python')
        elif ext == '.npy':
            arr = np.load(file_path)
            return pd.DataFrame(arr)
        elif ext == '.pkl':
            with open(file_path, 'rb') as f:
                obj = pickle.load(f)
                if isinstance(obj, pd.DataFrame):
                    return obj
                elif isinstance(obj, np.ndarray):
                    return pd.DataFrame(obj)
                else:
                    raise TypeError(f"Pickled object is {type(obj)}, expected DataFrame or ndarray")
        else:
            raise ValueError(f"Unsupported format: {ext}. Supported: {StatisticalAnalysisEngine.supported_formats}")

    @staticmethod
    def _find_data_files(directory: str, extensions: set = None) -> List[str]:
        """Find all data files in directory, optionally filtered by extension."""
        if extensions is None:
            extensions = {'.csv', '.txt', '.npy', '.pkl'}

        files = []
        for ext in extensions:
            files.extend(glob.glob(os.path.join(directory, f"*{ext}")))
        return sorted(files)

    def bin_dataset(self, dataset_size):
        if dataset_size < 500:
            return "tiny"
        elif dataset_size >= 500 and dataset_size < 5000:
            return "small"
        elif dataset_size >= 5000 and dataset_size < 50000:
            return "medium"
        else:
            return "large"

    def CLS_AD_STATS(self):
        global_min = float('inf')  
        total_size = 0
        class_wise_dist = []                                         
        classes_path = os.path.join(self.data_path, "classes")
                                                                                
        for cls_name in os.listdir(classes_path):
            cls_size = 0
            cls_path = os.path.join(classes_path, cls_name)                      
            if not os.path.isdir(cls_path):                                      
                continue                                                         
                                                                        
            data_files = self._find_data_files(cls_path)
            for data_file in data_files:
                try:
                    df = self._load_data_file(data_file)
                    global_min = min(global_min, df.shape[0])
                    total_size += df.shape[0]
                    cls_size += df.shape[0]
                except Exception as e:
                    print(f"Warning: Could not load {data_file}: {e}")
                    continue

            class_wise_dist.append(cls_size)    

        dataset_bucket = self.bin_dataset(total_size)
                                                                                
        return global_min if global_min != float('inf') else None, total_size, class_wise_dist, dataset_bucket

    def REGR_FORECAST_STATS(self):
        global_min = float('inf')
        total_size = 0                                                                                                                           
        files_path = os.path.join(self.data_path, "files")
        if not os.path.isdir(files_path):
            return
                   
        data_files = self._find_data_files(files_path)
        for data_file in data_files:
            try:
                df = self._load_data_file(data_file)
                global_min = min(global_min, df.shape[0])
                total_size += df.shape[0]
            except Exception as e:
                print(f"Warning: Could not load {data_file}: {e}")
                continue

        dataset_bucket = self.bin_dataset(total_size)
                                                                    
        return global_min if global_min != float('inf') else None, total_size, dataset_bucket
        

def analyse_dataset(formatted_dataset_path: str, task_family: str) -> Dict[str, Any]:
    statsEngine = StatisticalAnalysisEngine(formatted_dataset_path, task_family)    
    if task_family == "classification" or task_family == "anomalydetection":
        min_sample_size, total_size, cls_wise_dist, dataset_bucket = statsEngine.CLS_AD_STATS()
        return {
            "min_sample_length": min_sample_size,
            "total_dataset_size": total_size,
            "class_wise_distribution": cls_wise_dist,
            "dataset_bucket": dataset_bucket
        }

    min_seq_length, total_size, dataset_bucket = statsEngine.REGR_FORECAST_STATS()
    return {
        "min_seq_length": min_seq_length,
        "total_dataset_size": total_size,
        "dataset_bucket": dataset_bucket
    }