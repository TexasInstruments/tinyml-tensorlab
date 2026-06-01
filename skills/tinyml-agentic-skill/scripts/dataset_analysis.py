import math
from typing import Dict, List, Any
import os
import pandas as pd
import glob

class StatisticalAnalysisEngine:
    def __init__(self, data_path, task_family):
        self.data_path = data_path
        self.task_family = task_family

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
                                                                        
            # Find all CSVs in sample folder                                 
            csv_files = glob.glob(os.path.join(cls_path, "*.csv"))
            for csv_file in csv_files:                                       
                df = pd.read_csv(csv_file)
                global_min = min(global_min, df.shape[0])
                total_size += df.shape[0]
                cls_size += df.shape[0]        

            class_wise_dist.append(cls_size)    

        dataset_bucket = self.bin_dataset(total_size)
                                                                                
        return global_min if global_min != float('inf') else None, total_size, class_wise_dist, dataset_bucket

    def REGR_FORECAST_STATS(self):
        global_min = float('inf')
        total_size = 0                                                                                                                           
        files_path = os.path.join(self.data_path, "files")
        if not os.path.isdir(files_path):
            return
                   
        csv_files = glob.glob(os.path.join(files_path, "*.csv"))
        for csv_file in csv_files:                                       
            df = pd.read_csv(csv_file)
            global_min = min(global_min, df.shape[0])
            total_size += df.shape[0]

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