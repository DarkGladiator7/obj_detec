from pathlib import Path
import sys
import os
sys.path.append(os.path.abspath("D:/obj_detec/yolov9_main"))

from yolov9_main.utils.general import check_dataset
data_path = r"D:\obj_detec\YOLO_DATA\yolo_dataset.yaml"  # Update this with your actual data.yaml path
dataset_info = check_dataset(data_path)

print(dataset_info)  # Should display the processed dataset dictionary