import os
import sys

sys.path.append(os.path.abspath("D:/obj_detec/"))

from yolov9_main.train_dual import run

# Define paths
YOLOV9_DIR = r"D:\obj_detec\yolov9_main"
DATASET_YAML =r"D:\obj_detec\YOLO_DATA\yolo_dataset.yaml"
WEIGHTS = r"D:\obj_detec\yolov9_main\weights\yolov9-c.pt"  # Ensure this is in the YOLOv9 repo

# Navigate to YOLOv9 directory
os.chdir(YOLOV9_DIR)

# Run train_dual.py with required arguments
#os.system(f"python train_dual.py --data {DATASET_YAML} --weights {WEIGHTS} --epochs 2 --batch-size 16 --img 640")


run(data=r"D:\obj_detec\YOLO_DATA\yolo_dataset.yaml", 
                     weights="D:/obj_detec/yolov9_main/weights/yolov9-c.pt",
                     epochs=2, 
                     batch_size=16, 
                     imgsz=640,
                     known=True)


