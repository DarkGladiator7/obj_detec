import torch
from ultralytics import YOLO

# Load YOLOv9 model
model = YOLO("yolov8n.pt")# Change if using a different weight file

# Train YOLOv9 on your dataset
model.train(
    data=r"D:\obj_detec\YOLO_DATA\yolo_dataset.yaml",  # Path to dataset YAML
    epochs=5,  # Adjust based on your needs
    batch=16,
    imgsz=640,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Evaluate on Test Dataset
metrics = model.val()
print("Validation Metrics:", metrics)

# Run Inference on a Sample Image
results = model.predict(r"D:\obj_detec\YOLO_DATA\YOLO_Val\images\000001.jpg", save=True)
print("Inference Completed!")
