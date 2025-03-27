import os
import json
import cv2
from tqdm import tqdm

# -------------- CONFIGURATION --------------
DATASET_DIR = r"D:\obj_detec\COCO_DATA\COCO_Train"  # Change to COCO_Test when needed
OUTPUT_DIR = r"D:\obj_detec\YOLO_Train"  # Where YOLO images & labels will be stored

# Automatically find the annotation file inside the dataset directory
annotation_file = None
for file in os.listdir(DATASET_DIR):
    if file.endswith(".json"):  # Look for a JSON annotation file
        annotation_file = os.path.join(DATASET_DIR, file)
        break

if annotation_file is None:
    raise FileNotFoundError("No COCO JSON annotation file found in the dataset directory!")

# Create output directories
YOLO_IMG_DIR = os.path.join(OUTPUT_DIR, "images")
YOLO_LABELS_DIR = os.path.join(OUTPUT_DIR, "labels")
os.makedirs(YOLO_IMG_DIR, exist_ok=True)
os.makedirs(YOLO_LABELS_DIR, exist_ok=True)

# Correct Class Order (Same as YOLO_Val & YOLO_Test)
CORRECT_CLASS_LIST = [
    "dog", "person", "train", "sofa", "chair", "car", "pottedplant", "diningtable",
    "horse", "cat", "cow", "bus", "bicycle", "motorbike", "bird", "tvmonitor",
    "sheep", "aeroplane", "boat", "bottle"
]

# Create class name to ID mapping based on correct order
class_name_to_id = {name: idx for idx, name in enumerate(CORRECT_CLASS_LIST)}


def convert_coco_to_yolo(coco_json, image_dir, output_dir):
    """ Convert COCO JSON annotations to YOLO format """

    # Load COCO JSON
    with open(coco_json, "r") as f:
        data = json.load(f)

    images = {img["id"]: img["file_name"] for img in data["images"]}
    categories = {cat["id"]: cat["name"] for cat in data["categories"]}

    # Ensure all dataset classes are in the correct order
    with open(os.path.join(output_dir, "classes.txt"), "w") as f:
        for name in CORRECT_CLASS_LIST:
            f.write(f"{name}\n")

    # Process annotations
    for ann in tqdm(data["annotations"], desc="Converting Annotations"):
        img_id = ann["image_id"]
        img_name = images[img_id]
        img_path = os.path.join(image_dir, img_name)

        # Copy image to YOLO images folder
        if os.path.exists(img_path):
            cv2.imwrite(os.path.join(YOLO_IMG_DIR, img_name), cv2.imread(img_path))

        # Read image dimensions
        img_data = cv2.imread(img_path)
        if img_data is None:
            continue
        img_h, img_w, _ = img_data.shape

        # Convert COCO bbox (x, y, width, height) → YOLO (x_center, y_center, width, height) normalized
        x, y, w, h = ann["bbox"]
        x_center, y_center = (x + w / 2) / img_w, (y + h / 2) / img_h
        w, h = w / img_w, h / img_h

        # Get correct class ID
        coco_class_name = categories[ann["category_id"]]
        if coco_class_name not in class_name_to_id:
            continue  # Skip if class is not in the correct mapping

        correct_class_id = class_name_to_id[coco_class_name]

        # Write YOLO annotation file
        label_path = os.path.join(YOLO_LABELS_DIR, img_name.replace(".jpg", ".txt"))
        with open(label_path, "a") as f:
            f.write(f"{correct_class_id} {x_center} {y_center} {w} {h}\n")

    print(f"✅ COCO annotations converted to YOLO format in {output_dir}")


# -------------- RUN CONVERSION --------------
if __name__ == "__main__":
    convert_coco_to_yolo(annotation_file, DATASET_DIR, OUTPUT_DIR)

