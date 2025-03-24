import os
import json
import cv2
import random
from tqdm import tqdm

# -------------- CONFIGURATION --------------
DATASET_DIR = r"D:\obj_detec\COCO_Test"  # Path to COCO_Test dataset
OUTPUT_DIR = r"D:\obj_detec\coco"  # Output directory for YOLO format
VAL_SPLIT = 0.2  # 20% Validation, 80% Test

# Automatically find COCO annotation file inside dataset directory
annotation_file = None
for file in os.listdir(DATASET_DIR):
    if file.endswith(".json"):  # Look for JSON annotation file
        annotation_file = os.path.join(DATASET_DIR, file)
        break

if annotation_file is None:
    raise FileNotFoundError("No COCO JSON annotation file found in the dataset directory!")

# Create YOLO output directories
YOLO_TEST_IMG_DIR = os.path.join(OUTPUT_DIR, "YOLO_Test/images")
YOLO_TEST_LABELS_DIR = os.path.join(OUTPUT_DIR, "YOLO_Test/labels")
YOLO_VAL_IMG_DIR = os.path.join(OUTPUT_DIR, "YOLO_Val/images")
YOLO_VAL_LABELS_DIR = os.path.join(OUTPUT_DIR, "YOLO_Val/labels")

os.makedirs(YOLO_TEST_IMG_DIR, exist_ok=True)
os.makedirs(YOLO_TEST_LABELS_DIR, exist_ok=True)
os.makedirs(YOLO_VAL_IMG_DIR, exist_ok=True)
os.makedirs(YOLO_VAL_LABELS_DIR, exist_ok=True)


def convert_coco_to_yolo(coco_json, image_dir, output_dir, val_split):
    """ Convert COCO JSON annotations to YOLO format and split into Test & Validation sets """

    # Load COCO JSON
    with open(coco_json, "r") as f:
        data = json.load(f)

    images = {img["id"]: img["file_name"] for img in data["images"]}
    categories = {cat["id"]: cat["name"] for cat in data["categories"]}

    # Create a classes.txt file (mapping class names to IDs)
    class_list = list(categories.values())
    with open(os.path.join(output_dir, "classes.txt"), "w") as f:
        for name in class_list:
            f.write(f"{name}\n")

    # Shuffle image IDs for random split
    image_ids = list(images.keys())
    random.shuffle(image_ids)
    val_size = int(len(image_ids) * val_split)
    val_ids = set(image_ids[:val_size])  # 20% for validation
    test_ids = set(image_ids[val_size:])  # 80% for testing

    # Process annotations
    for ann in tqdm(data["annotations"], desc="Converting Annotations"):
        img_id = ann["image_id"]
        img_name = images[img_id]
        img_path = os.path.join(image_dir, img_name)

        # Determine the target folder (Test or Validation)
        if img_id in val_ids:
            img_output_dir, label_output_dir = YOLO_VAL_IMG_DIR, YOLO_VAL_LABELS_DIR
        else:
            img_output_dir, label_output_dir = YOLO_TEST_IMG_DIR, YOLO_TEST_LABELS_DIR

        # Copy image to respective folder
        if os.path.exists(img_path):
            cv2.imwrite(os.path.join(img_output_dir, img_name), cv2.imread(img_path))

        # Read image dimensions
        img_data = cv2.imread(img_path)
        if img_data is None:
            continue
        img_h, img_w, _ = img_data.shape

        # Convert COCO bbox (x, y, width, height) → YOLO (x_center, y_center, width, height) normalized
        x, y, w, h = ann["bbox"]
        x_center, y_center = (x + w / 2) / img_w, (y + h / 2) / img_h
        w, h = w / img_w, h / img_h

        # Write YOLO annotation file
        label_path = os.path.join(label_output_dir, img_name.replace(".jpg", ".txt"))
        with open(label_path, "a") as f:
            f.write(f"{ann['category_id']} {x_center} {y_center} {w} {h}\n")

    print(f"✅ COCO_Test split into YOLO_Test (80%) & YOLO_Val (20%)")


# -------------- RUN CONVERSION --------------
if __name__ == "__main__":
    convert_coco_to_yolo(annotation_file, DATASET_DIR, OUTPUT_DIR, VAL_SPLIT)
