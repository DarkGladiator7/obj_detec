"""
📌 YOLO to COCO Format Conversion

🔹 YOLO Format (TXT):
    - Each image has a corresponding `.txt` annotation file.
    - Bounding boxes are stored in `[class_id, x_center, y_center, width, height]` (normalized).

🔹 COCO Format (JSON):
    - Single JSON file storing all annotations.
    - Bounding boxes in `[x, y, width, height]` (absolute values).
    - Uses category IDs for object classes.

🔹 Conversion Steps:
    1. Load YOLO `.txt` annotations.
    2. Extract image details (filename, width, height).
    3. Convert bounding boxes:
       `[x_center, y_center, w, h] → [x, y, width, height]`
    4. Assign category IDs based on class mapping.
    5. Save as `annotations.json`.

📂 **Output Structure:**
    ├── COCO_Dataset/
    │   ├── images/         (Copied images)
    │   ├── annotations.json (Generated COCO annotations)
"""

import os
import json
import cv2
from tqdm import tqdm

# -------------- CONFIGURATION --------------
YOLO_DIR = r"D:\obj_detec\YOLO_Train"  # Change to YOLO_Val or YOLO_Test when needed
OUTPUT_DIR = r"D:\obj_detec\COCO_Converted"  # Where COCO dataset will be stored

YOLO_IMG_DIR = os.path.join(YOLO_DIR, "images")
YOLO_LABELS_DIR = os.path.join(YOLO_DIR, "labels")

# Read class names from classes.txt
classes_file = os.path.join(YOLO_DIR, "classes.txt")
if not os.path.exists(classes_file):
    raise FileNotFoundError("No 'classes.txt' found in the YOLO directory!")

with open(classes_file, "r") as f:
    CLASS_NAMES = [line.strip() for line in f.readlines()]

# Create category mapping
category_mapping = {name: idx + 1 for idx, name in enumerate(CLASS_NAMES)}

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)


def convert_yolo_to_coco(yolo_img_dir, yolo_labels_dir, output_dir):
    """ Convert YOLO annotations to COCO JSON format """

    coco_annotations = {"images": [], "annotations": [], "categories": []}
    annotation_id = 1

    # Define category details
    for name, cat_id in category_mapping.items():
        coco_annotations["categories"].append({"id": cat_id, "name": name})

    # Process each image
    for img_file in tqdm(os.listdir(yolo_img_dir), desc="Converting Annotations"):
        if not img_file.lower().endswith((".jpg", ".png")):
            continue

        img_path = os.path.join(yolo_img_dir, img_file)
        label_path = os.path.join(yolo_labels_dir, img_file.replace(".jpg", ".txt").replace(".png", ".txt"))

        # Skip if no corresponding label file
        if not os.path.exists(label_path):
            continue

        # Read image dimensions
        img_data = cv2.imread(img_path)
        if img_data is None:
            continue
        img_h, img_w, _ = img_data.shape

        # Add image details to COCO format
        img_id = len(coco_annotations["images"]) + 1
        coco_annotations["images"].append({"id": img_id, "file_name": img_file, "width": img_w, "height": img_h})

        # Read YOLO annotation file
        with open(label_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            class_id, x_center, y_center, w, h = map(float, parts)

            # Convert to COCO format (absolute values)
            x, y = (x_center - w / 2) * img_w, (y_center - h / 2) * img_h
            w, h = w * img_w, h * img_h

            # Add annotation to COCO format
            coco_annotations["annotations"].append({
                "id": annotation_id,
                "image_id": img_id,
                "category_id": int(class_id) + 1,
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0
            })
            annotation_id += 1

    # Save COCO JSON file
    with open(os.path.join(output_dir, "annotations.json"), "w") as f:
        json.dump(coco_annotations, f, indent=4)

    print(f"✅ YOLO annotations converted to COCO format in {output_dir}")


# -------------- RUN CONVERSION --------------
if __name__ == "__main__":
    convert_yolo_to_coco(YOLO_IMG_DIR, YOLO_LABELS_DIR, OUTPUT_DIR)
