"""
📌 COCO to YOLO Format Conversion

🔹 COCO Format (JSON):
    - Annotations stored in a JSON file.
    - Bounding boxes in `[x, y, width, height]` format.

🔹 YOLO Format (.txt):
    - Uses a `.txt` file for each image.
    - Bounding boxes in `[class_id, x_center, y_center, width, height]` format (normalized).

🔹 Conversion Steps:
    1. Load COCO JSON annotations.
    2. Extract image details (filename, width, height).
    3. Convert bounding boxes:
       `[x, y, width, height] → [x_center, y_center, width, height]` (normalized)
    4. Assign class IDs according to the category list.
    5. Save images & annotations in YOLO format.

📂 **Output Structure:**
    ├── YOLO_Dataset/
    │   ├── images/        (Converted images)
    │   ├── labels/        (YOLO labels)
    │   ├── classes.txt    (Class names in order)

✅ Example YOLO Annotation (per image):
    ```
    0 0.5 0.5 0.2 0.3
    1 0.3 0.4 0.15 0.25
    ```

"""

import os
import json
import cv2
from tqdm import tqdm

# -------------- CONFIGURATION --------------
DATASET_DIR = r"D:\obj_detec\COCO_DATA\COCO_Train"  # Change for different datasets
OUTPUT_DIR = r"D:\obj_detec\YOLO_Train"  # Output directory for YOLO format

# Automatically find the annotation file in the dataset directory
annotation_file = None
for file in os.listdir(DATASET_DIR):
    if file.endswith(".json"):  # Look for a JSON annotation file
        annotation_file = os.path.join(DATASET_DIR, file)
        break

if annotation_file is None:
    raise FileNotFoundError(
        "No COCO JSON annotation file found in the dataset directory!")

# Create output directories
YOLO_IMG_DIR = os.path.join(OUTPUT_DIR, "images")
YOLO_LABELS_DIR = os.path.join(OUTPUT_DIR, "labels")
os.makedirs(YOLO_IMG_DIR, exist_ok=True)
os.makedirs(YOLO_LABELS_DIR, exist_ok=True)


def convert_coco_to_yolo(coco_json, image_dir, output_dir):
    """ Convert COCO JSON annotations to YOLO format """

    # Load COCO JSON
    with open(coco_json, "r") as f:
        data = json.load(f)

    # Map image IDs to filenames
    images = {img["id"]: img["file_name"] for img in data["images"]}
    # Map category IDs to category names
    categories = {cat["id"]: cat["name"] for cat in data["categories"]}

    # Create a class list based on COCO dataset categories
    class_list = sorted(categories.values())  # Ensures consistent order
    class_name_to_id = {name: idx for idx, name in enumerate(class_list)}

    # Save class names in `classes.txt`
    with open(os.path.join(output_dir, "classes.txt"), "w") as f:
        for name in class_list:
            f.write(f"{name}\n")

    # Process each annotation
    for ann in tqdm(data["annotations"], desc="Converting Annotations"):
        img_id = ann["image_id"]
        img_name = images[img_id]
        img_path = os.path.join(image_dir, img_name)

        # Skip if the image file doesn't exist
        if not os.path.exists(img_path):
            continue

        # Read image dimensions
        img_data = cv2.imread(img_path)
        if img_data is None:
            continue
        img_h, img_w, _ = img_data.shape

        # Copy image to YOLO images folder
        cv2.imwrite(os.path.join(YOLO_IMG_DIR, img_name), img_data)

        # Convert COCO bbox (x, y, width, height) → YOLO (x_center, y_center, width, height) normalized
        x, y, w, h = ann["bbox"]
        x_center, y_center = (x + w / 2) / img_w, (y + h / 2) / img_h
        w, h = w / img_w, h / img_h

        # Get correct class ID
        coco_class_name = categories[ann["category_id"]]
        correct_class_id = class_name_to_id[coco_class_name]

        # Write YOLO annotation file
        label_path = os.path.join(YOLO_LABELS_DIR, img_name.replace(
            ".jpg", ".txt").replace(".png", ".txt"))
        with open(label_path, "a") as f:
            f.write(f"{correct_class_id} {x_center} {y_center} {w} {h}\n")

    print(f"✅ COCO annotations converted to YOLO format in {output_dir}")


# -------------- RUN CONVERSION --------------
if __name__ == "__main__":
    convert_coco_to_yolo(annotation_file, DATASET_DIR, OUTPUT_DIR)
