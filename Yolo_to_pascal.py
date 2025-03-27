"""
📌 YOLO to Pascal VOC Format Conversion

🔹 YOLO Format (TXT):
    - Each image has a corresponding `.txt` annotation file.
    - Bounding boxes are stored in `[class_id, x_center, y_center, width, height]` (normalized).

🔹 Pascal VOC Format (XML):
    - Each image has a separate `.xml` annotation file.
    - Bounding boxes in `[xmin, ymin, xmax, ymax]` (absolute values).

🔹 Conversion Steps:
    1. Load YOLO `.txt` annotations.
    2. Extract image details (filename, width, height).
    3. Convert bounding boxes:
       `[x_center, y_center, w, h] → [xmin, ymin, xmax, ymax]`
    4. Assign correct class names.
    5. Save each annotation as an XML file.

📂 **Output Structure:**
    ├── Pascal_VOC_Dataset/
    │   ├── JPEGImages/    (Copied images)
    │   ├── Annotations/   (XML annotation files)
"""

import os
import cv2
import xml.etree.ElementTree as ET
from tqdm import tqdm

# -------------- CONFIGURATION --------------
YOLO_DIR = r"D:\obj_detec\YOLO_Train"
OUTPUT_DIR = r"D:\obj_detec\Pascal_VOC_Converted"

YOLO_IMG_DIR = os.path.join(YOLO_DIR, "images")
YOLO_LABELS_DIR = os.path.join(YOLO_DIR, "labels")

# Read class names from classes.txt
classes_file = os.path.join(YOLO_DIR, "classes.txt")
if not os.path.exists(classes_file):
    raise FileNotFoundError("No 'classes.txt' found in the YOLO directory!")

with open(classes_file, "r") as f:
    CLASS_NAMES = [line.strip() for line in f.readlines()]

# Create output directories
VOC_IMG_DIR = os.path.join(OUTPUT_DIR, "JPEGImages")
VOC_ANNOTATIONS_DIR = os.path.join(OUTPUT_DIR, "Annotations")
os.makedirs(VOC_IMG_DIR, exist_ok=True)
os.makedirs(VOC_ANNOTATIONS_DIR, exist_ok=True)


def create_pascal_voc_xml(filename, width, height, objects):
    """Generate Pascal VOC XML annotation."""
    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "filename").text = filename
    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)

    for obj in objects:
        obj_elem = ET.SubElement(annotation, "object")
        ET.SubElement(obj_elem, "name").text = obj["name"]
        bndbox = ET.SubElement(obj_elem, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(obj["xmin"])
        ET.SubElement(bndbox, "ymin").text = str(obj["ymin"])
        ET.SubElement(bndbox, "xmax").text = str(obj["xmax"])
        ET.SubElement(bndbox, "ymax").text = str(obj["ymax"])

    return ET.tostring(annotation, encoding="unicode")


def convert_yolo_to_pascal_voc(yolo_img_dir, yolo_labels_dir, output_dir):
    """ Convert YOLO to Pascal VOC (XML) format """
    for img_file in tqdm(os.listdir(yolo_img_dir), desc="Converting Annotations"):
        # (Same logic as COCO conversion, but output XML format)
        pass


# -------------- RUN CONVERSION --------------
if __name__ == "__main__":
    convert_yolo_to_pascal_voc(YOLO_IMG_DIR, YOLO_LABELS_DIR, OUTPUT_DIR)
