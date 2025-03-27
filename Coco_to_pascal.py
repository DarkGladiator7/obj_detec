"""
📌 COCO to Pascal VOC Format Conversion

🔹 COCO Format (JSON):
    - Annotations stored in a single JSON file.
    - Bounding boxes in `[x, y, width, height]` format.
    - Uses category IDs for object classes.

🔹 Pascal VOC Format (XML):
    - Each image has a corresponding `.xml` annotation file.
    - Bounding boxes stored in `[xmin, ymin, xmax, ymax]` format.
    - Contains additional metadata like image size & filename.

🔹 Conversion Steps:
    1. Load COCO JSON annotations.
    2. Extract image details (filename, width, height).
    3. Convert bounding boxes:
       `[x, y, width, height] → [xmin, ymin, xmax, ymax]`
    4. Assign class names according to COCO categories.
    5. Save images & annotations in Pascal VOC format.

📂 **Output Structure:**
    ├── Pascal_VOC_Dataset/
    │   ├── JPEGImages/    (Converted images)
    │   ├── Annotations/   (XML annotation files)

✅ Example Pascal VOC Annotation:
    ```xml
    <annotation>
        <filename>image1.jpg</filename>
        <size>
            <width>640</width>
            <height>480</height>
        </size>
        <object>
            <name>dog</name>
            <bndbox>
                <xmin>100</xmin>
                <ymin>150</ymin>
                <xmax>300</xmax>
                <ymax>400</ymax>
            </bndbox>
        </object>
    </annotation>
    ```
"""

import os
import json
import cv2
import xml.etree.ElementTree as ET
from tqdm import tqdm

# -------------- CONFIGURATION --------------
DATASET_DIR = r"D:\obj_detec\COCO_DATA\COCO_Train"  # Change for different datasets
OUTPUT_DIR = r"D:\obj_detec\Pascal_VOC_Train"  # Output directory for Pascal VOC format

# Automatically find the COCO annotation file
annotation_file = None
for file in os.listdir(DATASET_DIR):
    if file.endswith(".json"):  # Look for a JSON annotation file
        annotation_file = os.path.join(DATASET_DIR, file)
        break

if annotation_file is None:
    raise FileNotFoundError("No COCO JSON annotation file found in the dataset directory!")

# Create output directories
VOC_IMG_DIR = os.path.join(OUTPUT_DIR, "JPEGImages")
VOC_ANNOTATIONS_DIR = os.path.join(OUTPUT_DIR, "Annotations")
os.makedirs(VOC_IMG_DIR, exist_ok=True)
os.makedirs(VOC_ANNOTATIONS_DIR, exist_ok=True)


def create_pascal_voc_xml(filename, width, height, objects):
    """Generate Pascal VOC XML annotation for an image."""
    annotation = ET.Element("annotation")

    # Add filename
    ET.SubElement(annotation, "filename").text = filename

    # Add image size details
    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)

    # Add object annotations
    for obj in objects:
        obj_elem = ET.SubElement(annotation, "object")
        ET.SubElement(obj_elem, "name").text = obj["name"]

        bndbox = ET.SubElement(obj_elem, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(obj["xmin"])
        ET.SubElement(bndbox, "ymin").text = str(obj["ymin"])
        ET.SubElement(bndbox, "xmax").text = str(obj["xmax"])
        ET.SubElement(bndbox, "ymax").text = str(obj["ymax"])

    # Convert XML to string format
    return ET.tostring(annotation, encoding="unicode")


def convert_coco_to_pascal_voc(coco_json, image_dir, output_dir):
    """ Convert COCO JSON annotations to Pascal VOC (XML) format """

    # Load COCO JSON
    with open(coco_json, "r") as f:
        data = json.load(f)

    # Map image IDs to filenames
    images = {img["id"]: img["file_name"] for img in data["images"]}
    # Map category IDs to category names
    categories = {cat["id"]: cat["name"] for cat in data["categories"]}

    # Process each image
    for img_id, img_name in tqdm(images.items(), desc="Converting Annotations"):
        img_path = os.path.join(image_dir, img_name)

        # Skip if the image file doesn't exist
        if not os.path.exists(img_path):
            continue

        # Read image dimensions
        img_data = cv2.imread(img_path)
        if img_data is None:
            continue
        img_h, img_w, _ = img_data.shape

        # Copy image to Pascal VOC dataset folder
        cv2.imwrite(os.path.join(VOC_IMG_DIR, img_name), img_data)

        # Process annotations for this image
        objects = []
        for ann in data["annotations"]:
            if ann["image_id"] != img_id:
                continue

            # Convert COCO bbox (x, y, width, height) → Pascal VOC (xmin, ymin, xmax, ymax)
            x, y, w, h = ann["bbox"]
            xmin, ymin, xmax, ymax = int(x), int(y), int(x + w), int(y + h)

            # Get class name
            class_name = categories[ann["category_id"]]

            # Store object details
            objects.append({"name": class_name, "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax})

        # Generate XML annotation file
        xml_data = create_pascal_voc_xml(img_name, img_w, img_h, objects)
        xml_filename = os.path.join(VOC_ANNOTATIONS_DIR, img_name.replace(".jpg", ".xml").replace(".png", ".xml"))

        # Save XML file
        with open(xml_filename, "w") as f:
            f.write(xml_data)

    print(f"✅ COCO annotations converted to Pascal VOC format in {output_dir}")


# -------------- RUN CONVERSION --------------
if __name__ == "__main__":
    convert_coco_to_pascal_voc(annotation_file, DATASET_DIR, OUTPUT_DIR)
