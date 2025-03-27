"""
📌 Pascal VOC to YOLO Format Conversion

🔹 Pascal VOC Format (XML):
    - Contains bounding box coordinates as (xmin, ymin, xmax, ymax).
    - Each annotation is stored in an XML file.

🔹 YOLO Format (TXT):
    - Each image has a corresponding `.txt` file with the format:
      `class_id x_center y_center width height`
    - Bounding box values are **normalized (0 to 1)** relative to image width & height.

🔹 Conversion Steps:
    1. Read Pascal VOC XML annotations.
    2. Convert bounding boxes:
       (xmin, ymin, xmax, ymax) → (x_center, y_center, width, height)
    3. Normalize values by dividing by image width & height.
    4. Save YOLO `.txt` annotation files.
    5. Store class names in `classes.txt`.

📂 **Output Structure:**
    ├── YOLO_Annotations_Test/
    │   ├── images/        (Converted images)
    │   ├── labels/        (YOLO annotation `.txt` files)
    │   ├── classes.txt    (Class names)

✅ Example YOLO Annotation:
    ```
    0 0.456789 0.512345 0.234567 0.345678
    1 0.678901 0.423456 0.198765 0.287654
    ```
    (where `0`, `1` are class IDs, followed by normalized bbox values)

"""

import os
import xml.etree.ElementTree as ET
from PIL import Image

def pascal_voc_to_yolo(voc_annotation_dir, voc_image_dir, yolo_output_dir):
    """Convert Pascal VOC XML annotations to YOLO format."""
    
    # Create YOLO output directories
    yolo_labels_dir = os.path.join(yolo_output_dir, "labels")
    yolo_images_dir = os.path.join(yolo_output_dir, "images")

    os.makedirs(yolo_labels_dir, exist_ok=True)
    os.makedirs(yolo_images_dir, exist_ok=True)

    class_mapping = {}  # Stores class names

    xml_files = [f for f in os.listdir(voc_annotation_dir) if f.endswith('.xml')]
    print(f"🔍 Found {len(xml_files)} XML annotation files.")

    if len(xml_files) == 0:
        print("❌ No XML files found. Check your VOC dataset path.")
        return

    for idx, xml_file in enumerate(xml_files):
        xml_path = os.path.join(voc_annotation_dir, xml_file)
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except Exception as e:
            print(f"❌ Error parsing {xml_file}: {e}")
            continue

        filename = root.find('filename').text  # e.g., '000005.jpg'
        image_path = None
        for ext in [".jpg", ".jpeg", ".png"]:
            temp_path = os.path.join(voc_image_dir, filename.replace(".jpg", ext))
            if os.path.exists(temp_path):
                image_path = temp_path
                break

        if image_path is None:
            print(f"⚠️ Warning: Image {filename} not found in {voc_image_dir}. Skipping...")
            continue

        image = Image.open(image_path)
        width, height = image.size
        image_id = idx + 1

        # Save image in YOLO folder
        image.save(os.path.join(yolo_images_dir, filename))

        yolo_annotation = []
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            if class_name not in class_mapping:
                class_mapping[class_name] = len(class_mapping)

            class_id = class_mapping[class_name]
            bndbox = obj.find('bndbox')
            xmin, ymin, xmax, ymax = (int(bndbox.find(pos).text) for pos in ['xmin', 'ymin', 'xmax', 'ymax'])

            if xmax <= xmin or ymax <= ymin:
                print(f"⚠️ Skipping invalid bbox in {xml_file}: [{xmin}, {ymin}, {xmax}, {ymax}]")
                continue

            # Convert to YOLO format (normalized)
            x_center = (xmin + xmax) / (2.0 * width)
            y_center = (ymin + ymax) / (2.0 * height)
            bbox_width = (xmax - xmin) / width
            bbox_height = (ymax - ymin) / height

            yolo_annotation.append(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}")

        # Save YOLO annotation file
        txt_filename = os.path.splitext(filename)[0] + ".txt"
        with open(os.path.join(yolo_labels_dir, txt_filename), "w") as f:
            f.write("\n".join(yolo_annotation))

    # Save class names
    with open(os.path.join(yolo_output_dir, "classes.txt"), "w") as f:
        for class_name in class_mapping:
            f.write(f"{class_name}\n")

    print(f"✅ Successfully converted {len(xml_files)} VOC files to YOLO format.")

# Set paths and run the function
voc_annotations_dir = r"D:\obj_detec\VOC_2007\VOCtest_06-Nov-2007\VOCdevkit\VOC2007\Annotations"
voc_images_dir = r"D:\obj_detec\VOC_2007\VOCtest_06-Nov-2007\VOCdevkit\VOC2007\JPEGImages"
yolo_output_dir = r"D:\obj_detec\YOLO_Annotations_Test"

pascal_voc_to_yolo(voc_annotations_dir, voc_images_dir, yolo_output_dir)
