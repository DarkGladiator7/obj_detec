import os
import json
import xml.etree.ElementTree as ET
from PIL import Image

def pascal_voc_to_coco(voc_annotation_dir, voc_image_dir, coco_dir, max_files=10):
    if not os.path.exists(coco_dir):
        os.makedirs(coco_dir)

    coco_annotations = {"images": [], "annotations": [], "categories": []}
    category_mapping = {}
    annotation_id = 1

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
        
        # Check if the image exists
        image_path = None
        for ext in [".jpg", ".jpeg", ".png"]:
            temp_path = os.path.join(voc_image_dir, filename.replace(".jpg", ext))
            if os.path.exists(temp_path):
                image_path = temp_path
                break

        if image_path is None:
            print(f"⚠️ Warning: Image {filename} not found in {voc_image_dir}. Skipping...")
            continue

        # Load image size
        image = Image.open(image_path)
        width, height = image.size
        image_id = idx + 1

        coco_annotations["images"].append({
            "id": image_id, "file_name": filename, "width": width, "height": height
        })

        image.save(os.path.join(coco_dir, filename))  # Save images in COCO folder

        for obj in root.findall('object'):
            category = obj.find('name').text
            if category not in category_mapping:
                category_id = len(category_mapping) + 1
                category_mapping[category] = category_id
                coco_annotations["categories"].append({"id": category_id, "name": category})
            else:
                category_id = category_mapping[category]

            bndbox = obj.find('bndbox')
            xmin, ymin, xmax, ymax = (int(bndbox.find(pos).text) for pos in ['xmin', 'ymin', 'xmax', 'ymax'])

            if xmax <= xmin or ymax <= ymin:
                print(f"⚠️ Skipping invalid bbox in {xml_file}: [{xmin}, {ymin}, {xmax}, {ymax}]")
                continue

            coco_annotations["annotations"].append({
                "id": annotation_id, "image_id": image_id, "category_id": category_id,
                "bbox": [xmin, ymin, xmax - xmin, ymax - ymin], "area": (xmax - xmin) * (ymax - ymin), "iscrowd": 0
            })
            annotation_id += 1

    with open(os.path.join(coco_dir, 'annotations_test.json'), 'w') as f:
        json.dump(coco_annotations, f, indent=4)

    print(f"✅ Successfully converted {len(xml_files)} VOC files to COCO format.")

# Set paths and run the function
annotations_dir = r"D:\obj_detec\VOC_2007\VOCtest_06-Nov-2007\VOCdevkit\VOC2007\Annotations"
images_dir = r"D:\obj_detec\VOC_2007\VOCtest_06-Nov-2007\VOCdevkit\VOC2007\JPEGImages"
coco_output_dir = r"D:\obj_detec\COCO_Annotations_Test"

pascal_voc_to_coco(annotations_dir, images_dir, coco_output_dir, max_files=10)
