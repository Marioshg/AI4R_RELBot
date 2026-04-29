import os
import xml.etree.ElementTree as ET
import shutil
from pathlib import Path

# Set up paths
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
SOURCE_DIR = os.path.join(BASE_DIR, "Hardhat")
OUTPUT_DIR = os.path.join(BASE_DIR, "Hardhat_YOLO")

# Define target classes (we found 'helmet' in the XML)
CLASSES = ["helmet", "head", "person"] 

def convert_bbox(size, box):
    """Converts PASCAL VOC bounding box to YOLO format."""
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def prepare_yolo_data(split_name="Train", yolo_split="train"):
    """Reads XMLs and copies images + text labels into YOLO directories."""
    ann_dir = os.path.join(SOURCE_DIR, split_name, "Annotation")
    img_dir = os.path.join(SOURCE_DIR, split_name, "JPEGImage")
    
    out_img_dir = os.path.join(OUTPUT_DIR, "images", yolo_split)
    out_lbl_dir = os.path.join(OUTPUT_DIR, "labels", yolo_split)
    
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)
    
    if not os.path.exists(ann_dir):
        print(f"Warning: {ann_dir} not found. Skipping.")
        return

    # Process all XML files
    xml_files = [f for f in os.listdir(ann_dir) if f.endswith('.xml')]
    print(f"Processing {len(xml_files)} files for {yolo_split} split...")
    
    for xml_file in xml_files:
        tree = ET.parse(os.path.join(ann_dir, xml_file))
        root = tree.getroot()
        
        # Get image dimensions
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        
        # Output YOLO label file
        txt_filename = xml_file.replace(".xml", ".txt")
        txt_path = os.path.join(out_lbl_dir, txt_filename)
        
        has_objects = False
        with open(txt_path, "w") as out_txt:
            for obj in root.iter('object'):
                difficult = obj.find('difficult').text if obj.find('difficult') is not None else 0
                cls_name = obj.find('name').text
                
                # We dynamically add the class if it's new
                if cls_name not in CLASSES:
                    CLASSES.append(cls_name)
                
                cls_id = CLASSES.index(cls_name)
                
                xmlbox = obj.find('bndbox')
                b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), 
                     float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
                
                bb = convert_bbox((w, h), b)
                out_txt.write(f"{cls_id} {' '.join([str(a) for a in bb])}\n")
                has_objects = True

        # Copy the corresponding image
        img_filename = root.find('filename').text
        src_img_path = os.path.join(img_dir, img_filename)
        dst_img_path = os.path.join(out_img_dir, img_filename)
        
        if os.path.exists(src_img_path):
            shutil.copy(src_img_path, dst_img_path)
        else:
            # Try to guess JPG if filename in XML is wrong or missing extension
            guess_img = os.path.join(img_dir, xml_file.replace(".xml", ".jpg"))
            if os.path.exists(guess_img):
                shutil.copy(guess_img, os.path.join(out_img_dir, xml_file.replace(".xml", ".jpg")))

def create_yaml():
    yaml_content = f"""path: {OUTPUT_DIR}
train: images/train
val: images/val

# Classes
names:
"""
    for i, cls in enumerate(CLASSES):
        yaml_content += f"  {i}: {cls}\n"
        
    yaml_path = os.path.join(BASE_DIR, "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"Created YOLO configuration file at: {yaml_path}")

if __name__ == "__main__":
    print("Converting Pascal VOC to YOLO format...")
    prepare_yolo_data("Train", "train")
    prepare_yolo_data("Test", "val")
    create_yaml()
    print("Done! Data is ready for YOLOv8.")