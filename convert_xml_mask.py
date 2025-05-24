import os
import numpy as np
import cv2
import xml.etree.ElementTree as ET
from PIL import Image
from tqdm import tqdm

def convert_xml_to_mask(xml_path, image_shape=(1000, 1000)):
    """
    将单个 XML 标注文件转换为二值 mask
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    mask = np.zeros(image_shape, dtype=np.uint8)

    for region in root.findall(".//Region"):
        points = []
        for vertex in region.findall(".//Vertex"):
            x = int(float(vertex.get('X')))
            y = int(float(vertex.get('Y')))
            points.append([x, y])
        if len(points) > 2:
            cv2.fillPoly(mask, [np.array(points, dtype=np.int32)], color=1)
    return mask

annotation_dir = "data/test/Annotations"
mask_output_dir = "data/test/Masks"

for filename in tqdm(os.listdir(annotation_dir), desc="Converting XML to Mask"):
    if filename.endswith(".xml"):
        xml_path = os.path.join(annotation_dir, filename)
        mask = convert_xml_to_mask(xml_path, image_shape=(1000, 1000))

        mask_img = Image.fromarray(mask * 255)
        output_path = os.path.join(mask_output_dir, filename.replace(".xml", ".png"))
        mask_img.save(output_path)

print("所有标注文件已成功转换为 mask。")
