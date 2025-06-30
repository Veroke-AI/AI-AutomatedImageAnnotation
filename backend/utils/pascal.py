import os
import json
import xml.etree.ElementTree as ET
from xml.dom.minidom import parseString

def coco_to_voc(coco_json_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for json_file in os.listdir(coco_json_dir):
        if not json_file.endswith(".json"):
            continue

        with open(os.path.join(coco_json_dir, json_file), 'r') as f:
            coco = json.load(f)

        image_info = coco['images'][0]
        image_id = image_info['id']
        filename = image_info['file_name']
        width = image_info['width']
        height = image_info['height']
        depth = 3  # Assuming RGB

        annotations = [ann for ann in coco['annotations'] if ann['image_id'] == image_id]
        category_map = {cat['id']: cat['name'] for cat in coco['categories']}

        annotation = ET.Element('annotation')
        ET.SubElement(annotation, 'folder').text = 'images'
        ET.SubElement(annotation, 'filename').text = filename
        ET.SubElement(annotation, 'path').text = filename

        size = ET.SubElement(annotation, 'size')
        ET.SubElement(size, 'width').text = str(width)
        ET.SubElement(size, 'height').text = str(height)
        ET.SubElement(size, 'depth').text = str(depth)

        ET.SubElement(annotation, 'segmented').text = '0'

        for ann in annotations:
            bbox = ann['bbox']
            category_id = ann['category_id']
            category_name = category_map.get(category_id, 'unknown')

            obj = ET.SubElement(annotation, 'object')
            ET.SubElement(obj, 'name').text = category_name
            ET.SubElement(obj, 'pose').text = 'Unspecified'
            ET.SubElement(obj, 'truncated').text = '0'
            ET.SubElement(obj, 'difficult').text = '0'

            bndbox = ET.SubElement(obj, 'bndbox')
            x_min = int(bbox[0])
            y_min = int(bbox[1])
            x_max = int(bbox[0] + bbox[2])
            y_max = int(bbox[1] + bbox[3])

            ET.SubElement(bndbox, 'xmin').text = str(x_min)
            ET.SubElement(bndbox, 'ymin').text = str(y_min)
            ET.SubElement(bndbox, 'xmax').text = str(x_max)
            ET.SubElement(bndbox, 'ymax').text = str(y_max)

        # Beautify XML
        xml_str = ET.tostring(annotation, encoding='utf-8')
        pretty_xml = parseString(xml_str).toprettyxml(indent="  ")

        xml_filename = os.path.splitext(filename)[0] + '.xml'
        with open(os.path.join(output_dir, xml_filename), 'w') as xml_file:
            xml_file.write(pretty_xml)

    print(f"Converted all COCO JSONs to Pascal VOC XML in: {output_dir}")