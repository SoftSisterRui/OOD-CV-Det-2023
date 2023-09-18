import os
import json
import xml.etree.ElementTree as ET

# 设置XML和图像文件夹的路径
xml_folder = '/root/mmdetection-main/data/OOD-CV-det-2023/train/add_occlusion_anno'
image_folder = '/root/mmdetection-main/data/OOD-CV-det-2023/train/add_occlusion_image'
output_json_path = '/root/mmdetection-main/work_dirs/output.json'

# 初始化COCO格式数据结构
coco_data = {
    "info": {},
    "licenses": [],
    "categories": [
        {
            "supercategory": "none",
            "id": 1,
            "name": "aeroplane"
        },
        {
            "supercategory": "none",
            "id": 2,
            "name": "bicycle"
        },
        {
            "supercategory": "none",
            "id": 3,
            "name": "boat"
        },
        {
            "supercategory": "none",
            "id": 4,
            "name": "bus"
        },
        {
            "supercategory": "none",
            "id": 5,
            "name": "car"
        },
        {
            "supercategory": "none",
            "id": 6,
            "name": "chair"
        },
        {
            "supercategory": "none",
            "id": 7,
            "name": "diningtable"
        },
        {
            "supercategory": "none",
            "id": 8,
            "name": "motorbike"
        },
        {
            "supercategory": "none",
            "id": 9,
            "name": "sofa"
        },
        {
            "supercategory": "none",
            "id": 10,
            "name": "train"
        }
    ],
    "images": [],
    "annotations": []
}

# 获取图像文件列表
image_files = os.listdir(image_folder)
image_files = [f for f in image_files if f.endswith('.jpg')]
image_files.sort()

# 遍历XML文件并转换为COCO格式
for xml_file in os.listdir(xml_folder):
    if xml_file.endswith('.xml'):
        # 解析XML文件
        xml_path = os.path.join(xml_folder, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # 获取图像信息
        image_info = {
            "file_name": xml_file[:-4] + '.jpg',  # 替换扩展名
            "height": int(root.find('size').find('height').text),
            "width": int(root.find('size').find('width').text),
            "id": xml_file[:-4]  # 去掉扩展名
        }

        # 添加图像信息到COCO数据结构
        coco_data["images"].append(image_info)

        # 获取目标标注信息
        for obj in root.findall('object'):
            category_name = obj.find('name').text
            category_id = 0
            for category in coco_data["categories"]:
                if category["name"] == category_name:
                    category_id = category["id"]

            bbox = obj.find('bndbox')
            x_min = float(bbox.find('xmin').text)
            y_min = float(bbox.find('ymin').text)
            x_max = float(bbox.find('xmax').text)
            y_max = float(bbox.find('ymax').text)

            # 计算面积
            area = (x_max - x_min) * (y_max - y_min)

            # 构建标注信息
            annotation_info = {
                "area": area,
                "iscrowd": 0,
                "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                "category_id": category_id,
                "ignore": 0,
                "segmentation": [],
                "image_id": xml_file[:-4],  # 使用图像的ID
                "id": len(coco_data["annotations"]) + 1  # 自动生成唯一ID
            }

            # 添加标注信息到COCO数据结构
            coco_data["annotations"].append(annotation_info)

# 保存为COCO格式的JSON文件
with open(output_json_path, 'w') as json_file:
    json.dump(coco_data, json_file)
