import cv2
import xml.etree.ElementTree as ET
import cv2
import xml.etree.ElementTree as ET

# 解析XML文件并获取边界框信息
def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    boxes = []

    for obj in root.findall('object'):
        box = obj.find('bndbox')
        xmin = int(box.find('xmin').text)
        ymin = int(box.find('ymin').text)
        xmax = int(box.find('xmax').text)
        ymax = int(box.find('ymax').text)
        boxes.append((xmin, ymin, xmax, ymax))

    return boxes

# 可视化XML文件中的边界框并保存图片
def visualize_and_save_boxes(image_path, xml_file, output_image_path):
    image = cv2.imread(image_path)
    boxes = parse_xml(xml_file)

    for xmin, ymin, xmax, ymax in boxes:
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    cv2.imwrite(output_image_path, image)

if __name__ == "__main__":
    image_path = '/root/mmdetection-main/data/OOD-CV-det-2023/train/Images/2008_000003.jpg'  # 你的图像文件路径
    xml_file = '/root/mmdetection-main/data/OOD-CV-det-2023/train/Annotations/2008_000003.xml'  # 你的XML文件路径
    output_image_path = '/root/mmdetection-main/work_dirs/output_image.jpg'  # 保存可视化结果的图像文件路径
    visualize_and_save_boxes(image_path, xml_file, output_image_path)

