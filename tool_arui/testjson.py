import json
import random

# 原始JSON文件路径
original_json_path = '/home/pr/download/mmdetection-main/data/OOD-CV-det-2023/phase2-anno/weather.json'  # 替换为你的原始JSON文件路径

# 目标JSON文件路径
target_json_path = '/data/OOD-CV-det-2023/phase2train.json'  # 替换为你的目标JSON文件路径

# 打开原始JSON文件并加载数据
with open(original_json_path, 'r') as original_json_file:
    original_data = json.load(original_json_file)

# 从原始数据中随机选择两千张图的信息
random_images = random.sample(original_data['images'], 50)
random_image_ids = set(image['id'] for image in random_images)
random_annotations = [annotation for annotation in original_data['annotations'] if annotation['image_id'] in random_image_ids]

# 打开目标JSON文件并加载数据
with open(target_json_path, 'r') as target_json_file:
    target_data = json.load(target_json_file)

# 将随机选择的图像信息和注解信息追加到目标JSON文件中的相应位置
target_data['images'].extend(random_images)
target_data['annotations'].extend(random_annotations)

# 保存目标JSON文件
with open(target_json_path, 'w') as target_json_file:
    json.dump(target_data, target_json_file, indent=4)

print(f"Randomly selected {len(random_images)} images and added to {target_json_path}.")
