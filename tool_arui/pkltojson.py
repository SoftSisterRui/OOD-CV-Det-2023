import pickle
import json

# 读取.pkl文件
pkl_file_path = "/root/mmdetection-main/work_dirs/yolox_test/context.pkl"
with open(pkl_file_path, "rb") as file:
    pkl_data = pickle.load(file)

# 转换为COCO格式的JSON
coco_results = []
for img_info in pkl_data:
    img_path = img_info['img_path']
    img_id = img_path.split('/')[-1].split('.')[0]  # 提取图片名称作为image_id
    pred_instances = img_info['pred_instances']

    for j in range(len(pred_instances['labels'])):
        result_entry = {
            "bbox": pred_instances['bboxes'][j].tolist(),
            "image_id": img_id,
            "score": pred_instances['scores'][j].item(),
            "category_id": pred_instances['labels'][j].item() + 1
        }
        coco_results.append(result_entry)

# 将COCO格式的JSON保存到文件
output_json_path = "/root/mmdetection-main/result/context.json"
with open(output_json_path, "w") as json_file:
    json.dump(coco_results, json_file)

print(f"Saved COCO format JSON to {output_json_path}")
