import os
import cv2
import numpy as np
from imgaug import augmenters as iaa
import imgaug as ia

# 定义一个lambda表达式，以p=0.5的概率去执行sometimes传递的图像增强
sometimes = lambda aug: iaa.Sometimes(0.5, aug)

# 建立一个名为seq的实例，定义增强方法，用于增强
aug = iaa.Sequential(
    [
        iaa.SomeOf((3, 4),
                   [
                       iaa.imgcorruptlike.Snow(severity=1),  # 下雨、大雪
                       # iaa.imgcorruptlike.Fog(severity=1)
                       # iaa.imgcorruptlike.Spatter(severity=5)
                       # iaa.Rain(drop_size=(0.10, 0.15), speed=(0.2, 0.5)),  # 雨
                       # iaa.Snowflakes(flake_size=(0.1, 0.3), speed=(0.01, 0.05)),  # 雪点
                       # iaa.FastSnowyLandscape(lightness_threshold=(50, 80), lightness_multiplier=(1.0, 1.5)),  # 雪地
                   ],
                   random_order=True  # 随机的顺序把这些操作用在图像上
                   )
    ]
)

# 定义图片文件夹路径
image_folder = '/home/pr/download/mmdetection-main/data/OOD-CV-det-2023/train/add_rain_snow_fog/snow'

# 定义保存增强后图片的文件夹路径
output_folder = '/home/pr/download/mmdetection-main/data/OOD-CV-det-2023/train/add_rain_snow_fog/add_snow'

# 创建保存输出文件夹
os.makedirs(output_folder, exist_ok=True)

# 遍历图片文件夹中的所有图片
for image_file in os.listdir(image_folder):
    if image_file.endswith(('.jpg', '.png', '.jpeg')):  # 只处理图片文件
        image_path = os.path.join(image_folder, image_file)

        # 加载图像
        image = cv2.imread(image_path, 1)

        # 进行数据增强
        aug_img = aug(image=image)

        # 保存增强后的图片
        output_path = os.path.join(output_folder, image_file)
        cv2.imwrite(output_path, aug_img)