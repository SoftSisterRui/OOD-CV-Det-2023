## Installation

Please refer to [Installation](https://mmdetection.readthedocs.io/en/latest/get_started.html) for installation instructions.

## OOD-CV Challenge 2023 (Detection Track - Self-supervised pretrain leaderboard)

Please put the training dataset into ./data/OOD-CV-det-2023/train/Images

After modifying the dataset path, use the following command for training.

```
python tools/train.py ./configs/yolox/yolox_x_8xb8-300e_coco.py --work-dir ./work_dirs/yolox_x_8xb8-300e_coco.py
```

When testing, it is necessary to first divide the test set into corresponding categories and modify the corresponding paths in the configuration file. Then use the following commands to test in sequence.

```
python tools/test.py ./configs/yolox/yolox_x_8xb8-300e_coco.py ./work_dirs/yolox_x_8xb8-300e_coco.py/epoch_300.pth 
```

## OOD-CV Challenge 2023 (Detection Track - ImageNet-1k leaderboard)

Add occlusion to the training set.(Requires training set and corresponding XML file)

```
python tools_arui/add_occlusion.py
```

Add weather changes to the training set.

```
python tools_arui/add_rain_snow_fog.py
```

Prepare a JSON file for the generated new dataset.

```
python tools_arui/xmltojson.py
```

Train the YOLOX model.

```
python tools/train.py ./configs/yolox/yolox_x_8xb8-300e_coco.py --work-dir ./work_dirs/yolox_x_8xb8-300e_coco.py
```

Train the DINO model.

```
python tools/train.py ./configs/dino/dino-5scale_swin-l_8xb2-36e_coco.py --work-dir ./work_dirs/dino-5scale_swin-l_8xb2-36e_coco.py
```

When testing, it is necessary to first divide the test set into corresponding categories and modify the corresponding paths in the configuration file. Then use the following commands to test in sequence.

```
python tools/test.py ./configs/yolox/yolox_x_8xb8-300e_coco.py ./work_dirs/yolox_x_8xb8-300e_coco.py/epoch_300.pth 
```

Command to use tta testing.
```
python tools/test.py ./configs/dino/dino-5scale_swin-l_8xb2-36e_coco.py ./work_dirs/dino-5scale_swin-l_8xb2-36e_coco.py/epoch36.pth --tta