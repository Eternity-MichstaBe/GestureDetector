# GestureDetector
基于骨架信息的手语识别

## 非正式获取静态特征数据集（不推荐）：
执行get_static_dataset.py，通过opencv调取摄像头捕捉图像，图像文件保存在static_photo_data文件夹，提取的特征文件保存在static_numpy_data文件夹

## 正式获取静态特征数据集（推荐）：
采集高质量图片（视频）数据集按action-sample-frame组织为static_photo_data文件夹，执行get_static_dataset.py文件，仅提取特征保存为static_numpy_data文件夹

## 非正式获取动态特征数据集（不推荐）：
执行get_dynamic_dataset.py，通过opencv调取摄像头捕捉图像，图像文件保存在dynamic_photo_data文件夹，提取的特征文件保存在dynamic_numpy_data文件夹

## 正式获取动态特征数据集（推荐）：
采集高质量图片（视频）数据集按action-sample-frame组织为dynamic_photo_data文件夹，执行get_dynamic_dataset.py文件，仅提取特征保存为dynamic_numpy_data文件夹

## 静态手语识别模型训练：
执行static_hand_detect.py

## 动态手语识别模型训练：
执行dynamic_hand_detect.py

## gradio界面展示：
执行gradio_app.py