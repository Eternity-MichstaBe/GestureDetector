"""
由于获取数据集对图片最低要求是手部清晰能被识别
大量的图片收集较为麻烦且需要挑选,
故将数据集获取功能独立出来
该功能为获取静态数据集用于静态手语识别
"""

import shutil
import cv2  # opencv_python:4.5.2.52
import numpy as np
import os
import mediapipe as mp  # 版本:0.9.0.1


def hand_detection(image, hand_model):
    """
    手部检测
    :param image:实时获取的图像
    :param hand_model:检测模型（holistic:整体检测）
    :return:返回图像和关键点模型检测信息
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR格式转为RGB
    image.flags.writeable = False
    results = hand_model.process(image)  # 处理RGB图像
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # 转回BGR输出
    return image, results


def draw_landmarks(image, results, hand_style, hand_line_style):
    """
    关键点绘制
    :param hand_line_style: 关键点连线样式
    :param hand_style: 关键点样式
    :param image:输入图像
    :param results:关键点模型检测信息
    :return:
    """
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks,
                              mp_holistic.HAND_CONNECTIONS, hand_style, hand_line_style)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks,
                              mp_holistic.HAND_CONNECTIONS, hand_style, hand_line_style)


def get_datasets(image_path):
    """
    创建数据目录
    :param image_path: 图片存放路径
    :return:无
    """
    if os.path.exists(image_path) is True:  # 目录存在选择是否重置
        judge_dir = input("图片目录已存在!是否重置？y/n")
        # 容错输入
        while True:
            if judge_dir == 'y' or judge_dir == 'n':
                break
            else:
                judge_dir = input("输入错误重新输入!是否重置？y/n")

        # 重置目录
        if judge_dir == 'y':
            shutil.rmtree(image_path, ignore_errors=True)
            for action in actions:
                for sequence in range(num_action):
                    try:
                        os.makedirs(os.path.join(image_path, action, str(sequence)))  # 新建目录
                    except Exception as e:
                        raise str(e)
            # 获取数据集
            get_photo(image_path)
        elif judge_dir == 'n':  # 不重建目录
            pass
    else:
        for action in actions:
            for sequence in range(num_action):
                try:
                    os.makedirs(os.path.join(image_path, action, str(sequence)))
                except Exception as e:
                    raise str(e)
        # 获取数据集
        get_photo(image_path)


def get_photo(image_path):
    """
    实时获取图片作为数据集
    :param image_path: 图片存放路径
    :return: 无
    """
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(3, 1280)
    cap.set(4, 720)

    with mp_holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75) as holistic:
        # print("请准备好手势,即将开始检测!")
        for action in actions:
            for sequence in range(num_action):
                for frame_sequence in range(num_frame):
                    ret, frame = cap.read()
                    frame = cv2.flip(frame, 1)  # 图像做镜像翻转左右手互换（符合习惯）
                    image, results = hand_detection(frame, holistic)  # 手部检测

                    draw_landmarks(image, results, hand_Style, handLine_Style)  # 绘制关键点

                    locate_height = int(frame.shape[0] / 2)  # 显示定位
                    locate_width = int(frame.shape[1] / 3)
                    # 获取并保存一个帧信息
                    if sequence == 0:
                        cv2.putText(image, 'STARTING COLLECTION', (locate_width, locate_height),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                                    4, cv2.LINE_AA)
                        cv2.putText(image, 'COLLECTING FRAMES FOR {} VIDEO NUMBER {}'
                                    .format(action, sequence), (50, 25), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (0, 0, 255), 1, cv2.LINE_AA)

                    else:
                        cv2.putText(image, 'COLLECTING FRAMES FOR {} VIDEO NUMBER {}'
                                    .format(action, sequence), (50, 25), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (0, 0, 255), 1, cv2.LINE_AA)

                    cv2.waitKey(0)
                    photo_path = os.path.join(image_path, action, str(sequence), str(frame_sequence))  # 图片保存路径
                    cv2.imwrite(photo_path + '.jpg', frame)
                    cv2.imshow('get_photo_data', image)

        cap.release()
        cv2.destroyAllWindows()


def extract_keypoint(results):
    """
    提取图片关键点信息
    :param results:关键点模型检测信息
    :return:返回关键点坐标信息
    """
    if results.right_hand_landmarks:
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()  # 提取左手关键点坐标信息
        # print(rh.shape)  # 空白数据格式保持一致
    else:
        rh = np.zeros(21 * 3)  # 空白数据格式
        # print('未检测到左手!')   # 图像做镜像翻转左右互换

    if results.left_hand_landmarks:
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()  # 提取右手关键点坐标信息
        # print(lh.shape)
    else:
        lh = np.zeros(21 * 3)
        # print('未检测到右手!')

    return np.concatenate([lh, rh])


def get_keep_keypoint(data_path, image_path):
    """
    从图片实时提取并保存关键点信息
    :param image_path: 图片存放路径
    :param data_path: 关键点信息存放路径
    :return: 无
    """

    with mp_holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75) as holistic:
        for action in actions:
            for sequence in range(num_action):
                for frame_sequence in range(num_frame):
                    # 图片路径
                    photo_path = os.path.join(image_path, action, str(sequence), '{}.jpg'.format(str(frame_sequence)))
                    print(photo_path + '已提取特征信息')
                    frame = cv2.imread(photo_path)
                    image, results = hand_detection(frame, holistic)  # 手部检测
                    draw_landmarks(image, results, hand_Style, handLine_Style)  # 绘制关键点
                    keypoint = extract_keypoint(results)  # 提取关键点信息
                    # print(keypoint.shape)
                    npy_dir = os.path.join(data_path, action, str(sequence))
                    if not os.path.exists(npy_dir):
                        os.makedirs(npy_dir)
                    npy_path = os.path.join(data_path, action, str(sequence), str(frame_sequence))  # numpy数组路径

                    np.save(npy_path, keypoint)  # 保存每帧的关键点信息


if __name__ == "__main__":
    IMAGE_PATH = 'static_photo_data'
    DATA_PATH = 'static_numpy_data'
    actions = np.array(['I', 'You', 'Good', 'Salute', 'Cow', 'Wait'])  # 动作集合
    num_action = 2  # 每个动作样本数
    num_frame = 1  # 每个样本帧数

    mp_holistic = mp.solutions.holistic  # 全身特征检测模型
    mp_drawing = mp.solutions.drawing_utils  # 绘制关键点

    hand_Style = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=5)  # 关键点样式
    handLine_Style = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=10)  # 连线样式

    # 获取图片数据集
    judge = input("是否采集图片?y/n")
    if judge == 'y':
        get_datasets(IMAGE_PATH)
    get_keep_keypoint(DATA_PATH, IMAGE_PATH)

