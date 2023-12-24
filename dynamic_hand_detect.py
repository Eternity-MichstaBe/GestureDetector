"""
完整的动态手语识别
"""
import time

import cv2     # opencv_python:4.5.2.52
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader, TensorDataset
import mediapipe as mp    # 版本:0.9.0.1
from sklearn.metrics import accuracy_score   # scikit_learn:0.24.1
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
from tqdm import tqdm


class LSTM(nn.Module):
    """
        Parameters：
        - input_size: feature size
        - hidden_size: number of hidden units
        - output_size: number of output
        - num_layers: layers of LSTM to stack
    """

    def __init__(self, input_size, hidden_size1=128, hidden_size2=64, output_size=6, num_layers=1, batch=1):
        super().__init__()

        self.lstm1 = nn.LSTM(input_size, hidden_size1, num_layers)
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, num_layers)
        self.l1 = nn.Linear(hidden_size2*batch, output_size)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        x = x.view(-1, b * h)
        x = self.l1(x)
        x = nn.functional.softmax(x, dim=1)

        return x


class GRU(nn.Module):
    """
        Parameters：
        - input_size: feature size
        - hidden_size: number of hidden units
        - output_size: number of output
        - num_layers: layers of LSTM to stack
    """

    def __init__(self, input_size, hidden_size1=128, hidden_size2=64, output_size=6, num_layers=1, batch=1):
        super().__init__()

        self.gru1 = nn.GRU(input_size, hidden_size1, num_layers)
        self.gru2 = nn.GRU(hidden_size1, hidden_size2, num_layers)
        self.l1 = nn.Linear(hidden_size2*batch, output_size)

    def forward(self, x):
        x, _ = self.gru1(x)
        x, _ = self.gru2(x)
        s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        x = x.view(-1, b * h)
        x = self.l1(x)
        x = nn.functional.softmax(x, dim=1)

        return x


def hand_detection(image, hand_model):
    """n
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
    # 左手关键点绘制
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks,
                              mp_holistic.HAND_CONNECTIONS, hand_style, hand_line_style)
    # 右手关键点绘制
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks,
                              mp_holistic.HAND_CONNECTIONS, hand_style, hand_line_style)
    # 面部关键点绘制
    mp_drawing.draw_landmarks(image, results.face_landmarks,
                              mp_holistic.FACEMESH_CONTOURS, hand_style, hand_line_style)
    # 身体关键点绘制
    mp_drawing.draw_landmarks(image, results.pose_landmarks,
                              mp_holistic.POSE_CONNECTIONS, hand_style, hand_line_style)


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
        lh = np.zeros(21 * 3)   # 63
        # print('未检测到右手!')

    if results.face_landmarks:
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten()  # 提取面部特征信息
        # print(face.shape)
    else:
        face = np.zeros(468 * 3)   # 1404

    if results.pose_landmarks:
        # 提取姿势特征
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten()
    else:
        pose = np.zeros(33 * 4)   # 132

    return np.concatenate([lh, rh, pose])


def get_datasets(data_path):
    """
    处理数据建立标签与特征信息的映射 划分数据集
    :param data_path: 关键点信息存放路径
    :return: 划分的训练集和测试集
    """
    label_map = {label: num for num, label in enumerate(actions)}  # 标签映射为数字
    sequences, labels = [], []   # sequences:每个动作的帧集合  label:对应动作标签
    for action in actions:
        for sequence in range(num_action):
            window = []
            for frame_sequence in range(num_frame):   # 获取每个动作所有的帧
                res = np.load(os.path.join(data_path, action, str(sequence),
                                           "{}.npy".format(frame_sequence)))
                window.append(res)

            # 加入帧及对应的标签
            sequences.append(window)
            labels.append(label_map[action])

    # print(sequences)

    x = np.array(sequences).astype('float32')
    # print("关键点数据格式:", x.shape)     # input_shape
    y = np.eye(len(actions))[labels].astype('float32')   # 将标签转为one-hot形式
    # print("标签格式:", y.shape)

    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=test_size)  # 划分训练集与测试集

    return torch.from_numpy(train_x), torch.from_numpy(test_x), torch.from_numpy(train_y), torch.from_numpy(test_y)


def train_model():
    """
    训练模型
    :return:model
    """
    model_type = input("选择模型 1:LSTM 2:GRU")
    if model_type == '1':
        model = LSTM(input_size=features_num, batch=num_frame)
    elif model_type == '2':
        model = GRU(input_size=features_num, batch=num_frame)
    else:
        model = GRU(input_size=features_num, batch=num_frame)

    loss_f = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    t_loss = []
    t_acc = []
    e_loss = []
    e_acc = []
    x = range(1, epochs + 1)

    bar = tqdm(range(epochs))
    for epoch in bar:
        loss = 0
        acc = 0
        for step, data in enumerate(train_data_loader):
            inputs, labels = data
            train_y_predict = model(inputs)
            train_loss = loss_f(train_y_predict, labels)
            loss += train_loss.detach().numpy()

            train_y_predict = np.argmax(train_y_predict.detach().numpy(), axis=1).tolist()
            train_y_true = np.argmax(labels, axis=1).tolist()
            train_acc = accuracy_score(train_y_predict, train_y_true)  # 正确率
            acc += train_acc

            train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            bar.set_description(
                'Train:Epoch{}/Step{}, Loss:{:.5f}, Acc:{:.3f}'.format(epoch + 1, step + 1, train_loss.item(),
                                                                       train_acc))

        t_loss.append(loss / (step + 1))
        t_acc.append(acc / (step + 1))

        model.eval()
        loss = 0
        acc = 0
        with torch.no_grad():
            for step, data in enumerate(eval_data_loader):
                inputs, labels = data
                eval_y_predict = model(inputs)
                eval_loss = loss_f(eval_y_predict, labels)
                loss += eval_loss.detach().numpy()

                eval_y_predict = np.argmax(eval_y_predict.detach().numpy(), axis=1).tolist()
                eval_y_true = np.argmax(labels, axis=1).tolist()
                eval_acc = accuracy_score(eval_y_predict, eval_y_true)  # 正确率
                acc += eval_acc
                bar.set_description(
                    'Eval:Epoch{}/Step{}, Loss:{:.5f}, Acc:{:.3f}'.format(epoch + 1, step + 1, eval_loss.item(),
                                                                          eval_acc))

        e_loss.append(loss / (step + 1))
        e_acc.append(acc / (step + 1))
        bar.set_description(
            'Epoch{}, Train_Loss/Eval_Loss:{:.5f}, {:.5f}, Train_Acc/Eval_Acc:{:.3f}, {:.3f}'.format(epoch + 1,
                                                                                                     t_loss[-1],
                                                                                                     e_loss[-1],
                                                                                                     t_acc[-1],
                                                                                                     e_acc[-1]))

    fig, ax = plt.subplots()
    ax.plot(x, t_loss, label='train_loss')
    ax.plot(x, e_loss, label='eval_loss')
    ax.plot(x, t_acc, label='train_acc')
    ax.plot(x, e_acc, label='eval_acc')
    ax.legend()  # 自动检测要在图例中显示的元素，并且显示
    plt.savefig('logs/dynamic_train_log_gru_126.png')
    plt.show()  # 图形可视化

    return model


# def make_metric(my_model):
#     """
#     混淆矩阵及正确率
#     :param my_model: 训练好的模型
#     :return: 无
#     """
#     y_predict = my_model.predict(x_test)
#     y_predict = np.argmax(y_predict, axis=1).tolist()
#     print("预测结果:", y_predict)
#     y_true = np.argmax(y_test, axis=1).tolist()
#     print("真实结果:", y_true)
#
#     confusion = confusion_matrix(y_true, y_predict)  # 混淆矩阵
#     plt.imshow(confusion, cmap=plt.cm.Blues)  # 绘制混淆矩阵
#     indices = range(len(confusion))  # 刻度
#     category = ['Body', 'Healthy', 'Progress', 'Study', 'Well', 'Work']
#     plt.xticks(indices, category, rotation=320)
#     plt.yticks(indices, category)
#     plt.colorbar()  # 设置渐变色
#     for first_index in range(len(confusion)):
#         sum = 0  # 计算每一类被预测为各类的占比
#         for second_index in range(len(confusion[first_index])):
#             sum += confusion[first_index][second_index]
#         for second_index in range(len(confusion[first_index])):
#             plt.text(first_index, second_index, round(confusion[first_index][second_index] / sum, 2))  # 保留两位小数
#     plt.show()
#
#     acc = accuracy_score(y_true, y_predict)  # 正确率
#     print("正确率:", acc)


def real_detect(my_model):
    """
    实时检测
    :return: 无
    """
    # print("开始实时检测!")
    sequences = []  # 收集一个动作所包含的所有帧
    sentences = []  # 收集每次的预测结果
    threshold = 0.75  # 可信度

    real_cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    real_cap.set(3, 1280)
    real_cap.set(4, 720)
    print("打开摄像头无错误")
    if not real_cap.isOpened():
        print("镜头未打开")

    with mp_holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75) as holistic:
        while real_cap.isOpened():
            start_time = time.time()
            # 捕捉帧识别关键点并绘制
            ret, frame = real_cap.read()
            if not ret:  # 未检测到画面结束循环
                print("有错误")
                break

            frame = cv2.flip(frame, 1)  # 图像做镜像翻转左右手互换（符合习惯）
            image, results = hand_detection(frame, holistic)
            draw_landmarks(image, results, hand_Style, handLine_Style)
            # 提取关键点信息
            keypoint = extract_keypoint(results)
            sequences.append(keypoint)
            # sequence.insert(0, keypoint)
            sequences = sequences[-num_frame:]  # 取后40帧做识别

            # 收集够40帧进行识别
            # 滑动窗口
            if len(sequences) == num_frame:
                test = np.expand_dims(sequences, axis=0).astype('float32')
                res = my_model(torch.from_numpy(test)).detach().numpy()[0]
                output = np.argmax(res)
                if res[output] > threshold:  # 最大概率大于可信度判断为有效预测
                    # 预测集合非空,若本次预测结果等同上次则不重复添加
                    if len(sentences) > 0:
                        if actions[output] != sentences[-1]:
                            sentences.append(actions[output])
                    # 预测集空直接添加
                    else:
                        sentences.append(actions[output])
                    sequences.clear()   # 预测结果准确率大于0.75清空上次帧列表

            # 输出预测结果
            cv2.rectangle(image, (0, 0), (1280, 40), (255, 255, 255), -1)
            cv2.putText(image, '  '.join(sentences), (3, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
            if len(sentences) > 6:         # 取最新的6次预测结果
                sentences = sentences[-6:]

            end_time = time.time()
            fps = int(1 / (end_time - start_time))
            cv2.putText(image, "FPS:{}".format(fps), (0, 600),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow('real_detect', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):   # 按下q退出循环
                break

        # 关闭镜头
        real_cap.release()
        cv2.destroyAllWindows()


def ved_detect(my_model):
    """
    读取视频检测
    :return:
    """
    sequences = []  # 收集一个动作所包含的所有帧
    sentences = []  # 收集每次的预测结果
    threshold = 0.75  # 可信度

    video = cv2.VideoCapture(ved_path)
    video.set(3, 1280)
    video.set(4, 720)

    with mp_holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75) as holistic:
        while video.isOpened():
            # 捕捉帧识别关键点并绘制
            start_time = time.time()
            ret, frame = video.read()
            if not ret:
                break

            image, results = hand_detection(frame, holistic)
            draw_landmarks(image, results, hand_Style, handLine_Style)
            # 提取关键点信息
            keypoint = extract_keypoint(results)
            sequences.append(keypoint)
            # sequence.insert(0, keypoint)
            sequences = sequences[-num_frame:]  # 取后40帧做识别

            # 收集够40帧进行识别
            if len(sequences) == num_frame:
                test = np.expand_dims(sequences, axis=0).astype('float32')
                res = my_model(torch.from_numpy(test)).detach().numpy()[0]
                output = np.argmax(res)
                if res[output] > threshold:  # 最大概率大于可信度判断为有效预测
                    # 预测集合非空,若本次预测结果等同上次则不重复添加
                    if len(sentences) > 0:
                        if actions[output] != sentences[-1]:
                            sentences.append(actions[output])
                    # 预测集空直接添加
                    else:
                        sentences.append(actions[output])
                    sequences.clear()  # 预测结果准确率大于0.75清空上次帧列表

            # 输出预测结果
            cv2.rectangle(image, (0, 0), (1280, 40), (255, 255, 255), -1)
            cv2.putText(image, '  '.join(sentences), (3, 30), cv2.FONT_HERSHEY_SIMPLEX
                        , 1, (0, 0, 0), 2, cv2.LINE_AA)
            if len(sentences) > 6:  # 取最新的6次预测结果
                sentences = sentences[-6:]

            end_time = time.time()
            fps = int(1 / (end_time - start_time))
            cv2.putText(image, "FPS:{}".format(fps), (0, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            cv2.imshow('vedio_detect', image)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

        video.release()
        cv2.destroyAllWindows()


def choose_function(my_model):
    """
    功能选择
    :return:
    """
    judge_fuc = input("实时检测:1 视频检测:2")
    if judge_fuc == '1':
        real_detect(my_model)  # 实时预测
    elif judge_fuc == '2':
        ved_detect(my_model)
    else:
        print("输入错误重新输入")


if __name__ == "__main__":
    DATA_PATH = './datasets/dynamic_numpy_data_126'   # 关键点信息路径
    IMAGE_PATH = './dynamic_photo_data'
    save_model_path = 'models/dynamic_model_gru_126'
    ved_path = 'test_vedio/test2.mp4'

    actions = np.array(['Work', 'Well', 'Study', 'Progress', 'Body', 'Healthy'])   # 动作集合
    num_action = 150  # 每个动作样本数
    num_frame = 40  # 每个样本帧数
    test_size = 0.1  # 测试集占比
    batch_size = 16  # 批次大小
    epochs = 100  # 训练轮数
    features_num = 126

    mp_holistic = mp.solutions.holistic  # 全身特征检测模型
    mp_drawing = mp.solutions.drawing_utils  # 绘制关键点

    hand_Style = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=5)   # 关键点样式
    handLine_Style = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=10)   # 连线样式

    x_train, x_test, y_train, y_test = get_datasets(DATA_PATH)
    train_data = TensorDataset(x_train, y_train)
    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    eval_data = TensorDataset(x_test, y_test)
    eval_data_loader = DataLoader(eval_data, batch_size=1, shuffle=True)

    is_train = input('是否训练模型y/n')
    if is_train == 'y':
        new_model = train_model()
        # make_metric(new_model)  # 测试集准确率
        is_save = input('是否保存模型y/n')
        if is_save == 'y':
            torch.save(new_model.state_dict(), save_model_path)

    test_model = LSTM(input_size=258, batch=40)
    test_model.load_state_dict(torch.load('models/dynamic_model_lstm_258'))
    test_model.eval()

    choose_function(test_model)
