"""
静态手语识别
"""
import os
import cv2  # opencv_python:4.5.2.52
import matplotlib.pyplot as plt
import mediapipe as mp  # 版本:0.9.0.1
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import accuracy_score  # scikit_learn:0.24.1
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class LSTM(nn.Module):
    """
        Parameters：
        - input_size: feature size
        - hidden_size: number of hidden units
        - output_size: number of output
        - num_layers: layers of LSTM to stack
    """

    def __init__(self, input_size, hidden_size1=64, hidden_size2=64, output_size=6, num_layers=1, batch=1):
        super().__init__()

        self.lstm1 = nn.LSTM(input_size, hidden_size1, num_layers)
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, num_layers)
        self.linear = nn.Linear(hidden_size2 * batch, output_size)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout(x)
        x, _ = self.lstm2(x)
        x = self.dropout(x)
        s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        x = x.view(-1, b * h)
        x = self.linear(x)
        x = nn.functional.softmax(x, dim=1)

        return x


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


def get_datasets(data_path):
    """
    处理数据建立标签与特征信息的映射 划分数据集
    :param data_path: 关键点信息存放路径
    :return: 划分的训练集和测试集
    """
    label_map = {label: num for num, label in enumerate(actions)}  # 标签映射为数字
    sequences, labels = [], []  # sequences:每个动作的帧集合  label:对应动作标签
    for action in actions:
        for sequence in range(num_action):
            window = []
            for frame_sequence in range(num_frame):  # 获取每个动作所有的帧
                res = np.load(os.path.join(data_path, action, str(sequence),
                                           "{}.npy".format(frame_sequence)))
                window.append(res)

            # 加入帧及对应的标签
            sequences.append(window)
            labels.append(label_map[action])

    # print(sequences)
    x = np.array(sequences).astype('float32')
    # print("关键点数据格式:", x.shape)  # input_shape
    y = np.eye(len(actions))[labels].astype('float32')
    # print("标签格式:", y.shape)
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=test_size)  # 划分训练集与测试集

    return torch.from_numpy(train_x), torch.from_numpy(test_x), torch.from_numpy(train_y), torch.from_numpy(test_y)


def train_model():
    """
    小批次训练模型
    :return:model
    """
    model = LSTM(input_size=num_feature, hidden_size1=64, hidden_size2=64, output_size=6, num_layers=1, batch=num_frame)
    loss_f = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    t_loss = []
    t_acc = []
    e_loss = []
    e_acc = []
    x = range(1, epochs+1)

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
            bar.set_description('Train:Epoch{}/Step{}, Loss:{:.5f}, Acc:{:.3f}'.format(epoch + 1, step+1, train_loss.item(), train_acc))

        t_loss.append(loss/(step+1))
        t_acc.append(acc/(step+1))

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
                bar.set_description('Eval:Epoch{}/Step{}, Loss:{:.5f}, Acc:{:.3f}'.format(epoch + 1, step+1, eval_loss.item(), eval_acc))

        e_loss.append(loss/(step+1))
        e_acc.append(acc/(step+1))
        bar.set_description('Epoch{}, Train_Loss/Eval_Loss:{:.5f}, {:.5f}, Train_Acc/Eval_Acc:{:.3f}, {:.3f}'.format(epoch + 1, t_loss[-1], e_loss[-1], t_acc[-1], e_acc[-1]))

    fig, ax = plt.subplots()
    ax.plot(x, t_loss, label='train_loss')
    ax.plot(x, e_loss, label='eval_loss')
    ax.plot(x, t_acc, label='train_acc')
    ax.plot(x, e_acc, label='eval_acc')
    ax.legend()  # 自动检测要在图例中显示的元素，并且显示
    plt.savefig('logs/static_train_log_lstm_126.png')
    plt.show()  # 图形可视化

    return model


# def make_metric(my_model):
#     """
#     混淆矩阵及正确率
#
#     :param my_model: 训练好的模型
#     :return: 无
#     """
#     y_predict = my_model(x_test)
#     y_predict = np.argmax(y_predict.detach().numpy(), axis=1).tolist()
#     print("预测结果:", y_predict)
#     y_true = np.argmax(y_test, axis=1).tolist()
#     print("真实结果:", y_true)
#
#     confusion = confusion_matrix(y_true, y_predict)  # 混淆矩阵
#     plt.imshow(confusion, cmap="Blues")  # 绘制混淆矩阵
#     indices = range(len(confusion))  # 刻度
#     category = actions.tolist()
#     plt.xticks(indices, category, rotation=320)
#     plt.yticks(indices, category)
#     plt.colorbar()  # 设置渐变色
#     for first_index in range(len(confusion)):
#         sums = 0  # 计算每一类被预测为各类的占比
#         for second_index in range(len(confusion[first_index])):
#             sums += confusion[first_index][second_index]
#         for second_index in range(len(confusion[first_index])):
#             plt.text(first_index, second_index, round(confusion[first_index][second_index] / sums, 2))  # 保留两位小数
#     plt.show()
#
#     acc = accuracy_score(y_true, y_predict)  # 正确率
#     print("正确率:", acc)


if __name__ == "__main__":
    DATA_PATH = './datasets/static_numpy_data'  # 关键点信息路径
    IMAGE_PATH = './static_photo_data'
    save_model_path = 'models/static_model_lstm_126'

    actions = np.array(['I', 'You', 'Good', 'Salute', 'Cow', 'Wait'])  # 动作集合
    num_action = 100  # 每个动作样本数
    num_frame = 1  # 每个样本帧数
    epochs = 150  # 训练轮数
    num_feature = 126
    test_size = 0.1  # 测试集占比
    batch_size = 16  # 批次大小

    mp_holistic = mp.solutions.holistic  # 全身特征检测模型
    mp_drawing = mp.solutions.drawing_utils  # 绘制关键点

    hand_Style = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=5)  # 关键点样式
    handLine_Style = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=10)  # 连线样式

    x_train, x_test, y_train, y_test = get_datasets(DATA_PATH)
    train_data = TensorDataset(x_train, y_train)
    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    eval_data = TensorDataset(x_test, y_test)
    eval_data_loader = DataLoader(eval_data, batch_size=1, shuffle=True)

    lstm = train_model()
    # make_metric(lstm)

    is_save = input('是否保存模型y/n')
    if is_save == 'y':
        torch.save(lstm.state_dict(), save_model_path)
