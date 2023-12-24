"""
图形用户界面
"""
import sys
import time
import cv2
import numpy as np
import mediapipe as mp    # 版本:0.9.0.1
import gradio as gr
import torch
import torch.nn as nn


class LSTM1(nn.Module):
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


class LSTM2(nn.Module):
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


class IsRun:
    def __init__(self):
        self.is_ved = 0
        self.is_real = 0

    def revise_is_ved(self):
        self.is_ved = (self.is_ved + 1) % 2

    def revise_is_real(self):
        self.is_real = (self.is_real + 1) % 2


def predict(image_path):
    """
    静态图片检测
    :param image_path: 图片路径
    :return: 无
    """
    image = cv2.imread(image_path)
    with mp_holistic.Holistic(static_image_mode=True, min_detection_confidence=0.75, min_tracking_confidence=0.75) as holistic:
        window = []
        sequence = []
        image, results = hand_detection(image, holistic)
        draw_landmarks(image, results, hand_Style, handLine_Style)   # 只绘制一次关键点
        keypoint = extract_keypoint(results, 'static')
        window.append(keypoint)   # 重复收集静态图片特征信息,对应一个动作的所有帧

        sequence.append(window)  # 一个动作帧的集合
        test = np.array(sequence).astype('float32')
        res = static_model(torch.from_numpy(test)).detach().numpy()[0]  # 预测
        output = np.argmax(res)

        return static_actions[output]


def hand_detection(image, my_model):
    """
    手部检测
    :param image:实时获取的图像
    :param my_model:检测模型（holistic:整体检测）
    :return:返回图像和关键点模型检测信息
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR格式转为RGB
    image.flags.writeable = False
    results = my_model.process(image)  # 处理RGB图像
    image.flags.writeable = True
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


def extract_keypoint(results, judge):
    """
    提取图片关键点信息
    :param judge: 静态检测或动态检测
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
        lh = np.zeros(21 * 3)  # 63
        # print('未检测到右手!')

    if results.face_landmarks:
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten()  # 提取面部特征信息
        # print(face.shape)
    else:
        face = np.zeros(468 * 3)  # 1404

    if results.pose_landmarks:
        # 提取姿势特征
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten()
    else:
        pose = np.zeros(33 * 4)  # 132

    if judge == 'static':
        return np.concatenate([lh, rh])   # 126

    elif judge == 'dynamic':
        return np.concatenate([lh, rh, pose])

    # 默认动态提取模式
    else:
        return np.concatenate([lh, rh, pose])


def load_model(judge):
    """
    加载已有模型(默认动态模型)
    :return:
    """
    if judge == 'static':
        model = LSTM1(input_size=126, batch=1)
        model.load_state_dict(torch.load('models/static_model_lstm_126'))
        model.eval()

    elif judge == 'dynamic':
        model = LSTM2(input_size=258, batch=40)
        model.load_state_dict(torch.load('models/dynamic_model_lstm_258'))
        model.eval()

    else:
        model = GRU(input_size=258, batch=40)
        model.load_state_dict(torch.load('models/dynamic_model_gru_258'))
        model.eval()

    return model


def transform(results):
    """
    结果转换为中文
    :param results:
    :return:
    """
    final = []
    for result in results:
        if result == 'Work':
            final.append('工作 ')

        elif result == 'Well':
            final.append('顺利 ')

        elif result == 'Study':
            final.append('学习 ')

        elif result == 'Progress':
            final.append('进步 ')

        elif result == 'Body':
            final.append('身体 ')

        elif result == 'Healthy':
            final.append('健康 ')

    return final


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    demo.title = "手语识别"

    with gr.Tab("静态手部识别"):
        with gr.Row():
            static_inputs = gr.Image(label="图片", height=700, sources=["upload", "webcam"], type="filepath")
            with gr.Accordion(label="模型反馈"):
                is_static_flip = gr.Slider(0, 1, step=1, label="镜像")
                with gr.Row():
                    static_hands_num = gr.Textbox(label="手部数目", interactive=False)
                with gr.Row():
                    with gr.Column():
                        static_l_hands_xpos = gr.Textbox(label="左手x坐标", interactive=False)
                        static_l_hands_ypos = gr.Textbox(label="左手y坐标", interactive=False)
                    with gr.Column():
                        static_r_hands_xpos = gr.Textbox(label="右手x坐标", interactive=False)
                        static_r_hands_ypos = gr.Textbox(label="右手y坐标", interactive=False)
                static_result = gr.Textbox(label="静态预测结果", interactive=False, lines=2)

        with gr.Row():
            static_start = gr.Button("启动检测")
            static_clear = gr.Button("清空内容")

        def static_detect(image_path, flip):
            result = predict(image_path)

            # 获取手部数目和左右手位置
            image = cv2.imread(image_path)
            with mp_holistic.Holistic(static_image_mode=True, min_detection_confidence=0.75,
                                      min_tracking_confidence=0.75) as holistic:
                image, results = hand_detection(image, holistic)
                draw_landmarks(image, results, hand_Style, handLine_Style)  # 只绘制一次关键点

                num = 0
                if results.left_hand_landmarks:  # 判断是否检测到左手
                    num += 1
                    for i, lm in enumerate(results.left_hand_landmarks.landmark):  # 手部21关键点位置
                        x_pos = int(lm.x * image.shape[1])
                        y_pos = int(lm.y * image.shape[0])
                        cv2.putText(image, str(i), (x_pos - 25, y_pos + 5),  # 标注各关键点
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        # 取9号关键点(手中部)作为手部位置
                        if i == 9:
                            # flip=0未进行镜像翻转 左手对应左手
                            if flip == 0:
                                text3 = "{}".format(x_pos)
                                text4 = "{}".format(y_pos)
                            # flip=1进行镜面反转 左手对应右手
                            elif flip == 1:
                                text5 = "{}".format(x_pos)
                                text6 = "{}".format(y_pos)

                else:
                    if flip == 0:
                        text3 = ""
                        text4 = ""
                    # flip=1进行镜面反转 左手对应右手
                    elif flip == 1:
                        text5 = ""
                        text6 = ""

                if results.right_hand_landmarks:  # 右手
                    num += 1
                    for i, lm in enumerate(results.right_hand_landmarks.landmark):
                        x_pos = int(lm.x * image.shape[1])
                        y_pos = int(lm.y * image.shape[0])
                        cv2.putText(image, str(i), (x_pos - 25, y_pos + 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        if i == 9:
                            if flip == 0:
                                text5 = "{}".format(x_pos)
                                text6 = "{}".format(y_pos)

                            # flip=1进行镜面反转 左手对应右手
                            elif flip == 1:
                                text3 = "{}".format(x_pos)
                                text4 = "{}".format(y_pos)

                else:
                    if flip == 0:
                        text5 = ""
                        text6 = ""

                    # flip=1进行镜面反转 左手对应右手
                    elif flip == 1:
                        text3 = ""
                        text4 = ""

            return result, image, str(num), text3, text4, text5, text6

        def reset():
            return None, "", "", "", "", "", ""

        static_start.click(
            fn=static_detect,
            inputs=[static_inputs, is_static_flip],
            outputs=[static_result, static_inputs, static_hands_num, static_l_hands_xpos, static_l_hands_ypos, static_r_hands_xpos, static_r_hands_ypos]
        )
        static_clear.click(
            fn=reset,
            inputs=None,
            outputs=[static_inputs, static_hands_num, static_l_hands_xpos, static_l_hands_ypos, static_r_hands_xpos, static_r_hands_ypos, static_result]
        )

    with gr.Tab("动态手语识别"):
        with gr.Tab("视频手语识别"):
            with gr.Row():
                ved_inputs = gr.Video(label="待测视频", height=700, sources=["upload"])
                ved_outputs = gr.Image(label="识别结果", height=700)

            with gr.Accordion(label="模型反馈"):
                with gr.Row():
                    with gr.Column():
                        ved_fps = gr.Textbox(label="实时帧数", interactive=False)
                    with gr.Column():
                        ved_hands_num = gr.Textbox(label="手部数目", interactive=False)
                with gr.Row():
                    with gr.Column():
                        ved_l_hands_xpos = gr.Textbox(label="左手x坐标", interactive=False)
                        ved_l_hands_ypos = gr.Textbox(label="左手y坐标", interactive=False)
                    with gr.Column():
                        ved_r_hands_xpos = gr.Textbox(label="右手x坐标", interactive=False)
                        ved_r_hands_ypos = gr.Textbox(label="右手y坐标", interactive=False)
                ved_result = gr.Textbox(label="动态预测结果", interactive=False, lines=4)

            with gr.Row():
                ved_start = gr.Button("启动检测")
                ved_close = gr.Button("停止检测")
                ved_clear = gr.Button("清空内容")

            def ved_detect(inputs):
                sequences = []  # 收集一个动作所包含的所有帧
                sentences = []  # 收集每次的预测结果
                threshold = 0.7  # 可信度

                cap = cv2.VideoCapture(inputs)
                is_run.revise_is_ved()
                with mp_holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75) as holistic:
                    while cap.isOpened():
                        start_time = time.time()
                        ret, image = cap.read()
                        if not ret:
                            break
                        if is_run.is_ved == 0:
                            break
                        image, results = hand_detection(image, holistic)
                        draw_landmarks(image, results, hand_Style, handLine_Style)
                        num = 0
                        if results.left_hand_landmarks:  # 判断是否检测到左手
                            num += 1
                            for i, lm in enumerate(results.left_hand_landmarks.landmark):  # 手部21关键点位置
                                x_pos = int(lm.x * image.shape[1])
                                y_pos = int(lm.y * image.shape[0])
                                # cv2.putText(image, str(i), (x_pos - 25, y_pos + 5),  # 标注各关键点
                                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                                # 取9号关键点(手中部)作为手部位置
                                if i == 9:
                                    text3 = "{}".format(x_pos)
                                    text4 = "{}".format(y_pos)
                        else:
                            text3 = ""
                            text4 = ""
                        if results.right_hand_landmarks:  # 右手
                            num += 1
                            for i, lm in enumerate(results.right_hand_landmarks.landmark):
                                x_pos = int(lm.x * image.shape[1])
                                y_pos = int(lm.y * image.shape[0])
                                # cv2.putText(image, str(i), (x_pos - 25, y_pos + 5),
                                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                                if i == 9:
                                    text5 = "{}".format(x_pos)
                                    text6 = "{}".format(y_pos)
                        else:
                            text5 = ""
                            text6 = ""
                        # 提取关键点信息
                        keypoint = extract_keypoint(results, 'dynamic')
                        sequences.append(keypoint)
                        sequences = sequences[-num_frame:]  # 取后40帧做识别
                        # 收集够40帧进行识别
                        if len(sequences) == num_frame:
                            test = np.expand_dims(sequences, axis=0).astype('float32')
                            res = dynamic_model(torch.from_numpy(test)).detach().numpy()[0]
                            output = np.argmax(res)
                            print(res[output])
                            if res[output] > threshold:  # 最大概率大于可信度判断为有效预测
                                # 预测集合非空,若本次预测结果等同上次则不重复添加
                                if len(sentences) > 0:
                                    if dynamic_actions[output] != sentences[-1]:
                                        sentences.append(dynamic_actions[output])
                                # 预测集空直接添加
                                else:
                                    sentences.append(dynamic_actions[output])
                                sequences.clear()  # 预测结果准确率大于0.75清空上次帧列表
                        end_time = time.time()
                        fps = int(1 / (end_time - start_time))

                        yield image, fps, str(num), text3, text4, text5, text6, ",".join(sentences)
                    cap.release()
                    cv2.destroyAllWindows()

            def ved_reset():
                return None, None, "", "", "", "", "", "", ""

            def ved_stop():
                is_run.revise_is_ved()

            ved_start.click(
                fn=ved_detect,
                inputs=[ved_inputs],
                outputs=[ved_outputs, ved_fps, ved_hands_num, ved_l_hands_xpos, ved_l_hands_ypos, ved_r_hands_xpos, ved_r_hands_ypos, ved_result]
            )
            ved_clear.click(
                fn=ved_reset,
                inputs=None,
                outputs=[ved_inputs, ved_outputs, ved_fps, ved_hands_num, ved_l_hands_xpos, ved_l_hands_ypos, ved_r_hands_xpos, ved_r_hands_ypos, ved_result]
            )
            ved_close.click(
                fn=ved_stop,
                inputs=None,
                outputs=None
            )

        with gr.Tab("实时手语识别"):
            with gr.Row():
                real_outputs = gr.Image(label="识别结果", height=700)

            with gr.Accordion(label="模型反馈"):
                with gr.Row():
                    with gr.Column():
                        real_fps = gr.Textbox(label="实时帧数", interactive=False)
                    with gr.Column():
                        real_hands_num = gr.Textbox(label="手部数目", interactive=False)
                with gr.Row():
                    with gr.Column():
                        real_l_hands_xpos = gr.Textbox(label="左手x坐标", interactive=False)
                        real_l_hands_ypos = gr.Textbox(label="左手y坐标", interactive=False)
                    with gr.Column():
                        real_r_hands_xpos = gr.Textbox(label="右手x坐标", interactive=False)
                        real_r_hands_ypos = gr.Textbox(label="右手y坐标", interactive=False)
                real_result = gr.Textbox(label="动态预测结果", interactive=False, lines=4)

            with gr.Row():
                real_start = gr.Button("启动检测")
                real_close = gr.Button("停止检测")
                real_clear = gr.Button("清空内容")

            def real_detect():
                sequences = []  # 收集一个动作所包含的所有帧
                sentences = []  # 收集每次的预测结果
                threshold = 0.7  # 可信度

                cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                is_run.revise_is_real()
                with mp_holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75) as holistic:
                    while cap.isOpened():
                        start_time = time.time()
                        ret, image = cap.read()
                        if not ret:
                            break
                        if is_run.is_real == 0:
                            break
                        image, results = hand_detection(image, holistic)
                        draw_landmarks(image, results, hand_Style, handLine_Style)
                        num = 0
                        if results.left_hand_landmarks:  # 判断是否检测到左手
                            num += 1
                            for i, lm in enumerate(results.left_hand_landmarks.landmark):  # 手部21关键点位置
                                x_pos = int(lm.x * image.shape[1])
                                y_pos = int(lm.y * image.shape[0])
                                # cv2.putText(image, str(i), (x_pos - 25, y_pos + 5),  # 标注各关键点
                                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                                # 取9号关键点(手中部)作为手部位置
                                if i == 9:
                                    text3 = "{}".format(x_pos)
                                    text4 = "{}".format(y_pos)
                        else:
                            text3 = ""
                            text4 = ""

                        if results.right_hand_landmarks:  # 右手
                            num += 1
                            for i, lm in enumerate(results.right_hand_landmarks.landmark):
                                x_pos = int(lm.x * image.shape[1])
                                y_pos = int(lm.y * image.shape[0])
                                # cv2.putText(image, str(i), (x_pos - 25, y_pos + 5),
                                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                                if i == 9:
                                    text5 = "{}".format(x_pos)
                                    text6 = "{}".format(y_pos)
                        else:
                            text5 = ""
                            text6 = ""

                        # 提取关键点信息
                        keypoint = extract_keypoint(results, 'dynamic')
                        sequences.append(keypoint)
                        sequences = sequences[-num_frame:]  # 取后40帧做识别
                        # 收集够40帧进行识别
                        if len(sequences) == num_frame:
                            test = np.expand_dims(sequences, axis=0).astype('float32')
                            res = dynamic_model(torch.from_numpy(test)).detach().numpy()[0]
                            output = np.argmax(res)
                            if res[output] > threshold:  # 最大概率大于可信度判断为有效预测
                                # 预测集合非空,若本次预测结果等同上次则不重复添加
                                if len(sentences) > 0:
                                    if dynamic_actions[output] != sentences[-1]:
                                        sentences.append(dynamic_actions[output])
                                # 预测集空直接添加
                                else:
                                    sentences.append(dynamic_actions[output])
                                sequences.clear()  # 预测结果准确率大于0.75清空上次帧列表
                                # self.voice.emit(voice_sentences)
                        end_time = time.time()
                        fps = int(1 / (end_time - start_time))

                        yield image, fps, str(num), text3, text4, text5, text6, ",".join(sentences)

                    cap.release()
                    cv2.destroyAllWindows()

            def real_reset():
                return None, "", "", "", "", "", "", ""

            def real_stop():
                is_run.revise_is_real()

            real_start.click(
                fn=real_detect,
                inputs=None,
                outputs=[real_outputs, real_fps, real_hands_num, real_l_hands_xpos, real_l_hands_ypos,
                         real_r_hands_xpos, real_r_hands_ypos, real_result]
            )

            real_clear.click(
                fn=real_reset,
                inputs=None,
                outputs=[real_outputs, real_fps, real_hands_num, real_l_hands_xpos, real_l_hands_ypos,
                         real_r_hands_xpos, real_r_hands_ypos, real_result]
            )

            real_close.click(
                fn=real_stop,
                inputs=None,
                outputs=None
            )


if __name__ == "__main__":
    static_actions = np.array(['I', 'You', 'Good', 'Salute', 'Cow', 'Wait'])
    dynamic_actions = np.array(['Work', 'Well', 'Study', 'Progress', 'Body', 'Healthy'])

    mp_holistic = mp.solutions.holistic  # 全身特征检测模型
    mp_drawing = mp.solutions.drawing_utils  # 绘制关键点

    hand_Style = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=3)  # 关键点样式
    handLine_Style = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)  # 连线样式

    num_frame = 40  # 每个样本帧数

    is_run = IsRun()

    static_model = load_model('static')
    dynamic_model = load_model('dynamic')

    demo.queue().launch()



