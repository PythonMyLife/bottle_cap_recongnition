# -*- coding: UTF-8 -*-
from tkinter import Tk, Canvas, Button, filedialog, StringVar, Label
from PIL import Image, ImageTk
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
import imutils
import cv2


def resize(w_box, h_box, pil_image):
    w, h = pil_image.size  # 获取图像的原始大小
    f1 = 1.0 * w_box / w
    f2 = 1.0 * h_box / h
    factor = min([f1, f2])
    width = int(w * factor)
    height = int(h * factor)
    return pil_image.resize((width, height), Image.ANTIALIAS)


# 创建一个颜色标签类
class ColorLabeler:
    def __init__(self):
        # 初始化一个颜色词典
        colors = OrderedDict({
            "red": (255, 0, 0),
            "yellow": (255, 165, 0),
            "blue": (0, 0, 255)})

        # 为LAB图像分配空间
        self.lab = np.zeros((len(colors), 1, 3), dtype="uint8")
        self.colorNames = []

        # 循环遍历颜色词典
        for (i, (name, rgb)) in enumerate(colors.items()):
            # 进行参数更新
            self.lab[i] = rgb
            self.colorNames.append(name)

        # 进行颜色空间的变换
        self.lab = cv2.cvtColor(self.lab, cv2.COLOR_RGB2LAB)

    def label(self, image, c):
        # 根据轮廓构造一个mask，然后计算mask区域的平均值
        mask = np.zeros(image.shape[:2], dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)
        mask = cv2.erode(mask, None, iterations=2)
        mean = cv2.mean(image, mask=mask)[:3]

        # 初始化最小距离
        minDist = (np.inf, None)

        # 遍历已知的LAB颜色值
        for (i, row) in enumerate(self.lab):
            # 计算当前l*a*b*颜色值与图像平均值之间的距离
            d = dist.euclidean(row[0], mean)

            # 如果当前的距离小于最小的距离，则进行变量更新
            if d < minDist[0]:
                minDist = (d, i)

        # 返回最小距离对应的颜色值
        return self.colorNames[minDist[1]]


# 创建形状检测类
class ShapeDetector:
    def __init__(self):
        pass

    def detect(self, c):
        # 初始化形状名和近似的轮廓
        shape = "unidentified"
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        # 如果当前的轮廓含有4或5个顶点，则其为侧面
        if len(approx) <= 5:
            shape = "left"
        # 否则的话，我们认为它是一个圆
        else:
            shape = "other"
        # 返回形状的名称
        return shape


class BottleCapRecognition:
    def __init__(self, root):
        self.root = root
        self.root.title("瓶盖识别")
        self.root.geometry('800x640')
        self.root.resizable(0, 0)
        self.root_frame()

    def root_frame(self):
        # canvas1为左侧操作框，canvas2为右侧上部图像框，canvas3为右侧下部图像框
        self.canvas1 = Canvas(self.root, bg="White", width=160, height=640)
        self.canvas1.place(x=0, y=0)
        self.canvas2 = Canvas(self.root, bg="White", width=640, height=320)
        self.canvas2.place(x=160, y=0)
        self.canvas3 = Canvas(self.root, bg="White", width=640, height=320)
        self.canvas3.place(x=160, y=320)

        # 图像选择按钮
        self.button_choose = Button(self.root, text='选择图片', command=lambda: self.choose(self.canvas2))
        self.canvas1.create_window(80, 50, anchor='center', window=self.button_choose)

        # 正面检测按钮
        self.button_front = Button(self.root, text='正面检测', command=lambda: self.front())
        self.canvas1.create_window(80, 100, anchor='center', window=self.button_front)

        # 反面检测按钮
        self.button_back = Button(self.root, text='反面检测', command=lambda: self.back())
        self.canvas1.create_window(80, 150, anchor='center', window=self.button_back)

        # 侧面检测按钮
        self.button_profile = Button(self.root, text='侧面检测', command=lambda: self.profile("all"))
        self.canvas1.create_window(80, 200, anchor='center', window=self.button_profile)

        # 红色侧面检测按钮
        self.button_profile = Button(self.root, text='红色侧面检测', command=lambda: self.profile("red"))
        self.canvas1.create_window(80, 250, anchor='center', window=self.button_profile)

        # 黄色侧面检测按钮
        self.button_profile = Button(self.root, text='黄色侧面检测', command=lambda: self.profile("yellow"))
        self.canvas1.create_window(80, 300, anchor='center', window=self.button_profile)

        # 蓝色侧面检测按钮
        self.button_profile = Button(self.root, text='蓝色侧面检测', command=lambda: self.profile("blue"))
        self.canvas1.create_window(80, 350, anchor='center', window=self.button_profile)

        # 形态检测按钮
        self.button_all = Button(self.root, text='形态检测', command=lambda: self.all())
        self.canvas1.create_window(80, 400, anchor='center', window=self.button_all)

        # 输出语句
        self.output = StringVar()
        # 输出框
        self.text = Label(self.root, textvariable=self.output)
        self.canvas1.create_window(80, 440, anchor='n', window=self.text)

    def choose(self, canvas2):
        file = filedialog.askopenfilename(parent=self.root, initialdir="C:/", title='Choose an image.')
        self.photo = Image.open(file)
        image = self.photo
        w, h = image.size
        if w > 640 or h > 320:
            image = resize(640, 320, image)
        else:
            image = image
        filename = ImageTk.PhotoImage(image)
        canvas2.image = filename
        canvas2.create_image(320, 160, image=filename)

    def show(self, image):
        w, h = image.size
        if w > 640 or h > 320:
            image = resize(640, 320, image)
        else:
            image = image
        file = ImageTk.PhotoImage(image)
        self.canvas3.image = file
        self.canvas3.create_image(320, 160, image=file)

    def front(self):
        long_text = '正面瓶盖的坐标为'
        # 将PIL图像转化为opencv图像
        target = cv2.cvtColor(np.array(self.photo), cv2.COLOR_RGB2BGR)
        iheight, iwidth = target.shape[:2]
        Array = np.zeros((iheight, iwidth))
        MArray160 = np.zeros((iheight, iwidth))
        Array160 = np.zeros((iheight, iwidth))
        MArray220 = np.zeros((iheight, iwidth))
        Array220 = np.zeros((iheight, iwidth))
        MArray210 = np.zeros((iheight, iwidth))
        Array210 = np.zeros((iheight, iwidth))
        template0 = cv2.imread(
            "D:/code/cv/together/bottle_cap_recongnition/bottle_cap_recognition/templete1/1down200.jpg")
        template1 = cv2.imread(
            "D:/code/cv/together/bottle_cap_recongnition/bottle_cap_recognition/templete1/2down200.jpg")
        template2 = cv2.imread(
            "D:/code/cv/together/bottle_cap_recongnition/bottle_cap_recognition/templete1/3down220.jpg")
        template3 = cv2.imread(
            "D:/code/cv/together/bottle_cap_recongnition/bottle_cap_recognition/templete1/4down220.jpg")
        template4 = cv2.imread(
            "D:/code/cv/together/bottle_cap_recongnition/bottle_cap_recognition/templete1/5down210.jpg")
        template5 = cv2.imread(
            "D:/code/cv/together/bottle_cap_recongnition/bottle_cap_recognition/templete1/6down210.jpg")
        template51 = cv2.imread(
            "D:/code/cv/together/bottle_cap_recongnition/bottle_cap_recognition/templete1/7down210.jpg")
        template6 = cv2.imread(
            "D:/code/cv/together/bottle_cap_recongnition/bottle_cap_recognition/templete1/1up200.jpg")
        template7 = cv2.imread(
            "D:/code/cv/together/bottle_cap_recongnition/bottle_cap_recognition/templete1/2up200.jpg")
        template8 = cv2.imread(
            "D:/code/cv/together/bottle_cap_recongnition/bottle_cap_recognition/templete1/3up200.jpg")
        template9 = cv2.imread(
            "D:/code/cv/together/bottle_cap_recongnition/bottle_cap_recognition/templete1/4up200.jpg")
        template10 = cv2.imread(
            "D:/code/cv/together/bottle_cap_recongnition/bottle_cap_recognition/templete1/5up200.jpg")
        template11 = cv2.imread(
            "D:/code/cv/together/bottle_cap_recongnition/bottle_cap_recognition/templete1/6up200.jpg")
        template12 = cv2.imread(
            "D:/code/cv/together/bottle_cap_recongnition/bottle_cap_recognition/templete1/7up220.jpg")
        template13 = cv2.imread(
            "D:/code/cv/together/bottle_cap_recongnition/bottle_cap_recognition/templete1/8up220.jpg")
        template14 = cv2.imread(
            "D:/code/cv/together/bottle_cap_recongnition/bottle_cap_recognition/templete1/9up220.jpg")
        template141 = cv2.imread(
            "D:/code/cv/together/bottle_cap_recongnition/bottle_cap_recognition/templete1/91up220.jpg")
        template142 = cv2.imread(
            "D:/code/cv/together/bottle_cap_recongnition/bottle_cap_recognition/templete1/92up220.jpg")
        template143 = cv2.imread(
            "D:/code/cv/together/bottle_cap_recongnition/bottle_cap_recognition/templete1/93up220.jpg")
        template15 = cv2.imread(
            "D:/code/cv/together/bottle_cap_recongnition/bottle_cap_recognition/templete1/10up210.jpg")
        template16 = cv2.imread(
            "D:/code/cv/together/bottle_cap_recongnition/bottle_cap_recognition/templete1/11up210.jpg")
        template17 = cv2.imread(
            "D:/code/cv/together/bottle_cap_recongnition/bottle_cap_recognition/templete1/12up210.jpg")
        template18 = cv2.imread(
            "D:/code/cv/together/bottle_cap_recongnition/bottle_cap_recognition/templete1/13up200.jpg")

        theight160, twidth160 = template0.shape[:2]
        theight220, twidth220 = template2.shape[:2]
        theight210, twidth210 = template4.shape[:2]
        result0 = cv2.matchTemplate(target, template0, cv2.TM_CCORR_NORMED)
        result1 = cv2.matchTemplate(target, template1, cv2.TM_CCORR_NORMED)
        result2 = cv2.matchTemplate(target, template2, cv2.TM_CCORR_NORMED)
        result3 = cv2.matchTemplate(target, template3, cv2.TM_CCORR_NORMED)
        result4 = cv2.matchTemplate(target, template4, cv2.TM_CCORR_NORMED)
        result5 = cv2.matchTemplate(target, template5, cv2.TM_CCORR_NORMED)
        result51 = cv2.matchTemplate(target, template51, cv2.TM_CCORR_NORMED)
        result6 = cv2.matchTemplate(target, template6, cv2.TM_CCORR_NORMED)
        result7 = cv2.matchTemplate(target, template7, cv2.TM_CCORR_NORMED)
        result8 = cv2.matchTemplate(target, template8, cv2.TM_CCORR_NORMED)
        result9 = cv2.matchTemplate(target, template9, cv2.TM_CCORR_NORMED)
        result10 = cv2.matchTemplate(target, template10, cv2.TM_CCORR_NORMED)
        result11 = cv2.matchTemplate(target, template11, cv2.TM_CCORR_NORMED)
        result12 = cv2.matchTemplate(target, template12, cv2.TM_CCORR_NORMED)
        result13 = cv2.matchTemplate(target, template13, cv2.TM_CCORR_NORMED)
        result14 = cv2.matchTemplate(target, template14, cv2.TM_CCORR_NORMED)
        result141 = cv2.matchTemplate(target, template141, cv2.TM_CCORR_NORMED)
        result142 = cv2.matchTemplate(target, template142, cv2.TM_CCORR_NORMED)
        result143 = cv2.matchTemplate(target, template143, cv2.TM_CCORR_NORMED)
        result15 = cv2.matchTemplate(target, template15, cv2.TM_CCORR_NORMED)
        result16 = cv2.matchTemplate(target, template16, cv2.TM_CCORR_NORMED)
        result17 = cv2.matchTemplate(target, template17, cv2.TM_CCORR_NORMED)

        height160, width160 = result0.shape[:2]
        height220, width220 = result2.shape[:2]
        height210, width210 = result4.shape[:2]
        print(iheight, iwidth, theight160, theight160, theight220, twidth220, height160, width160, height220, width220)
        for i in range(height160):
            for j in range(width160):
                MArray160[i + theight160 - 1, j + twidth160 - 1] = max(result0[i, j], result1[i, j])
                Array160[i + theight160 - 1, j + twidth160 - 1] = max(result6[i, j], result7[i, j], result8[i, j],
                                                                      result9[i, j], result10[i, j], result11[i, j])
        for i in range(height220):
            for j in range(width220):
                Array220[i + theight220 - 1, j + twidth220 - 1] = max(result12[i, j], result13[i, j], result14[i, j],
                                                                      result141[i, j], result142[i, j], result143[i, j])
                MArray220[i + theight220 - 1, j + twidth220 - 1] = max(result2[i, j], result3[i, j])
        for i in range(height210):
            for j in range(width210):
                Array210[i + theight210 - 1, j + twidth210 - 1] = max(result15[i, j], result16[i, j], result17[i, j])
                MArray210[i + theight210 - 1, j + twidth210 - 1] = max(result4[i, j], result5[i, j], result51[i, j])
        for i in range(iheight):
            for j in range(iwidth):
                Array160[i, j] = max(Array160[i, j], Array220[i, j], Array210[i, j])
                MArray160[i, j] = max(MArray160[i, j], MArray220[i, j], MArray210[i, j])
                if MArray160[i, j] - Array160[i, j] > -0.030:
                    Array[i, j] = 0
                else:
                    Array[i, j] = Array160[i, j]
        print(result0.shape[:2])
        # cv2.normalize(result, result, 0, 1, cv2.NORM_MINMAX, -1 )

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(Array)
        # 方框标记正面
        cv2.rectangle(target, max_loc, (max_loc[0] - twidth160, max_loc[1] - theight160), (255, 255, 225), 2)
        max_x = int((2 * max_loc[0] - twidth160) / 2)
        max_y = int((2 * max_loc[1] - theight160) / 2)
        cv2.putText(target, "*", (max_x - 1, max_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        long_text += '\n[' + str(max_x) + ',' + str(max_y) + ']'
        temp_loc = max_loc
        other_loc = max_loc
        numOfloc = 1

        threshold = 0.95
        loc = np.where(Array >= threshold)
        wrong = 1
        for other_loc in zip(*loc[::-1]):
            if (temp_loc[0] + wrong < other_loc[0]) or (temp_loc[1] + wrong < other_loc[1]):
                numOfloc = numOfloc + 1
                temp_loc = other_loc
                x = int((2 * other_loc[0] - twidth160) / 2)
                y = int((2 * other_loc[1] - theight160) / 2)
                if abs(max_x - x) > twidth160 / 2 or abs(max_y - y) > theight160 / 2:
                    cv2.rectangle(target, other_loc, (other_loc[0] - twidth160, other_loc[1] - theight160),
                                  (255, 255, 255), 2)
                    cv2.putText(target, "*", (x - 1, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    long_text += '\n[' + str(x) + ',' + str(y) + ']'
        # 将opencv图像转化为PIL图像
        img_output = Image.fromarray(cv2.cvtColor(target, cv2.COLOR_BGR2RGB))
        self.show(img_output)
        self.output.set(long_text)

    def back(self):
        long_text = '反面瓶盖的坐标为'
        # 将PIL图像转化为opencv图像
        target = cv2.cvtColor(np.array(self.photo), cv2.COLOR_RGB2BGR)
        iheight, iwidth = target.shape[:2]
        Array = np.zeros((iheight, iwidth))
        MArray160 = np.zeros((iheight, iwidth))
        Array160 = np.zeros((iheight, iwidth))
        MArray220 = np.zeros((iheight, iwidth))
        Array220 = np.zeros((iheight, iwidth))
        MArray210 = np.zeros((iheight, iwidth))
        Array210 = np.zeros((iheight, iwidth))
        template0 = cv2.imread("D:/code/cv/together/bottle_cap_recongnition/bottle_cap_recognition/templete/1up200.jpg")
        template1 = cv2.imread("D:/code/cv/together/bottle_cap_recongnition/bottle_cap_recognition/templete/2up200.jpg")
        template2 = cv2.imread("D:/code/cv/together/bottle_cap_recongnition/bottle_cap_recognition/templete/4up220.jpg")
        template3 = cv2.imread("D:/code/cv/together/bottle_cap_recongnition/bottle_cap_recognition/templete/5up220.jpg")
        template4 = cv2.imread("D:/code/cv/together/bottle_cap_recongnition/bottle_cap_recognition/templete/6up210.jpg")
        template5 = cv2.imread("D:/code/cv/together/bottle_cap_recongnition/bottle_cap_recognition/templete/7up210.jpg")
        template6 = cv2.imread(
            "D:/code/cv/together/bottle_cap_recongnition/bottle_cap_recognition/templete/1down200.jpg")
        template7 = cv2.imread(
            "D:/code/cv/together/bottle_cap_recongnition/bottle_cap_recognition/templete/2down200.jpg")
        template8 = cv2.imread(
            "D:/code/cv/together/bottle_cap_recongnition/bottle_cap_recognition/templete/31down200.jpg")
        template9 = cv2.imread(
            "D:/code/cv/together/bottle_cap_recongnition/bottle_cap_recognition/templete/4down200.jpg")
        template10 = cv2.imread(
            "D:/code/cv/together/bottle_cap_recongnition/bottle_cap_recognition/templete/5down200.jpg")
        template11 = cv2.imread(
            "D:/code/cv/together/bottle_cap_recongnition/bottle_cap_recognition/templete/6down200.jpg")
        template12 = cv2.imread(
            "D:/code/cv/together/bottle_cap_recongnition/bottle_cap_recognition/templete/7down220.jpg")
        template13 = cv2.imread(
            "D:/code/cv/together/bottle_cap_recongnition/bottle_cap_recognition/templete/8down220.jpg")
        template14 = cv2.imread(
            "D:/code/cv/together/bottle_cap_recongnition/bottle_cap_recognition/templete/9down220.jpg")
        template15 = cv2.imread(
            "D:/code/cv/together/bottle_cap_recongnition/bottle_cap_recognition/templete/101down210.jpg")
        template16 = cv2.imread(
            "D:/code/cv/together/bottle_cap_recongnition/bottle_cap_recognition/templete/11down210.jpg")
        template17 = cv2.imread(
            "D:/code/cv/together/bottle_cap_recongnition/bottle_cap_recognition/templete/12down210.jpg")
        template18 = cv2.imread(
            "D:/code/cv/together/bottle_cap_recongnition/bottle_cap_recognition/templete/3down200.jpg")

        theight160, twidth160 = template0.shape[:2]
        theight220, twidth220 = template2.shape[:2]
        theight210, twidth210 = template4.shape[:2]
        result0 = cv2.matchTemplate(target, template0, cv2.TM_CCORR_NORMED)
        result1 = cv2.matchTemplate(target, template1, cv2.TM_CCORR_NORMED)
        result2 = cv2.matchTemplate(target, template2, cv2.TM_CCORR_NORMED)
        result3 = cv2.matchTemplate(target, template3, cv2.TM_CCORR_NORMED)
        result4 = cv2.matchTemplate(target, template4, cv2.TM_CCORR_NORMED)
        result5 = cv2.matchTemplate(target, template5, cv2.TM_CCORR_NORMED)
        result6 = cv2.matchTemplate(target, template6, cv2.TM_CCORR_NORMED)
        result7 = cv2.matchTemplate(target, template7, cv2.TM_CCORR_NORMED)
        result8 = cv2.matchTemplate(target, template8, cv2.TM_CCORR_NORMED)
        result9 = cv2.matchTemplate(target, template9, cv2.TM_CCORR_NORMED)
        result10 = cv2.matchTemplate(target, template10, cv2.TM_CCORR_NORMED)
        result11 = cv2.matchTemplate(target, template11, cv2.TM_CCORR_NORMED)
        result12 = cv2.matchTemplate(target, template12, cv2.TM_CCORR_NORMED)
        result13 = cv2.matchTemplate(target, template13, cv2.TM_CCORR_NORMED)
        result14 = cv2.matchTemplate(target, template14, cv2.TM_CCORR_NORMED)
        result15 = cv2.matchTemplate(target, template15, cv2.TM_CCORR_NORMED)
        result16 = cv2.matchTemplate(target, template16, cv2.TM_CCORR_NORMED)
        result17 = cv2.matchTemplate(target, template17, cv2.TM_CCORR_NORMED)
        result18 = cv2.matchTemplate(target, template18, cv2.TM_CCORR_NORMED)
        height160, width160 = result0.shape[:2]
        height220, width220 = result2.shape[:2]
        height210, width210 = result4.shape[:2]
        print(iheight, iwidth, theight160, theight160, theight220, twidth220, height160, width160, height220, width220)
        for i in range(height160):
            for j in range(width160):
                MArray160[i + theight160 - 1, j + twidth160 - 1] = max(result0[i, j], result1[i, j])
                Array160[i + theight160 - 1, j + twidth160 - 1] = max(result6[i, j], result7[i, j], result8[i, j],
                                                                      result9[i, j], result10[i, j], result11[i, j],
                                                                      result18[i, j])
        for i in range(height220):
            for j in range(width220):
                Array220[i + theight220 - 1, j + twidth220 - 1] = max(result12[i, j], result13[i, j], result14[i, j])
                MArray220[i + theight220 - 1, j + twidth220 - 1] = max(result2[i, j], result3[i, j])
        for i in range(height210):
            for j in range(width210):
                Array210[i + theight210 - 1, j + twidth210 - 1] = max(result15[i, j], result16[i, j], result17[i, j])
                MArray210[i + theight210 - 1, j + twidth210 - 1] = max(result4[i, j], result5[i, j])
        for i in range(iheight):
            for j in range(iwidth):
                Array160[i, j] = max(Array160[i, j], Array220[i, j], Array210[i, j])
                MArray160[i, j] = max(MArray160[i, j], MArray220[i, j], MArray210[i, j])
                if MArray160[i, j] - Array160[i, j] > -0.035:
                    Array[i, j] = 0
                else:
                    Array[i, j] = Array160[i, j]
        print(result0.shape[:2])
        # cv2.normalize(result, result, 0, 1, cv2.NORM_MINMAX, -1 )

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(Array)
        # 方框标记反面
        cv2.rectangle(target, max_loc, (max_loc[0] - twidth160, max_loc[1] - theight160), (0, 0, 255), 2)
        max_x = int((2 * max_loc[0] - twidth160) / 2)
        max_y = int((2 * max_loc[1] - theight160) / 2)
        cv2.putText(target, "*", (max_x - 1, max_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        long_text += '\n[' + str(max_x) + ',' + str(max_y) + ']'
        temp_loc = max_loc
        other_loc = max_loc
        numOfloc = 1

        threshold = 0.95
        loc = np.where(Array >= threshold)
        wrong = 4
        for other_loc in zip(*loc[::-1]):
            if (temp_loc[0] + wrong < other_loc[0]) or (temp_loc[1] + wrong < other_loc[1]):
                numOfloc = numOfloc + 1
                temp_loc = other_loc
                x = int((2 * other_loc[0] - twidth160) / 2)
                y = int((2 * other_loc[1] - theight160) / 2)
                if abs(max_x - x) > twidth160 / 2 or abs(max_y - y) > theight160 / 2:
                    cv2.rectangle(target, other_loc, (other_loc[0] - twidth160, other_loc[1] - theight160), (0, 0, 255),
                                  2)
                    cv2.putText(target, "*", (x - 1, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    long_text += '\n[' + str(x) + ',' + str(y) + ']'
        # 将opencv图像转化为PIL图像
        img_output = Image.fromarray(cv2.cvtColor(target, cv2.COLOR_BGR2RGB))
        self.show(img_output)
        self.output.set(long_text)

    def profile(self, sc):
        long_text = '侧面瓶盖的坐标为'
        # 将PIL图像转化为opencv图像
        img_cv = cv2.cvtColor(np.array(self.photo), cv2.COLOR_RGB2BGR)
        # 进行裁剪操作
        resized = imutils.resize(img_cv, width=300)
        ratio = img_cv.shape[0] / float(resized.shape[0])
        # 进行高斯模糊操作
        blurred = cv2.GaussianBlur(resized, (5, 5), 0)
        # 进行图片灰度化
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("gray", gray)
        # 进行颜色空间的变换
        lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
        # 进行阈值分割
        thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)[1]
        # cv2.imshow("thresh", thresh)
        # 在二值图片中寻找轮廓
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        # 初始化形状检测器和颜色标签
        sd = ShapeDetector()
        cl = ColorLabeler()
        # 遍历每一个轮廓
        for c in cnts:
            # 计算每一个轮廓的中心点
            M = cv2.moments(c)
            if (M["m00"] == 0):  # this is a line
                shape = "line"
                continue
            cX = int((M["m10"] / M["m00"]) * ratio)
            cY = int((M["m01"] / M["m00"]) * ratio)
            # 进行颜色检测和形状检测
            shape = sd.detect(c)
            color = cl.label(lab, c)
            # 进行坐标变换
            c = c.astype("float")
            c *= ratio
            c = c.astype("int")
            if sc == "all":
                goal = "left"
                text = "{}".format(shape)
            else:
                goal = sc + " left"
                text = "{} {}".format(color, shape)
            if text == goal:
                if sc == "all":
                    cv2.fillPoly(img_cv, [c], (255, 255, 255))
                if sc == "red":
                    cv2.fillPoly(img_cv, [c], (0, 0, 255))
                if sc == "yellow":
                    cv2.fillPoly(img_cv, [c], (0, 255, 255))
                if sc == "blue":
                    cv2.fillPoly(img_cv, [c], (255, 0, 0))
                # 绘制轮廓并显示结果
                # cv2.drawContours(img_cv, [c], -1, (0, 0, 255), 2)
                cv2.putText(img_cv, "*", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                long_text += '\n[' + str(cX) + ',' + str(cY) + ']'
        # 将opencv图像转化为PIL图像
        img_output = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        self.show(img_output)
        self.output.set(long_text)

    def all(self):
        img_input = self.photo
        img_output = img_input
        self.show(img_output)
        long_text = '正面瓶盖的坐标为\n' + '[1,1]' + '\n背面瓶盖的坐标为\n' + '[1,1]' + '\n侧面瓶盖的坐标为\n' + '[1,1]'
        self.output.set(long_text)


if __name__ == "__main__":
    tk_root = Tk()
    recognition = BottleCapRecognition(tk_root)
    tk_root.mainloop()
