# -*- coding: UTF-8 -*-
from tkinter import Tk, Canvas, Button, filedialog, StringVar, Label
from PIL import Image, ImageTk
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

        # 侧面检测按钮
        self.button_profile = Button(self.root, text='侧面检测', command=lambda: self.profile())
        self.canvas1.create_window(80, 150, anchor='center', window=self.button_profile)

        # 背面检测按钮
        self.button_back = Button(self.root, text='背面检测', command=lambda: self.back())
        self.canvas1.create_window(80, 200, anchor='center', window=self.button_back)

        # 形态检测按钮
        self.button_all = Button(self.root, text='形态检测', command=lambda: self.all())
        self.canvas1.create_window(80, 250, anchor='center', window=self.button_all)

        # 输出语句
        self.output = StringVar()
        # 输出框
        self.text = Label(self.root, textvariable=self.output)
        self.canvas1.create_window(80, 290, anchor='n', window=self.text)


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
        img_input = self.photo
        img_output = img_input
        self.show(img_output)
        long_text = '正面瓶盖的坐标为\n' + '[1,1]'
        self.output.set(long_text)

    def back(self):
        img_input = self.photo
        img_output = img_input
        self.show(img_output)
        long_text = '背面瓶盖的坐标为\n' + '[1,1]'
        self.output.set(long_text)

    def profile(self):
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
        # 进行颜色空间的变换
        lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
        # 进行阈值分割
        thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)[1]
        # 在二值图片中寻找轮廓
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        # 初始化形状检测器和颜色标签
        sd = ShapeDetector()
        # 遍历每一个轮廓
        for c in cnts:
            # 计算每一个轮廓的中心点
            M = cv2.moments(c)
            if(M["m00"]==0): # this is a line
                shape = "line" 
                continue
            cX = int((M["m10"] / M["m00"]) * ratio)
            cY = int((M["m01"] / M["m00"]) * ratio)
            # 进行颜色检测和形状检测
            shape = sd.detect(c)
            #color = cl.label(lab, c)
            # 进行坐标变换
            c = c.astype("float")
            c *= ratio
            c = c.astype("int")
            text = "{}".format(shape)
            if(text=="left"):
                # 绘制轮廓并显示结果
                cv2.drawContours(img_cv, [c], -1, (0, 0, 255), 2)
                cv2.putText(img_cv, text, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                long_text += '\n[' + str(cX) + ',' + str(cY) + ']'
        # 将opencv图像转化为PIL图像
        img_output = Image.fromarray(cv2.cvtColor(img_cv,cv2.COLOR_BGR2RGB))
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
