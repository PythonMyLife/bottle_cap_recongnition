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
        Array150 = np.zeros((iheight, iwidth))
        Array160 = np.zeros((iheight, iwidth))
        # 引入模板
        template = cv2.imread("D:/myGithub/bottle_cap_recongnition/bottle_cap_recognition/templete/down1black150.jpg")
        template1 = cv2.imread("D:/myGithub/bottle_cap_recongnition/bottle_cap_recognition/templete/down2black150.jpg")
        template2 = cv2.imread("D:/myGithub/bottle_cap_recongnition/bottle_cap_recognition/templete/down3black150.jpg")
        template3 = cv2.imread("D:/myGithub/bottle_cap_recongnition/bottle_cap_recognition/templete/up1black150.jpg")
        template4 = cv2.imread("D:/myGithub/bottle_cap_recongnition/bottle_cap_recognition/templete/up2black150.jpg")
        template5 = cv2.imread("D:/myGithub/bottle_cap_recongnition/bottle_cap_recognition/templete/up3black150.jpg")
        theight150, twidth150 = template.shape[:2]
        theight160, twidth160 = template3.shape[:2]
        # 模板匹配
        result = cv2.matchTemplate(target, template, cv2.TM_CCORR_NORMED)
        result1 = cv2.matchTemplate(target, template1, cv2.TM_CCORR_NORMED)
        result2 = cv2.matchTemplate(target, template2, cv2.TM_CCORR_NORMED)
        result3 = cv2.matchTemplate(target, template3, cv2.TM_CCORR_NORMED)
        result4 = cv2.matchTemplate(target, template4, cv2.TM_CCORR_NORMED)
        result5 = cv2.matchTemplate(target, template5, cv2.TM_CCORR_NORMED)
        height150, width150 = result.shape[:2]
        height160, width160 = result3.shape[:2]
        # 处理反面瓶盖
        for i in range(height150):
            for j in range(width150):
                Array150[i + theight150 - 1, j + twidth150 - 1] = max(result[i, j], result1[i, j], result2[i, j])
        # 处理正面瓶盖
        for i in range(height160):
            for j in range(width160):
                Array160[i + theight160 - 1, j + twidth160 - 1] = max(result3[i, j], result4[i, j], result5[i, j])
        # 减少正反瓶盖的相互干扰
        for i in range(iheight):
            for j in range(iwidth):
                if Array150[i, j] - Array160[i, j] > -0.01:
                    Array[i, j] = 0
                else:
                    Array[i, j] = Array160[i, j]
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(Array)
        # 方框标记正面
        cv2.rectangle(target, max_loc, (max_loc[0]-twidth160, max_loc[1]-theight160), (255, 255, 225), 2)
        max_x = int((2*max_loc[0]-twidth160)/2)
        max_y = int((2*max_loc[1]-theight160)/2)
        cv2.putText(target, "*", (max_x-1, max_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        long_text += '\n[' + str(max_x) + ',' + str(max_y) + ']'
        temp_loc = max_loc
        other_loc = max_loc
        numOfloc = 1

        threshold = 0.95
        loc = np.where(Array >= threshold)
        wrong = 15
        for other_loc in zip(*loc[::-1]):
            if (temp_loc[0] + wrong < other_loc[0]) or (temp_loc[1] + wrong < other_loc[1]):
                numOfloc = numOfloc + 1
                temp_loc = other_loc
                x = int((2*other_loc[0]-twidth160)/2)
                y = int((2*other_loc[1]-theight160)/2)
                if abs(max_x - x) > twidth160/2 or abs(max_y - y) > theight160/2:
                    cv2.rectangle(target, other_loc, (other_loc[0] - twidth160, other_loc[1] - theight160), (255, 255, 255), 2)
                    cv2.putText(target, "*", (x-1, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    long_text += '\n[' + str(x) + ',' + str(y) + ']'
        # 将opencv图像转化为PIL图像
        img_output = Image.fromarray(cv2.cvtColor(target,cv2.COLOR_BGR2RGB))
        self.show(img_output)
        self.output.set(long_text)

    def back(self):
        long_text = '反面瓶盖的坐标为'
        # 将PIL图像转化为opencv图像
        target = cv2.cvtColor(np.array(self.photo), cv2.COLOR_RGB2BGR)
        iheight, iwidth = target.shape[:2]
        Array = np.zeros((iheight, iwidth))
        Array150 = np.zeros((iheight, iwidth))
        Array160 = np.zeros((iheight, iwidth))
        # 引入模板
        # template = cv2.cvtColor(np.array(Image.open("./templete/up1black150.jpg")), cv2.COLOR_RGB2BGR)
        template = cv2.imread("D:/myGithub/bottle_cap_recongnition/bottle_cap_recognition/templete/up1black150.jpg")
        template1 = cv2.imread("D:/myGithub/bottle_cap_recongnition/bottle_cap_recognition/templete/up2black150.jpg")
        template2 = cv2.imread("D:/myGithub/bottle_cap_recongnition/bottle_cap_recognition/templete/up3black150.jpg")
        template3 = cv2.imread("D:/myGithub/bottle_cap_recongnition/bottle_cap_recognition/templete/down1black150.jpg")
        template4 = cv2.imread("D:/myGithub/bottle_cap_recongnition/bottle_cap_recognition/templete/down2black150.jpg")
        template5 = cv2.imread("D:/myGithub/bottle_cap_recongnition/bottle_cap_recognition/templete/down3black150.jpg")
        theight150, twidth150 = template.shape[:2]
        theight160, twidth160 = template3.shape[:2]
        # 模板匹配
        result = cv2.matchTemplate(target, template, cv2.TM_CCORR_NORMED)
        result1 = cv2.matchTemplate(target, template1, cv2.TM_CCORR_NORMED)
        result2 = cv2.matchTemplate(target, template2, cv2.TM_CCORR_NORMED)
        result3 = cv2.matchTemplate(target, template3, cv2.TM_CCORR_NORMED)
        result4 = cv2.matchTemplate(target, template4, cv2.TM_CCORR_NORMED)
        result5 = cv2.matchTemplate(target, template5, cv2.TM_CCORR_NORMED)
        height150, width150 = result.shape[:2]
        height160, width160 = result3.shape[:2]
        # 处理正面瓶盖
        for i in range(height150):
            for j in range(width150):
                Array150[i + theight150 - 1, j + twidth150 - 1] = max(result[i, j], result1[i, j], result2[i, j])
        # 处理反面瓶盖
        for i in range(height160):
            for j in range(width160):
                Array160[i + theight160 - 1, j + twidth160 - 1] = max(result3[i, j], result4[i, j], result5[i, j])
        # 减少正反瓶盖的相互干扰
        for i in range(iheight):
            for j in range(iwidth):
                if Array150[i, j] - Array160[i, j] > -0.028:
                    Array[i, j] = 0
                else:
                    Array[i, j] = Array160[i, j]
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(Array)
        # 方框标记反面
        cv2.rectangle(target, max_loc, (max_loc[0]-twidth160, max_loc[1]-theight160), (0, 0, 255), 2)
        max_x = int((2*max_loc[0]-twidth160)/2)
        max_y = int((2*max_loc[1]-theight160)/2)
        cv2.putText(target, "*", (max_x-1, max_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        long_text += '\n[' + str(max_x) + ',' + str(max_y) + ']'
        temp_loc = max_loc
        other_loc = max_loc
        numOfloc = 1

        threshold = 0.9
        loc = np.where(Array >= threshold)
        wrong = 40
        for other_loc in zip(*loc[::-1]):
            if (temp_loc[0] + wrong < other_loc[0]) or (temp_loc[1] + wrong < other_loc[1]):
                numOfloc = numOfloc + 1
                temp_loc = other_loc
                x = int((2*other_loc[0]-twidth160)/2)
                y = int((2*other_loc[1]-theight160)/2)
                if abs(max_x - x) > twidth160/2 or abs(max_y - y) > theight160/2:
                    cv2.rectangle(target, other_loc, (other_loc[0] - twidth160, other_loc[1] - theight160), (0, 0, 255), 2)
                    cv2.putText(target, "*", (x-1, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    long_text += '\n[' + str(x) + ',' + str(y) + ']'
        # 将opencv图像转化为PIL图像
        img_output = Image.fromarray(cv2.cvtColor(target,cv2.COLOR_BGR2RGB))
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
        #cv2.imshow("gray", gray)
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
            if(M["m00"]==0): # this is a line
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
            if text==goal:
                if sc == "all":
                    cv2.fillPoly(img_cv,[c],(255, 255, 255))
                if sc == "red":
                    cv2.fillPoly(img_cv,[c],(0, 0, 255))
                if sc == "yellow":
                    cv2.fillPoly(img_cv,[c],(0, 255, 255))
                if sc == "blue":
                    cv2.fillPoly(img_cv,[c],(255, 0, 0))
                # 绘制轮廓并显示结果
                # cv2.drawContours(img_cv, [c], -1, (0, 0, 255), 2)
                cv2.putText(img_cv, "*", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
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
