# -*- coding: UTF-8 -*-
from tkinter import Tk, Canvas, Button, filedialog, StringVar, Label
from PIL import Image, ImageTk


def resize(w_box, h_box, pil_image):
    w, h = pil_image.size  # 获取图像的原始大小
    f1 = 1.0 * w_box / w
    f2 = 1.0 * h_box / h
    factor = min([f1, f2])
    width = int(w * factor)
    height = int(h * factor)
    return pil_image.resize((width, height), Image.ANTIALIAS)


class bottle_cap_recognition:
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
        self.canvas1.create_window(50, 10, anchor='nw', window=self.button_choose)

        # 正面检测按钮
        self.button_front = Button(self.root, text='正面检测', command=lambda: self.front())
        self.canvas1.create_window(50, 50, anchor='nw', window=self.button_front)

        # 侧面检测按钮
        self.button_profile = Button(self.root, text='侧面检测', command=lambda: self.profile())
        self.canvas1.create_window(50, 90, anchor='nw', window=self.button_profile)

        # 背面检测按钮
        self.button_back = Button(self.root, text='背面检测', command=lambda: self.back())
        self.canvas1.create_window(50, 130, anchor='nw', window=self.button_back)

        # 形态检测按钮
        self.button_all = Button(self.root, text='形态检测', command=lambda: self.all())
        self.canvas1.create_window(50, 170, anchor='nw', window=self.button_all)

        # 输出语句
        self.output = StringVar()
        # 输出框
        self.text = Label(self.root, textvariable=self.output)
        self.canvas1.create_window(10, 210, anchor='nw', window=self.text)



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
        img_input = self.photo
        img_output = img_input
        self.show(img_output)
        long_text = '侧面瓶盖的坐标为\n' + '[1,1]'
        self.output.set(long_text)

    def all(self):
        img_input = self.photo
        img_output = img_input
        self.show(img_output)
        long_text = '正面瓶盖的坐标为\n' + '[1,1]' + '\n背面瓶盖的坐标为\n' + '[1,1]' + '\n侧面瓶盖的坐标为\n' + '[1,1]'
        self.output.set(long_text)

if __name__ == "__main__":
    tk_root = Tk()
    recognition = bottle_cap_recognition(tk_root)
    tk_root.mainloop()
