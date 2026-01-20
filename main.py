from PIL import Image, ImageOps, ImageDraw, ImageFilter, ImageChops, ImageEnhance, ImagePalette, ImageSequence, ImageTransform, ImageCms, UnidentifiedImageError,\
PngImagePlugin, GimpGradientFile, TiffImagePlugin, features
from PIL import ImageFile
# 导入 Resampling
from PIL import Image
import os
import numpy as np

# TensorFlow 2.x 中集成了 keras
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
import tkinter as tk
from tkinter import filedialog
# 模型保存路径
MODEL_PATH = "mnist_cnn.h5"

# ---------------------------
# 加载预训练模型
# ---------------------------
if os.path.exists(MODEL_PATH):
    print("加载已保存的模型...")
    model = load_model(MODEL_PATH)
else:
    raise FileNotFoundError("未找到训练好的模型，请先运行原始代码训练模型")

# ---------------------------
# 图像预处理函数
# ---------------------------
def preprocess_image(image):
    """预处理图像：转换为模型需要的格式"""
    image = image.convert('L')
    image = image.resize((28, 28), Image.LANCZOS)
    img_array = np.array(image)

    # 若图像背景过白则反转颜色（MNIST 风格）
    if np.mean(img_array) > 127:
        img_array = 255 - img_array

    img_array = img_array.astype('float32') / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    return img_array

# ---------------------------
# GUI 应用程序类
# ---------------------------
class DigitRecognizerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("手写数字识别系统")
        self.geometry("400x500")

        # 创建主容器
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        # 初始化所有界面
        self.frames = {}
        for F in (HomePage, UploadPage, DrawPage):
            frame = F(parent=container, controller=self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(HomePage)

    def show_frame(self, page_class):
        """显示指定页面"""
        frame = self.frames[page_class]
        frame.tkraise()

# ---------------------------
# 首页
# ---------------------------
class HomePage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        label = tk.Label(self, text="手写数字识别系统", font=("Arial", 16))
        label.pack(pady=20)

        upload_btn = tk.Button(
            self,
            text="上传图片识别",
            width=20,
            command=lambda: controller.show_frame(UploadPage)  # 修复这里缺少的括号
        )
        upload_btn.pack(pady=10)

        draw_btn = tk.Button(
            self,
            text="手写板识别",
            width=20,
            command=lambda: controller.show_frame(DrawPage))
        draw_btn.pack(pady=10)

# ---------------------------
# 上传页面
# ---------------------------
class UploadPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        # 创建界面元素
        label = tk.Label(self, text="上传图片识别", font=("Arial", 14))
        label.pack(pady=10)

        self.result_label = tk.Label(self, text="", font=("Arial", 18))
        self.result_label.pack(pady=20)

        btn_frame = tk.Frame(self)
        btn_frame.pack(pady=10)

        upload_btn = tk.Button(
            btn_frame,
            text="选择图片",
            width=15,
            command=self.upload_image)
        upload_btn.pack(side="left", padx=5)

        back_btn = tk.Button(
            btn_frame,
            text="返回首页",
            width=15,
            command=lambda: controller.show_frame(HomePage))
        back_btn.pack(side="left", padx=5)

    def upload_image(self):
        """处理图片上传"""
        file_types = [("图片文件", "*.png *.jpg *.jpeg")]
        file_path = filedialog.askopenfilename(filetypes=file_types)

        if file_path:
            try:
                image = Image.open(file_path)
                processed = preprocess_image(image)
                prediction = np.argmax(model.predict(processed), axis=-1)[0]
                self.result_label.config(text=f"识别结果: {prediction}")
            except Exception as e:
                self.result_label.config(text=f"错误: {str(e)}")

# ---------------------------
# 手写板页面
# ---------------------------
class DrawPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        # 绘图相关变量
        self.image = Image.new("L", (280, 280), 255)  # 内存中的灰度图像
        self.draw = ImageDraw.Draw(self.image)
        self.old_x = None
        self.old_y = None

        # 创建界面元素
        label = tk.Label(self, text="手写板识别", font=("Arial", 14))
        label.pack(pady=10)

        self.canvas = tk.Canvas(
            self,
            width=280,
            height=280,
            bg="white",
            cursor="crosshair")
        self.canvas.pack()

        self.result_label = tk.Label(self, text="", font=("Arial", 18))
        self.result_label.pack(pady=10)

        btn_frame = tk.Frame(self)
        btn_frame.pack(pady=10)

        clear_btn = tk.Button(
            btn_frame,
            text="清空画板",
            width=15,
            command=self.clear_canvas)
        clear_btn.pack(side="left", padx=5)

        predict_btn = tk.Button(
            btn_frame,
            text="开始识别",
            width=15,
            command=self.predict_digit)
        predict_btn.pack(side="left", padx=5)

        back_btn = tk.Button(
            btn_frame,
            text="返回首页",
            width=15,
            command=lambda: controller.show_frame(HomePage))
        back_btn.pack(side="left", padx=5)

        # 绑定鼠标事件
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset)

    def paint(self, event):
        """处理鼠标拖动绘制"""
        x, y = event.x, event.y
        if self.old_x and self.old_y:
            # 在画布上绘制
            self.canvas.create_line(
                self.old_x, self.old_y, x, y,
                width=15,
                fill="black",
                capstyle=tk.ROUND,
                smooth=True
            )
            # 在内存图像上绘制
            self.draw.line(
                [(self.old_x, self.old_y), (x, y)],
                fill=0,
                width=15
            )
        self.old_x = x
        self.old_y = y

    def reset(self, event):
        """重置鼠标坐标"""
        self.old_x = None
        self.old_y = None

    def clear_canvas(self):
        """清空画板"""
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), 255)
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="")

    def predict_digit(self):
        """进行数字识别"""
        # 将图像缩放到模型输入尺寸
        img = self.image.resize((28, 28), Image.LANCZOS)
        processed = preprocess_image(img)
        prediction = np.argmax(model.predict(processed), axis=-1)[0]
        self.result_label.config(text=f"识别结果: {prediction}")

# ---------------------------
# 运行应用程序
# ---------------------------
if __name__ == "__main__":
    app = DigitRecognizerApp()
    app.mainloop()