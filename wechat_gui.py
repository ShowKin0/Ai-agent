"""
=================================================================================
微信窗口自动化 - GUI版本（带自动监听）
当对方发送消息时自动截取识别

这是一个Python学习示例，演示了：
1. GUI图形界面编程 (CustomTkinter)
2. Windows窗口自动化 (uiautomation)
3. 屏幕截图和图像处理 (pyautogui, PIL)
4. OCR文字识别 (EasyOCR)
5. 多线程编程 (threading)
=================================================================================
"""
# ==============================================================================
# 第1部分：导入模块
# ==============================================================================

# import: 引入模块，类似于"把工具箱拿过来"
# as ctk: 给模块起个别名，方便后面使用

import customtkinter as ctk  # GUI界面库，类似tkinter但更美观
import uiautomation as auto  # Windows UI自动化库，用来操作微信窗口
import pyautogui            # 屏幕截图和鼠标键盘控制库
import time                 # 时间相关，如暂停( sleep )
import pyperclip           # 剪贴板操作，复制/粘贴文本
from pywinauto.keyboard import send_keys  # 发送键盘按键（如Ctrl+V）
import threading           # 多线程，让程序同时做多件事
import hashlib             # 哈希算法，用于生成文件的"指纹"

# ==============================================================================
# 第2部分：全局变量
# ==============================================================================

# 全局变量：在整个程序都能使用的变量
# 命名规则：以下划线开头表示"内部使用"

_wechat_window = None      # 存储微信窗口对象，初始为空
_ocr_engine = None        # 存储OCR识别引擎，初始为空
_ocr_ready = False         # OCR是否准备就绪的标志
_last_screenshot_hash = None  # 上一次截图的哈希值，用于比较是否有新消息
_monitoring = False        # 是否正在监控的标志

# ==============================================================================
# 第3部分：GUI主题设置
# ==============================================================================

# 设置界面主题
ctk.set_appearance_mode("dark")  # "dark"深色模式，还有"light"浅色模式
ctk.set_default_color_theme("blue")  # 蓝色主题，还有"green", "red"等

# ==============================================================================
# 第4部分：主类定义
# ==============================================================================

# class: 定义一个类（ blueprints 蓝图/模板）
# WeChatGUI(ctk.CTk): 继承CustomTkinter的CTk类，获得GUI功能
class WeChatGUI(ctk.CTk):

    # __init__: 初始化方法，创建对象时自动调用
    # self: 代表"自己"这个对象，类似中文的"我"
    def __init__(self):
        # super().__init__(): 调用父类的初始化方法
        super().__init__()

        # 设置窗口标题
        self.title("WeChat Automation Tool - Auto Monitor")

        # 设置窗口大小 ("宽度x高度")
        self.geometry("600x750")

        # 调用方法设置界面
        self.setup_ui()

        # after(毫秒, 函数): 延迟执行，常用于定时任务
        # 100毫秒后调用 init_ocr_background 初始化OCR
        self.after(100, self.init_ocr_background)

        # 设置监控间隔（2秒 = 2000毫秒）
        self.check_interval = 2000

        # 启动自动监控循环
        self.after(self.check_interval, self.auto_monitor)

    # ==========================================================================
    # 第5部分：setup_ui - 设置用户界面
    # ==========================================================================

    def setup_ui(self):
        """设置所有UI组件"""

        # ----------------- 标题 -----------------
        # ctk.CTkLabel: 标签组件，显示文本
        self.title_label = ctk.CTkLabel(
            self,  # 父容器
            text="WeChat Automation - Auto Monitor",  # 显示的文字
            font=ctk.CTkFont(size=20, weight="bold")  # 字体设置
        )
        self.title_label.pack(pady=15)  # pack(): 放置组件，pady:上下边距

        # ----------------- 连接状态区域 -----------------
        # ctk.CTkFrame: 框架/容器，用于分组
        self.status_frame = ctk.CTkFrame(self)
        self.status_frame.pack(pady=10, padx=20, fill="x")  # fill="x"横向填充

        # 状态文字标签
        self.status_label = ctk.CTkLabel(
            self.status_frame,
            text="Status: Not connected",
            font=ctk.CTkFont(size=14)
        )
        self.status_label.pack(pady=5)

        # 连接按钮
        # command: 点击按钮时执行的函数
        self.connect_btn = ctk.CTkButton(
            self.status_frame,
            text="Connect WeChat",
            command=self.connect_wechat  # 点击时调用 connect_wechat 方法
        )
        self.connect_btn.pack(pady=5)

        # ----------------- 监控控制区域 -----------------
        self.monitor_frame = ctk.CTkFrame(self)
        self.monitor_frame.pack(pady=10, padx=20, fill="x")

        # 监控按钮（初始禁用，等连接微信后启用）
        self.monitor_btn = ctk.CTkButton(
            self.monitor_frame,
            text="Start Auto Monitor",
            command=self.toggle_monitor,  # 点击切换监控状态
            state="disabled",  # 禁用状态
            fg_color="orange"  # 前景色（按钮颜色）
        )
        self.monitor_btn.pack(pady=5)

        # 监控状态标签
        self.monitor_label = ctk.CTkLabel(
            self.monitor_frame,
            text="Monitor: Stopped",
            font=ctk.CTkFont(size=12)
        )
        self.monitor_label.pack(pady=5)

        # ----------------- 操作按钮区域 -----------------
        self.btn_frame = ctk.CTkFrame(self)
        self.btn_frame.pack(pady=10, padx=20, fill="x")

        # 手动捕获按钮
        self.capture_btn = ctk.CTkButton(
            self.btn_frame,
            text="Manual Capture",
            command=self.manual_capture,
            state="disabled"
        )
        # side="left"靠左, padx水平边距, expand展开, fill填充
        self.capture_btn.pack(side="left", padx=5, expand=True, fill="x")

        # 清空按钮
        self.clear_btn = ctk.CTkButton(
            self.btn_frame,
            text="Clear",
            command=self.clear_messages
        )
        self.clear_btn.pack(side="left", padx=5)

        # ----------------- 消息显示区域 -----------------
        self.msg_label = ctk.CTkLabel(self, text="Messages Log:")
        self.msg_label.pack(pady=(10, 0), padx=20, anchor="w")  # anchor="w"靠左

        # ctk.CTkTextbox: 多行文本框，用于显示日志
        self.msg_textbox = ctk.CTkTextbox(self, height=280)
        self.msg_textbox.pack(pady=10, padx=20, fill="both", expand=True)

        # ----------------- 发送消息区域 -----------------
        self.send_frame = ctk.CTkFrame(self)
        self.send_frame.pack(pady=10, padx=20, fill="x")

        # 输入框 placeholder: 占位符提示文字
        self.send_entry = ctk.CTkEntry(
            self.send_frame,
            placeholder_text="Enter message to send..."
        )
        self.send_entry.pack(side="left", padx=5, expand=True, fill="x")

        # 绑定回车键：当按回车时发送消息
        # bind: 绑定事件， "<Return>"回车键， lambda创建匿名函数
        self.send_entry.bind("<Return>", lambda e: self.send_message())

        # 发送按钮
        self.send_btn = ctk.CTkButton(
            self.send_frame,
            text="Send",
            command=self.send_message,
            width=80,
            state="disabled"
        )
        self.send_btn.pack(side="left", padx=5)

        # ----------------- OCR状态 -----------------
        self.ocr_label = ctk.CTkLabel(
            self,
            text="OCR: Initializing...",
            font=ctk.CTkFont(size=12)
        )
        self.ocr_label.pack(pady=5)

    # ==========================================================================
    # 第6部分：日志功能
    # ==========================================================================

    def log_message(self, text, tag=None):
        """在文本框中添加日志消息，带时间戳"""
        # time.strftime: 格式化时间 "%H:%M:%S" 时:分:秒
        timestamp = time.strftime("%H:%M:%S")

        # insert: 插入文本 ("end"末尾位置)
        self.msg_textbox.insert("end", f"[{timestamp}] {text}\n")

        # see: 滚动到指定位置，"end"滚动到最底部
        self.msg_textbox.see("end")

    # ==========================================================================
    # 第7部分：OCR初始化（后台执行）
    # ==========================================================================

    def init_ocr_background(self):
        """在后台初始化OCR引擎"""
        # global: 声明使用全局变量
        global _ocr_engine, _ocr_ready

        try:
            # import: 在函数内部导入，延迟加载
            import easyocr

            # 更新界面显示
            self.ocr_label.configure(text="OCR: Loading models...")
            self.update()  # 强制更新界面

            # 创建OCR阅读器
            # ['ch_sim', 'en']: 支持中文简体和英文
            # gpu=False: 不使用GPU（用CPU）
            # verbose=False: 不输出详细信息
            _ocr_engine = easyocr.Reader(['ch_sim', 'en'], gpu=False, verbose=False)

            # 设置标志
            _ocr_ready = True

            # configure: 修改组件属性
            self.ocr_label.configure(text="OCR: Ready", text_color="green")
            self.log_message("OCR engine ready!")

        except Exception as e:
            # 捕获异常并显示错误
            self.ocr_label.configure(text=f"OCR Error: {str(e)[:30]}", text_color="red")
            self.log_message(f"OCR Error: {e}")

    # ==========================================================================
    # 第8部分：连接微信
    # ==========================================================================

    def connect_wechat(self):
        """连接到微信窗口"""
        global _wechat_window

        try:
            # WindowControl: 创建窗口控制对象
            # searchDepth=1: 搜索深度，只找顶层窗口
            # Name='微信': 窗口标题包含"微信"
            _wechat_window = auto.WindowControl(searchDepth=1, Name='微信')

            # Exists(3): 检查窗口是否存在，等待3秒
            if _wechat_window.Exists(3):
                # BoundingRectangle: 获取窗口位置和大小
                rect = _wechat_window.BoundingRectangle

                # 修改状态文字
                self.status_label.configure(
                    text=f"Connected: {rect.width()}x{rect.height()}",
                    text_color="green"  # 绿色表示成功
                )

                # configure(state="disabled"): 禁用按钮
                self.connect_btn.configure(state="disabled")

                # 启用相关按钮
                self.monitor_btn.configure(state="normal")
                self.capture_btn.configure(state="normal")
                self.send_btn.configure(state="normal")

                self.log_message("WeChat connected!")
            else:
                self.status_label.configure(text="WeChat not found!", text_color="red")
                self.log_message("WeChat window not found!")

        except Exception as e:
            self.status_label.configure(text=f"Error: {str(e)[:50]}", text_color="red")
            self.log_message(f"Connect Error: {e}")

    # ==========================================================================
    # 第9部分：监控开关
    # ==========================================================================

    def toggle_monitor(self):
        """切换自动监控的开关状态"""
        global _monitoring

        if _monitoring:
            # 如果正在监控，则停止
            _monitoring = False
            self.monitor_btn.configure(text="Start Auto Monitor", fg_color="orange")
            self.monitor_label.configure(text="Monitor: Stopped")
            self.log_message("Auto monitor stopped")
        else:
            # 如果没有监控，则开始
            _monitoring = True
            self.monitor_btn.configure(text="Stop Monitor", fg_color="red")
            self.monitor_label.configure(text="Monitor: Running")
            self.log_message("Auto monitor started! Will check every 2 seconds.")
            # 立即执行一次检测
            self.do_check()

    # ==========================================================================
    # 第10部分：手动捕获
    # ==========================================================================

    def manual_capture(self):
        """手动点击捕获按钮时调用"""
        # threading.Thread: 创建新线程
        # target: 线程执行的目标函数
        # daemon=True: 设为守护线程，主程序结束它也结束
        thread = threading.Thread(target=self._do_capture, daemon=True)
        thread.start()  # 启动线程

    # ==========================================================================
    # 第11部分：截图功能
    # ==========================================================================

    def capture_screenshot(self):
        """截取微信聊天区域并返回图像"""
        global _wechat_window

        # 如果没有连接微信，返回空
        if not _wechat_window:
            return None

        try:
            # 获取窗口位置信息
            rect = _wechat_window.BoundingRectangle

            # 拆分成4个变量（解包）
            x = rect.left      # 左边
            y = rect.top       # 顶边
            width = rect.width()    # 宽度
            height = rect.height()  # 高度

            # pyautogui.screenshot(): 截取整个屏幕
            screenshot = pyautogui.screenshot()

            # 裁剪聊天区域
            # 微信窗口布局：
            # - 左侧好友列表宽度: 约280-300px
            # - 顶部标题栏高度: 约40px
            # - 底部输入框高度: 约80px
            # - 中间灰色区域: 消息显示区域

            left_panel_width = 300     # 左侧好友列表宽度
            top_bar_height = 40        # 顶部标题栏高度
            bottom_input_height = 120  # 底部输入框高度（更大一些，排除输入框）

            # 计算聊天区域坐标
            chat_x = x + left_panel_width       # 聊天区域左边
            chat_y = y + top_bar_height         # 聊天区域顶边
            chat_width = width - left_panel_width   # 聊天区域宽度
            chat_height = height - top_bar_height - bottom_input_height  # 高度

            # crop: 裁剪图像，参数是(左, 上, 右, 下)四个角坐标
            chat_screenshot = screenshot.crop((
                chat_x, chat_y,
                chat_x + chat_width, chat_y + chat_height
            ))

            return chat_screenshot  # 返回裁剪后的图像

        except Exception as e:
            self.log_message(f"Capture error: {e}")
            return None

    # ==========================================================================
    # 第12部分：图像哈希
    # ==========================================================================

    def get_image_hash(self, image):
        """获取图像的哈希值（用于比较是否有变化）"""
        if image is None:
            return None

        # io.BytesIO: 创建内存中的字节流对象
        import io
        buffer = io.BytesIO()

        # 将图像保存为PNG格式到内存
        image.save(buffer, format='PNG')

        # hashlib.md5: 创建MD5哈希对象
        # digest(): 获取哈希值
        return hashlib.md5(buffer.getvalue()).hexdigest()

    # ==========================================================================
    # 第13部分：自动监控循环
    # ==========================================================================

    def auto_monitor(self):
        """定时自动监控循环"""
        global _monitoring

        # 如果正在监控，且微信已连接，OCR已就绪
        if _monitoring and _wechat_window and _ocr_ready:
            self.do_check()  # 执行一次检测

        # after: 设置下一次执行，形成循环
        self.after(self.check_interval, self.auto_monitor)

    # ==========================================================================
    # 第14部分：执行检测
    # ==========================================================================

    def do_check(self):
        """在新线程中执行检测，避免阻塞界面"""
        thread = threading.Thread(target=self._check_new_messages, daemon=True)
        thread.start()

    # ==========================================================================
    # 第15部分：检测新消息
    # ==========================================================================

    def _check_new_messages(self):
        """检测是否有新消息（比较截图变化）"""
        global _last_screenshot_hash

        try:
            # 1. 截图
            chat_screenshot = self.capture_screenshot()
            if chat_screenshot is None:
                return

            # 保存用于调试（可查看截图）
            chat_screenshot.save("chat_area.png")

            # 2. 计算当前截图的哈希值
            current_hash = self.get_image_hash(chat_screenshot)

            # 3. 比较
            if _last_screenshot_hash is None:
                # 第一次运行，保存哈希但不做任何事
                _last_screenshot_hash = current_hash
                self.log_message("Initial capture saved, waiting for changes...")
                return

            # 如果哈希不同，说明有变化（新消息）
            if current_hash != _last_screenshot_hash:
                self.log_message("Change detected! Recognizing...")

                # 更新哈希值
                _last_screenshot_hash = current_hash

                # 调用OCR识别
                self.recognize_and_log(chat_screenshot)

        except Exception as e:
            print(f"Check error: {e}")

    # ==========================================================================
    # 第16部分：OCR识别并记录
    # ==========================================================================

    def recognize_and_log(self, image):
        """用OCR识别图像中的文字并显示"""
        global _ocr_engine

        if not _ocr_engine:
            return

        try:
            # 保存原始图像用于调试
            image.save("chat_area_original.png")

            # ========== 图像预处理，提高OCR准确率 ==========
            from PIL import Image, ImageEnhance
            import numpy as np

            # 1. 转换为灰度图
            gray_image = image.convert('L')

            # 2. 增加对比度（让文字更清晰）
            enhancer = ImageEnhance.Contrast(gray_image)
            enhanced_image = enhancer.enhance(2.0)

            # 3. 适当放大图像（OCR对大图识别更准）
            width, height = enhanced_image.size
            # 放大1.5倍
            big_image = enhanced_image.resize((int(width * 1.5), int(height * 1.5)), Image.Resampling.LANCZOS)

            # 保存处理后的图像
            big_image.save("chat_area_processed.png")

            # 使用处理后的图像进行OCR识别
            result = _ocr_engine.readtext("chat_area_processed.png")

            # 提取文字
            texts = []
            if result:
                # 遍历每个识别结果
                for item in result:
                    text = item[1]  # 第2个元素是文字内容

                    # 过滤空文字
                    # startswith(">"): 过滤掉以">"开头的文字（自己发送的消息）
                    if text and len(text.strip()) > 0 and not text.strip().startswith(">"):
                        import re
                        time_pattern_1 = r'^\d{1,2}:\d{2}$'
                        time_pattern_2 = r'^\d{1,2};\d{2}$'
                        # 匹配纯数字 (如 36588, 5885 等无意义数字)
                        number_pattern = r'^\d+$'

                        stripped = text.strip()
                        if re.match(time_pattern_1, stripped) or re.match(time_pattern_2, stripped):
                            continue  # 跳过时间
                        if re.match(number_pattern, stripped) and len(stripped) >= 3:
                            continue  # 跳过长数字串

                        texts.append(text)

            # 如果有识别到文字
            if texts:
                # join: 用换行符连接所有文字
                print(texts)
                full_text = "\n".join(texts)

                self.log_message("=== NEW MESSAGE DETECTED ===")

                # 插入到文本框
                self.msg_textbox.insert("end", f"--- New ---\n{full_text}\n---\n")
                self.msg_textbox.see("end")

                self.log_message(f"Recognized {len(texts)} text blocks")
            else:
                self.log_message("Screen changed but no text detected")

        except Exception as e:
            self.log_message(f"OCR Error: {e}")

    # ==========================================================================
    # 第17部分：手动捕获执行
    # ==========================================================================

    def _do_capture(self):
        """手动捕获并识别（在线程中运行）"""
        self.log_message("Manual capture...")

        try:
            chat_screenshot = self.capture_screenshot()
            if chat_screenshot:
                self.recognize_and_log(chat_screenshot)
        except Exception as e:
            self.log_message(f"Capture error: {e}")

    # ==========================================================================
    # 第18部分：发送消息
    # ==========================================================================

    def send_message(self):
        """发送消息到微信"""
        # get(): 获取输入框中的文字
        message = self.send_entry.get()

        if not message:
            return  # 如果为空，直接返回

        if not _wechat_window:
            self.log_message("WeChat not connected!")
            return

        try:
            # 在消息开头添加 ">"，这样OCR时会过滤掉自己发的消息
            message_to_send = ">" + message

            # SetActive: 激活窗口到前台
            _wechat_window.SetActive()
            time.sleep(0.2)  # 等待0.2秒

            # SetFocus: 获取焦点
            _wechat_window.SetFocus()
            time.sleep(0.2)

            # pyperclip.copy: 复制到剪贴板（发送带#的消息）
            pyperclip.copy(message_to_send)


            # send_keys: 发送键盘按键
            # "^v" 代表 Ctrl+V（粘贴）
            send_keys("^v")
            time.sleep(0.1)

            # "{ENTER}" 代表回车键（发送）
            send_keys("{ENTER}")

            # 清空输入框
            # delete(0, "end"): 从第0个字符删除到末尾
            self.send_entry.delete(0, "end")

            # 日志显示发送的内容（不带#，便于阅读）
            self.log_message(f"Sent: {message}")

        except Exception as e:
            self.log_message(f"Send Error: {e}")

    # ==========================================================================
    # 第19部分：清空日志
    # ==========================================================================

    def clear_messages(self):
        """清空消息日志和重置检测"""
        # delete: 删除文本 "1.0"从第一行开始，"end"到末尾
        self.msg_textbox.delete("1.0", "end")

        global _last_screenshot_hash
        _last_screenshot_hash = None  # 重置哈希，重新开始检测

        self.log_message("Log cleared")

# ==============================================================================
# 第20部分：程序入口
# ==============================================================================

# __name__: Python特殊变量，当前文件作为主程序运行时值为'__main__'
# 这段代码只在直接运行本文件时执行，import时不执行
if __name__ == '__main__':
    # 创建应用对象
    app = WeChatGUI()

    # mainloop: 进入事件循环，等待用户操作（阻塞程序）
    app.mainloop()

# ==============================================================================
# 附录：Python基础知识点总结
# ==============================================================================
"""
1. 变量和数据类型
   - 字符串: "hello" 或 'hello'
   - 数字: 100, 3.14
   - 布尔: True, False
   - 列表: [1, 2, 3]
   - 字典: {"name": "Tom"}

2. 流程控制
   - if/elif/else: 条件判断
   - for: 循环
   - while: 条件循环

3. 函数
   - def 函数名(参数): 定义函数
   - return: 返回值

4. 面向对象
   - class: 定义类
   - self: 代表实例本身
   - __init__: 构造函数

5. 模块导入
   - import 模块名
   - from 模块 import 函数

6. 线程
   - threading.Thread: 创建线程
   - .start(): 启动线程
"""