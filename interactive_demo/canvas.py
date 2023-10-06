import math
import os
import sys
import time

from PIL import Image, ImageTk
from PyQt5.QtCore import Qt, QEvent, QRectF, QObject
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QScrollBar, QMainWindow, QGraphicsScene, QGraphicsPixmapItem


def handle_exception(exit_code=0):
    """ Use: @land.logger.handle_exception(0)
        before every function which could cast an exception """

    def wrapper(func):
        def inner(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except:
                if exit_code != 0:  # if zero, don't exit from the program
                    sys.exit(exit_code)  # exit from the program

        return inner

    return wrapper


# 根据滚动范围的大小来自动隐藏或显示滚动条，以提供更好的用户体验。这在某些情况下可以防止不必要的滚动条显示。
class AutoScrollBar(QScrollBar):
    def __init__(self, orientation, parent=None):
        super().__init__(orientation, parent)
        self.hide_if_not_needed()

    def hide_if_not_needed(self):
        lo = self.minimum()
        hi = self.maximum()
        if lo <= 0.0 and hi >= 1.0:
            self.setVisible(False)
        else:
            self.setVisible(True)

    def setRange(self, min_val, max_val):
        if min_val != max_val:
            super().setRange(min_val, max_val)
        else:
            raise ValueError("Cannot set a range with min_val equal to max_val")
        self.hide_if_not_needed()


class MyEventFilter(QObject):
    def __init__(self, canvas_container):
        super().__init__()
        self.canvas_container = canvas_container

    # eventFilter方法在PyQt中用于事件过滤
    # obj参数表示安装了事件过滤器的QObject对象，即接收事件的对象
    def eventFilter(self, obj, event):
        print(type(obj))
        print(event.type())
        # 处理窗口大小改变事件
        if event.type() == QEvent.Resize:
            obj.__size_changed(event)
        # 鼠标按钮按下事件
        elif event.type() == QEvent.MouseButtonPress:
            if event.button() == Qt.LeftButton:
                print("左键点击了")
                self.canvas_container.left_mouse_button(event)
            elif event.button() == Qt.RightButton:
                obj.__right_mouse_button_pressed(event)
            elif event.button() == Qt.MiddleButton:
                obj.__right_mouse_button_pressed(event)
        # 鼠标按钮释放事件
        elif event.type() == QEvent.MouseButtonRelease:
            if event.button() == Qt.RightButton:
                obj.__right_mouse_button_released(event)
            elif event.button() == Qt.MiddleButton:
                obj.__right_mouse_button_released(event)
        # 鼠标移动事件
        elif event.type() == QEvent.MouseMove:
            if event.buttons() == Qt.RightButton:
                obj.__right_mouse_button_motion(event)
            elif event.buttons() == Qt.MiddleButton:
                obj.__right_mouse_button_motion(event)
        # 鼠标滚轮事件
        # elif event.type() == QEvent.Wheel:
        #     obj.__wheel(event)
        return super().eventFilter(obj, event)


class CanvasImage(QMainWindow):
    def __init__(self, canvas_frame, canvas):
        super().__init__()

        self.current_scale = 1.0
        self.__delta = 1.2
        self.__previous_state = 0
        self.canvas_frame = canvas_frame

        self.scene = QGraphicsScene()

        self.canvas = canvas

        self.canvas.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.canvas.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.hbar = AutoScrollBar(Qt.Horizontal, canvas_frame)
        self.vbar = AutoScrollBar(Qt.Vertical, canvas_frame)

        self.canvas.setHorizontalScrollBar(self.hbar)
        self.canvas.setVerticalScrollBar(self.vbar)

        self.hbar.valueChanged.connect(self.__scroll_x)
        self.vbar.valueChanged.connect(self.__scroll_y)

        self.container = None
        self._click_callback = None
        # 绑定事件处理函数 将事件过滤器安装到Canvas上
        self.event_filter = MyEventFilter(self)  # 将CanvasImage类的实例传递给事件过滤器
        self.canvas.installEventFilter(self.event_filter)

    def register_click_callback(self, click_callback):
        self._click_callback = click_callback
        print("self._click_callback = click_callback")

    def _show_image(self, image):
        if image is not None:
            height, width, channel = image.shape
            bytes_per_line = 3 * width
            q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)

            # 创建一个QGraphicsScene对象，并设置为self.canvas的场景
            self.scene = QGraphicsScene()
            self.canvas.setScene(self.scene)

            # 添加self.image_item到场景中
            self.image_item = QGraphicsPixmapItem()
            self.container = self.image_item
            self.scene.addItem(self.image_item)

            # 获取当前图像的宽度和高度  # 计算等比例缩放后的新尺寸  # 目标尺寸  使用scaled方法进行等比例缩放  高度固定等比例缩放
            target_size = 1000
            current_width = pixmap.width()
            current_height = pixmap.height()
            # print("current_width")
            # print(current_width)
            # print("current_height")
            # print(current_height)

            self.scaled = current_height / target_size

            if current_width > current_height:
                new_width = target_size
                new_height = int(current_height * (target_size / current_width))
            else:
                new_height = target_size
                new_width = int(current_width * (target_size / current_height))
                # print("new_width")
                # print(new_width)
                # print("new_height")
                # print(new_height)
            pixmap = pixmap.scaled(new_width, new_height)

            # 设置图像到self.image_item上
            self.image_item.setPixmap(pixmap)

    def reload_image(self, image, reset_canvas=True):

        self.__original_image = image.copy()
        self.__current_image = image.copy()
        self.canvas.setFocus()
        self._show_image(self.__original_image)

    def grid(self, **kw):
        """ Put CanvasImage widget on the parent widget """
        self.__imframe.grid(**kw)  # place CanvasImage widget on the grid
        self.__imframe.grid(sticky='nswe')  # make frame container sticky
        self.__imframe.rowconfigure(0, weight=1)  # make canvas expandable
        self.__imframe.columnconfigure(0, weight=1)

    # def __show_image(self):
    #     box_image = self.canvas.coords(self.container)  # get image area
    #     box_canvas = (self.canvas.canvasx(0),  # get visible area of the canvas
    #                   self.canvas.canvasy(0),
    #                   self.canvas.canvasx(self.canvas.winfo_width()),
    #                   self.canvas.canvasy(self.canvas.winfo_height()))
    #     box_img_int = tuple(map(int, box_image))  # convert to integer or it will not work properly
    #     # Get scroll region box
    #     box_scroll = [min(box_img_int[0], box_canvas[0]), min(box_img_int[1], box_canvas[1]),
    #                   max(box_img_int[2], box_canvas[2]), max(box_img_int[3], box_canvas[3])]
    #     # Horizontal part of the image is in the visible area
    #     if box_scroll[0] == box_canvas[0] and box_scroll[2] == box_canvas[2]:
    #         box_scroll[0] = box_img_int[0]
    #         box_scroll[2] = box_img_int[2]
    #     # Vertical part of the image is in the visible area
    #     if box_scroll[1] == box_canvas[1] and box_scroll[3] == box_canvas[3]:
    #         box_scroll[1] = box_img_int[1]
    #         box_scroll[3] = box_img_int[3]
    #     # Convert scroll region to tuple and to integer
    #     self.canvas.configure(scrollregion=tuple(map(int, box_scroll)))  # set scroll region
    #     x1 = max(box_canvas[0] - box_image[0], 0)  # get coordinates (x1,y1,x2,y2) of the image tile
    #     y1 = max(box_canvas[1] - box_image[1], 0)
    #     x2 = min(box_canvas[2], box_image[2]) - box_image[0]
    #     y2 = min(box_canvas[3], box_image[3]) - box_image[1]
    #
    #     if int(x2 - x1) > 0 and int(y2 - y1) > 0:  # show image if it in the visible area
    #         border_width = 2
    #         sx1, sx2 = x1 / self.current_scale, x2 / self.current_scale
    #         sy1, sy2 = y1 / self.current_scale, y2 / self.current_scale
    #         crop_x, crop_y = max(0, math.floor(sx1 - border_width)), max(0, math.floor(sy1 - border_width))
    #         crop_w, crop_h = math.ceil(sx2 - sx1 + 2 * border_width), math.ceil(sy2 - sy1 + 2 * border_width)
    #         crop_w = min(crop_w, self.__original_image.width - crop_x)
    #         crop_h = min(crop_h, self.__original_image.height - crop_y)
    #
    #         __current_image = self.__original_image.crop((crop_x, crop_y,
    #                                                       crop_x + crop_w, crop_y + crop_h))
    #         crop_zw = int(round(crop_w * self.current_scale))
    #         crop_zh = int(round(crop_h * self.current_scale))
    #         zoom_sx, zoom_sy = crop_zw / crop_w, crop_zh / crop_h
    #         crop_zx, crop_zy = crop_x * zoom_sx, crop_y * zoom_sy
    #         self.real_scale = (zoom_sx, zoom_sy)
    #
    #         interpolation = Image.NEAREST if self.current_scale > 2.0 else Image.ANTIALIAS
    #         __current_image = __current_image.resize((crop_zw, crop_zh), interpolation)
    #         zx1, zy1 = x1 - crop_zx, y1 - crop_zy
    #         zx2 = min(zx1 + self.canvas.winfo_width(), __current_image.width)
    #         zy2 = min(zy1 + self.canvas.winfo_height(), __current_image.height)
    #
    #         self.__current_image = __current_image.crop((zx1, zy1, zx2, zy2))
    #
    #         imagetk = ImageTk.PhotoImage(self.__current_image)
    #         imageid = self.canvas.create_image(max(box_canvas[0], box_img_int[0]),
    #                                            max(box_canvas[1], box_img_int[1]),
    #                                            anchor='nw', image=imagetk)
    #         self.canvas.lower(imageid)  # set image into background
    #         self.canvas.imagetk = imagetk  # keep an extra reference to prevent garbage-collection

    def _get_click_coordinates(self, event):
        print("我进来了")
        # 获取鼠标事件在视图坐标系中的坐标
        pos = self.canvas.mapToScene(event.x(), event.y())
        x = pos.x()
        y = pos.y()
        # if self.outside(x, y):
        #     return None

        return y*self.scaled, x*self.scaled
    # ================================================ Canvas Routines =================================================
    def _reset_canvas_offset(self):
        # 设置滚动区域
        scroll_region = QRectF(0, 0, 5000, 5000)
        self.canvas.setSceneRect(scroll_region)

        # 重置视图的滚动位置
        self.canvas.setScene(self.scene)
        self.canvas.centerOn(0, 0)

    def _change_canvas_scale(self, relative_scale, x=0, y=0):
        new_scale = self.current_scale * relative_scale

        if new_scale > 20:
            return

        if new_scale * self.__original_image.width < self.canvas.winfo_width() and \
                new_scale * self.__original_image.height < self.canvas.winfo_height():
            return

        self.current_scale = new_scale
        self.canvas.scale('all', x, y, relative_scale, relative_scale)  # rescale all objects

    # noinspection PyUnusedLocal
    def __scroll_x(self, *args, **kwargs):
        """ Scroll canvas horizontally and redraw the image """
        self.canvas.xview(*args)  # scroll horizontally
        self.__show_image()  # redraw the image

    # noinspection PyUnusedLocal
    def __scroll_y(self, *args, **kwargs):
        """ Scroll canvas vertically and redraw the image """
        self.canvas.yview(*args)  # scroll vertically
        self.__show_image()  # redraw the image

    def __size_changed(self):
        new_scale_w = self.canvas.winfo_width() / (self.current_scale * self.__original_image.width)
        new_scale_h = self.canvas.winfo_height() / (self.current_scale * self.__original_image.height)
        new_scale = min(new_scale_w, new_scale_h)
        if new_scale > 1.0:
            self._change_canvas_scale(new_scale)
        self.__show_image()

    # ================================================ Mouse callbacks =================================================
    def __wheel(self, event):
        """ Zoom with mouse wheel """

        # 设置或者是得到鼠标相对于目标事件的父元素的外边界在x坐标上的位置。
        x = self.canvas.canvasx(event.x)  # get coordinates of the event on the canvas
        y = self.canvas.canvasy(event.y)
        if self.outside(x, y): return  # zoom only inside image area

        scale = 1.0
        # Respond to Linux (event.num) or Windows (event.delta) wheel event
        if event.num == 5 or event.delta == -120 or event.delta == 1:  # scroll down, zoom out, smaller
            scale /= self.__delta
        if event.num == 4 or event.delta == 120 or event.delta == -1:  # scroll up, zoom in, bigger
            scale *= self.__delta

        self._change_canvas_scale(scale, x, y)
        self.__show_image()

    def left_mouse_button(self, event):
        print("test")
        if self._click_callback is None:
            return

        coords = self._get_click_coordinates(event)
        print(coords)
        if coords is not None:
            self._click_callback(is_positive=True, x=coords[0], y=coords[1])

    def __right_mouse_button_pressed(self, event):
        """ Remember previous coordinates for scrolling with the mouse """
        self._last_rb_click_time = time.time()
        self._last_rb_click_event = event
        self.canvas.scan_mark(event.x, event.y)

    def __right_mouse_button_released(self, event):
        time_delta = time.time() - self._last_rb_click_time
        move_delta = math.sqrt((event.x - self._last_rb_click_event.x) ** 2 +
                               (event.y - self._last_rb_click_event.y) ** 2)
        if time_delta > 0.5 or move_delta > 3:
            return

        if self._click_callback is None:
            return

        coords = self._get_click_coordinates(self._last_rb_click_event)

        if coords is not None:
            self._click_callback(is_positive=False, x=coords[0], y=coords[1])

    def __right_mouse_button_motion(self, event):
        """ Drag (move) canvas to the new position """
        move_delta = math.sqrt((event.x - self._last_rb_click_event.x) ** 2 +
                               (event.y - self._last_rb_click_event.y) ** 2)
        if move_delta > 3:
            self.canvas.scan_dragto(event.x, event.y, gain=1)
            self.__show_image()  # zoom tile and show it on the canvas

    def outside(self, x, y):
        """ Checks if the point (x,y) is outside the image area """
        bbox = self.canvas.coords(self.container)  # get image area
        if bbox[0] < x < bbox[2] and bbox[1] < y < bbox[3]:
            return False  # point (x,y) is inside the image area
        else:
            return True  # point (x,y) is outside the image area

    # ================================================= Keys Callback ==================================================
    def __keystroke(self, event):
        """ Scrolling with the keyboard.
            Independent from the language of the keyboard, CapsLock, <Ctrl>+<key>, etc. """
        if event.state - self.__previous_state == 4:  # means that the Control key is pressed
            pass  # do nothing if Control key is pressed
        else:
            self.__previous_state = event.state  # remember the last keystroke state
            # Up, Down, Left, Right keystrokes
            self.keycodes = {}  # init key codes
            if os.name == 'nt':  # Windows OS
                self.keycodes = {
                    'd': [68, 39, 102],
                    'a': [65, 37, 100],
                    'w': [87, 38, 104],
                    's': [83, 40, 98],
                }
            else:  # Linux OS
                self.keycodes = {
                    'd': [40, 114, 85],
                    'a': [38, 113, 83],
                    'w': [25, 111, 80],
                    's': [39, 116, 88],
                }
            if event.keycode in self.keycodes['d']:  # scroll right, keys 'd' or 'Right'
                self.__scroll_x('scroll', 1, 'unit', event=event)
            elif event.keycode in self.keycodes['a']:  # scroll left, keys 'a' or 'Left'
                self.__scroll_x('scroll', -1, 'unit', event=event)
            elif event.keycode in self.keycodes['w']:  # scroll up, keys 'w' or 'Up'
                self.__scroll_y('scroll', -1, 'unit', event=event)
            elif event.keycode in self.keycodes['s']:  # scroll down, keys 's' or 'Down'
                self.__scroll_y('scroll', 1, 'unit', event=event)
