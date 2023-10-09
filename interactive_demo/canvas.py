import os
import sys

from PyQt5.QtCore import Qt, QEvent, QRectF, QObject
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QScrollBar, QMainWindow, QGraphicsScene, QGraphicsPixmapItem
from pyqt5_plugins.examplebuttonplugin import QtGui


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


class AutoScrollBar(QScrollBar):
    """ A scrollbar that hides itself if it's not needed. Works with any layout manager. """

    def set(self, min_val, max_val):
        if min_val <= 0.0 and max_val >= 1.0:
            self.hide()
        else:
            self.show()
            super().setRange(min_val, max_val)

    def show(self):
        self.setVisible(True)

    def hide(self):
        self.setVisible(False)


class MyEventFilter(QObject):
    def __init__(self, canvas_container):
        super().__init__()
        self.canvas_container = canvas_container

    # eventFilter方法在PyQt中用于事件过滤
    # obj参数表示安装了事件过滤器的QObject对象，即接收事件的对象
    def eventFilter(self, obj, event):
        # 鼠标按钮按下事件
        if event.type() == QEvent.MouseButtonPress:
            if event.button() == Qt.LeftButton:
                self.canvas_container.left_mouse_button(event)
            elif event.button() == Qt.RightButton:
                self.canvas_container.right_mouse_button_pressed(event)
            elif event.button() == Qt.MiddleButton:
                self.canvas_container.right_mouse_button_pressed(event)
        # 鼠标滚轮事件
        elif event.type() == QEvent.Wheel:
            self.canvas_container.wheel(event)
        return super().eventFilter(obj, event)


class CanvasImage(QMainWindow):
    def __init__(self, canvas_frame, canvas):
        super().__init__()

        self.scaled = None
        self.__delta = 1.1
        self.wheel_scale = 1.0
        self.per_scale = 1.0
        self.__previous_state = 0

        self.canvas_frame = canvas_frame
        self.scene = QGraphicsScene()
        self.original_pixmap = None
        self.canvas = canvas

        self.container = None
        self._click_callback = None
        # 绑定事件处理函数 将事件过滤器安装到Canvas上
        self.event_filter = MyEventFilter(self)  # 将CanvasImage类的实例传递给事件过滤器
        self.canvas.installEventFilter(self.event_filter)

    def register_click_callback(self, click_callback):
        self._click_callback = click_callback

    def _show_image(self, image):
        if image is not None:
            height, width, channel = image.shape
            bytes_per_line = 3 * width
            self.q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            self.pixmap = QPixmap.fromImage(self.q_image)

            # 创建一个QGraphicsScene对象，并设置为self.canvas的场景
            self.scene = QGraphicsScene()
            self.canvas.setScene(self.scene)

            # 添加self.image_pixmap_item到场景中
            self.image_pixmap_item = QGraphicsPixmapItem()
            self.container = self.image_pixmap_item
            self.scene.addItem(self.image_pixmap_item)

            # 获取当前图像的宽度和高度  # 计算等比例缩放后的新尺寸  # 目标尺寸  使用scaled方法进行等比例缩放  高度固定等比例缩放
            target_size = 1000
            current_width = self.pixmap.width()
            current_height = self.pixmap.height()

            self.scaled = current_height / target_size

            # 让原本比较大的长度固定为1000
            if current_width > current_height:
                new_width = target_size
                new_height = int(current_height * (1 / self.scaled))
            else:
                new_height = target_size
                new_width = int(current_width * (1 / self.scaled))

            self.pixmap = self.pixmap.scaled(new_width, new_height)
            self.original_pixmap = self.pixmap

            # 设置图像到self.image_pixmap_item上
            self.image_pixmap_item.setPixmap(self.pixmap)

    def reload_image(self, image, reset_canvas=True):
        self.__original_image = image.copy()
        self.__current_image = image.copy()
        self._show_image(self.__original_image)

    def re_show_image(self):
        self._show_image(self.__original_image)
        print("self._show_image(self.__original_image)")

    def grid(self, **kw):
        self.__imframe.grid(**kw)  # place CanvasImage widget on the grid
        self.__imframe.grid(sticky='nswe')  # make frame container sticky
        self.__imframe.rowconfigure(0, weight=1)  # make canvas expandable
        self.__imframe.columnconfigure(0, weight=1)

    def _get_click_coordinates(self, event):
        # 获取鼠标事件在视图坐标系中的坐标
        pos = self.canvas.mapToScene(event.x(), event.y())
        x = pos.x()
        y = pos.y()
        print(y * self.scaled, x * self.scaled)
        return y * self.scaled, x * self.scaled

    def _reset_canvas_offset(self):
        # 设置滚动区域
        scroll_region = QRectF(0, 0, 5000, 5000)
        self.canvas.setSceneRect(scroll_region)

        # 重置视图的滚动位置
        self.canvas.setScene(self.scene)
        self.canvas.centerOn(0, 0)

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

    def wheel(self, event):

        coords = self._get_click_coordinates(event)
        y = coords[0]
        x = coords[1]

        # 检查鼠标滚轮向下滚动的情况。它会根据不同的条件来执行缩小图像的操作
        delta = event.angleDelta().y()  # 获取滚轮滚动的垂直方向的增量

        # 根据滚动方向来确定缩放比例的变化
        if delta < 0:  # 向下滚动，缩小图像
            self.wheel_scale = 0.9 * self.wheel_scale

        elif delta > 0:  # 向上滚动，放大图像
            self.wheel_scale = 1.1 * self.wheel_scale

        # 传递缩放的中心点  self.per_scale=1.1
        self._change_pixmmap_scale(self.wheel_scale, x, y)

    def _change_pixmmap_scale(self, relative_scale, x=0, y=0):

        height, width, channel = self.__original_image.shape

        if relative_scale > 20:
            return
        if relative_scale * width < self.canvas.width() and relative_scale * height < self.canvas.height():
            return

        # 创建一个变换矩阵，按照正确的顺序进行平移、缩放和反向平移
        transform = QtGui.QTransform()
        # 将中心点 (center_x, center_y) 移动到坐标系统的原点，以便在之后的缩放操作中以这个点为中心进行缩放
        transform.translate(x, y)
        transform.scale(relative_scale, relative_scale)

        # 使用变换矩阵对 QPixmap 进行缩放
        self.pixmap = self.original_pixmap.transformed(transform)
        # 获取缩放中心点坐标

        # 设置图像到self.image_pixmap_item上
        self.image_pixmap_item.setPixmap(self.pixmap)

    def left_mouse_button(self, event):
        if self._click_callback is None:
            return
        coords = self._get_click_coordinates(event)
        if coords is not None:
            self._click_callback(is_positive=True, x=coords[0], y=coords[1])

    def right_mouse_button_pressed(self, event):

        self._last_rb_click_event = event
        coords = self._get_click_coordinates(self._last_rb_click_event)
        self._click_callback(is_positive=False, x=coords[0], y=coords[1])

    def outside(self, x, y):
        bbox = self.canvas.coords(self.container)
        if bbox[0] < x < bbox[2] and bbox[1] < y < bbox[3]:
            return False
        else:
            return True

    def __keystroke(self, event):
        if event.state - self.__previous_state == 4:  # means that the Control key is pressed
            pass  # do nothing if Control key is pressed
        else:
            self.__previous_state = event.state  # remember the last keystroke state
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
