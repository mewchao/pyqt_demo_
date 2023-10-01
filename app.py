import argparse
import sys

import cv2
import numpy as np
import torch
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QAction, QGraphicsView, QGraphicsScene, QSizePolicy, QFrame, QGraphicsPixmapItem, \
    QPushButton, QHBoxLayout, QWidget, QVBoxLayout, QGroupBox

from interactive_demo.controller import InteractiveController
from interactive_demo.wrappers import FocusLabelFrame
from isegm.inference import utils
from isegm.utils import exp


class InteractiveDemoApp(QtWidgets.QMainWindow):
    def __init__(self, args, model):
        super().__init__()
        self.setWindowTitle("Reviving Iterative Training with Mask Guidance for Interactive Segmentation")
        self.setGeometry(100, 100, 800, 600)
        self.brs_modes = ['NoBRS', 'RGB-BRS', 'DistMap-BRS', 'f-BRS-A', 'f-BRS-B', 'f-BRS-C']
        self.limit_longest_size = args.limit_longest_size

        self.controller = InteractiveController(model, args.device,
                                                predictor_params={'brs_mode': 'NoBRS'},
                                                update_image_callback=self._update_image)

        # 创建保存遮罩按钮
        self.save_mask_btn = QPushButton('Save mask', self)
        self.save_mask_btn.clicked.connect(self._save_mask_callback)
        self.save_mask_btn.setEnabled(False)  # 设置按钮不可用

        # 创建加载遮罩按钮
        self.load_mask_btn = QPushButton('Load mask', self)
        self.load_mask_btn.clicked.connect(self._load_mask_callback)
        self.load_mask_btn.setEnabled(False)  # 设置按钮不可用

        self._init_state()
        self._add_menu()
        self._add_canvas()
        self._add_buttons()
        print("test")
        # def _update_image(self, reset_canvas=False)方法添加的
        self.image_item = QGraphicsPixmapItem()
        # self.canvas_scene.addItem(self.image_item)

    # 初始化应用程序的状态，包括一些布尔值、整数值、双精度浮点数以及字符串值的变量。
    # 这些变量似乎用于跟踪和控制应用程序的行为和用户界面的不同方面
    def _init_state(self):
        self.state = {
            'zoomin_params': {
                'use_zoom_in': True,
                'fixed_crop': True,
                'skip_clicks': -1,
                'target_size': min(400, self.limit_longest_size),
                'expansion_ratio': 1.4
            },
            'predictor_params': {
                'net_clicks_limit': 8
            },
            'brs_mode': 'NoBRS',
            'prob_thresh': 0.5,
            'lbfgs_max_iters': 20,
            'alpha_blend': 0.5,
            'click_radius': 3,
        }

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.groupBox.setTitle(_translate("Form", "Image"))
        self.FinishObject.setText(_translate("Form", "Finish Object"))
        self.pushButton.setText(_translate("Form", "Reset clicks"))
        self.undo_clicks.setText(_translate("Form", "Undo clicks"))
        self.loadpicture.setText(_translate("Form", "Loadpicture"))
        self.savemask.setText(_translate("Form", "Savemask"))
        self.exit.setText(_translate("Form", "Exit"))

    # 创建应用程序的菜单栏，并向菜单栏添加一些按钮。以下是该方法的主要功能：
    def _add_menu(self):
        # 需要根据具体的需求来实现按钮的点击事件处理方法
        menubar = self.menuBar()

        load_image_action = QAction('Load image', self)
        load_image_action.triggered.connect(self._load_image_callback)
        menubar.addAction(load_image_action)

        save_mask_action = QAction('Save mask', self)
        save_mask_action.triggered.connect(self._save_mask_callback)
        save_mask_action.setEnabled(False)  # Disable initially
        menubar.addAction(save_mask_action)

        load_mask_action = QAction('Load mask', self)
        load_mask_action.triggered.connect(self._load_mask_callback)
        load_mask_action.setEnabled(False)  # Disable initially
        menubar.addAction(load_mask_action)

        about_action = QAction('About', self)
        about_action.triggered.connect(self._about_callback)
        menubar.addAction(about_action)

        exit_action = QAction('Exit', self)
        exit_action.triggered.connect(self.close)
        menubar.addAction(exit_action)

    # 建一个包含图像显示的画布，并将其添加到用户界面中。
    def _add_canvas(self):
        # 这是整个窗口的布局
        self.setWindowTitle("Interactive Demo App")
        self.setGeometry(10, 10, 1000, 1000)

        # Canvas Frame
        self.canvas_frame = QtWidgets.QGroupBox("Image", self)

        # 左上角的位置是(30距离屏幕左边的距离, 50距离屏幕顶部的距离)，并且窗口的宽度为780像素，高度为500像素
        self.canvas_frame.setGeometry(30, 50, 180, 400)

        self.canvas_frame_layout = QtWidgets.QGridLayout(self.canvas_frame)
        self.canvas_frame_layout.setContentsMargins(5, 5, 5, 5)

        self.canvas = QtWidgets.QGraphicsView(self)
        self.canvas.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.canvas.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.canvas_frame_layout.addWidget(self.canvas, 0, 0)
        self.canvas_frame.setLayout(self.canvas_frame_layout)

        self.image_on_canvas = None

        self.show()

    def _add_buttons(self):
        # 展示control_frame，在整窗口个中和canvas_frame水平并列
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        horizontal_layout = QHBoxLayout()
        central_widget.setLayout(horizontal_layout)

        self.control_frame = QtWidgets.QGroupBox("Controls", self)
        self.control_frame.setGeometry(250, 50, 300, 400)

        horizontal_layout.addWidget(self.canvas_frame)
        horizontal_layout.addWidget(self.control_frame)

        self.control_frame_layout = QtWidgets.QGridLayout(self.control_frame)
        self.control_frame_layout.setContentsMargins(5, 5, 5, 5)

        # 展示clicks_options_frame，在control_frame中垂直展示
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        horizontal_layout = QVBoxLayout()
        central_widget.setLayout(horizontal_layout)

        self.control_frame = QGroupBox("Controls", central_widget)
        horizontal_layout.addWidget(self.control_frame)

        control_layout = QVBoxLayout(self.control_frame)
        self.control_frame.setLayout(control_layout)

        # Clicks Management
        self.clicks_options_frame = QGroupBox("Clicks management", self.control_frame)
        control_layout.addWidget(self.clicks_options_frame)

        # self.finish_object_button = QtWidgets.QPushButton("Finish\nobject", self.clicks_options_frame)
        # self.finish_object_button.setGeometry(1500, 20, 70, 40)
        # self.finish_object_button.setEnabled(False)
        #
        # self.undo_click_button = QtWidgets.QPushButton("Undo click", self.clicks_options_frame)
        # self.undo_click_button.setGeometry(1500, 20, 70, 40)
        # self.undo_click_button.setEnabled(False)

        self.show()

        # self.reset_clicks_button = QtWidgets.QPushButton("Reset clicks", self.clicks_options_frame)
        # self.reset_clicks_button.setGeometry(170, 20, 70, 40)
        # self.reset_clicks_button.setEnabled(False)
        #
        # # ZoomIn Options
        # self.zoomin_options_frame = QtWidgets.QGroupBox("ZoomIn options", self.control_frame)
        # self.zoomin_options_frame.setGeometry(270, 20, 250, 70)
        #
        # self.use_zoomin_checkbox = QtWidgets.QCheckBox("Use ZoomIn", self.zoomin_options_frame)
        # self.use_zoomin_checkbox.setGeometry(10, 20, 150, 20)
        #
        # self.fixed_crop_checkbox = QtWidgets.QCheckBox("Fixed crop", self.zoomin_options_frame)
        # self.fixed_crop_checkbox.setGeometry(10, 50, 150, 20)
        #
        # # BRS Options
        # self.brs_options_frame = QtWidgets.QGroupBox("BRS options", self.control_frame)
        # self.brs_options_frame.setGeometry(530, 20, 250, 70)
        #
        # self.brs_mode_combo = QtWidgets.QComboBox(self.brs_options_frame)
        # self.brs_mode_combo.setGeometry(10, 20, 100, 30)
        # self.brs_mode_combo.addItems(["Mode 1", "Mode 2", "Mode 3"])  # Replace with your modes
        #
        # self.network_clicks_label = QtWidgets.QLabel("Network clicks", self.brs_options_frame)
        # self.network_clicks_label.setGeometry(120, 20, 100, 30)
        #
        # self.network_clicks_spinbox = QtWidgets.QSpinBox(self.brs_options_frame)
        # self.network_clicks_spinbox.setGeometry(230, 20, 60, 30)
        #
        # # Other Options (Threshold, Alpha Blend, Click Radius)
        # # Add similar code for these options...
        #
        # # Predictions Threshold
        # self.prob_thresh_frame = QtWidgets.QGroupBox("Predictions threshold", self.control_frame)
        # self.prob_thresh_frame.setGeometry(10, 100, 250, 70)
        #
        # self.prob_thresh_slider = QtWidgets.QSlider(Qt.Horizontal, self.prob_thresh_frame)
        # self.prob_thresh_slider.setGeometry(10, 20, 230, 30)
        # self.prob_thresh_slider.setRange(0, 100)
        # self.prob_thresh_slider.setValue(50)
        # # self.prob_thresh_slider.valueChanged.connect(self._update_prob_thresh)
        #
        # # Alpha Blending Coefficient
        # self.alpha_blend_frame = QtWidgets.QGroupBox("Alpha blending coefficient", self.control_frame)
        # self.alpha_blend_frame.setGeometry(270, 100, 250, 70)
        #
        # self.alpha_blend_slider = QtWidgets.QSlider(Qt.Horizontal, self.alpha_blend_frame)
        # self.alpha_blend_slider.setGeometry(10, 20, 230, 30)
        # self.alpha_blend_slider.setRange(0, 100)
        # self.alpha_blend_slider.setValue(50)
        # # self.alpha_blend_slider.valueChanged.connect(self._update_blend_alpha)
        #
        # # Visualisation Click Radius
        # self.click_radius_frame = QtWidgets.QGroupBox("Visualisation click radius", self.control_frame)
        # self.click_radius_frame.setGeometry(530, 100, 250, 70)
        #
        # self.click_radius_slider = QtWidgets.QSlider(Qt.Horizontal, self.click_radius_frame)
        # self.click_radius_slider.setGeometry(10, 20, 230, 30)
        # self.click_radius_slider.setRange(0, 7)
        # self.click_radius_slider.setValue(3)
        # # self.click_radius_slider.valueChanged.connect(self._update_click_radius)

        self.show()

    def _load_image_callback(self):
        # 打开一个文件对话框，允许用户选择图像文件。用户可以在文件对话框中浏览文件系统，
        # 并选择符合指定文件类型的图像文件（例如，jpg、jpeg、png、bmp、tiff）。选择的文件名存储在变量filename中
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Choose an image", "",
                                                            "Images (*.jpg *.jpeg *.png *.bmp *.tiff);;All files (*)")

        if filename:
            # 通过OpenCV（cv2）库加载图像文件并将其转换为RGB颜色空间。加载的图像存储在变量image中
            image = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
            # 将新的图像设置为应用程序中的当前图像
            self.controller.set_image(image)
            # 目的是将这两个按钮从禁用状态切换到正常状态，使用户可以点击它们执行相应的操作，例如保存或加载遮罩
            self.save_mask_btn.setEnabled(True)
            self.load_mask_btn.setEnabled(True)
            self._update_image()

    def _save_mask_callback(self):
        mask = self.controller.result_mask
        if mask is not None:
            filename, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save the current mask as...", "",
                                                                "PNG image (*.png);;BMP image (*.bmp);;All files (*)")
            if filename:
                if mask.max() < 256:
                    mask = (mask.astype(np.uint8) * (255 // mask.max())).astype(np.uint8)
                cv2.imwrite(filename, mask)

    def _load_mask_callback(self):
        if not self.controller.net.with_prev_mask:
            QtWidgets.QMessageBox.warning(self, "Warning",
                                          "The current model doesn't support loading external masks. Please use ITER-M models for that purpose.")
            return
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Choose a mask image", "",
                                                            "Binary mask (png, bmp) (*.png *.bmp);;All files (*)")
        if filename:
            mask = cv2.imread(filename)[:, :, 0] > 127
            self.controller.set_mask(mask)
            self._update_image()

    def _about_callback(self):
        QtWidgets.QMessageBox.about(self, "About Demo", "Developed by:\nK.Sofiiuk and I. Petrov\nThe MIT License, 2021")

    def _reset_last_object(self):
        self.state['alpha_blend'] = 0.5
        self.state['prob_thresh'] = 0.5
        self.controller.reset_last_object()

    # 更新应用程序中的图像显示，以便将最新的可视化内容显示在界面上
    def _update_image(self, reset_canvas=False):
        # 这个方法用于将新的图像更新到应用程序的图像视图
        def _update_image(self, reset_canvas=False):
            image = self.controller.get_visualization(alpha_blend=self.state['alpha_blend'].get(),
                                                      click_radius=self.state['click_radius'].get())
            if image is not None:
                q_image = QImage(image.data, image.shape[1], image.shape[0], image.shape[1] * 3, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_image)
                self.image_item.setPixmap(pixmap)

            self._set_click_dependent_widgets_state()

    def _set_click_dependent_widgets_state(self):
        after_1st_click_state = QtWidgets.QPushButton.Enabled if self.controller.is_incomplete_mask else QtWidgets.QPushButton.Disabled
        before_1st_click_state = QtWidgets.QPushButton.Disabled if self.controller.is_incomplete_mask else QtWidgets.QPushButton.Enabled

        self.finish_object_button.setEnabled(after_1st_click_state)
        self.undo_click_button.setEnabled(after_1st_click_state)
        self.reset_clicks_button.setEnabled(after_1st_click_state)

        if self.state['brs_mode'] == 'NoBRS':
            self.net_clicks_entry.setEnabled(False)
            self.lbfgs_iters_entry.setEnabled(False)

    # Rest of the callback methods can be added similarly


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint', type=str, default='coco_lvis_h32_itermask.pth',
                        help='The path to the checkpoint. '
                             'This can be a relative path (relative to cfg.INTERACTIVE_MODELS_PATH) '
                             'or an absolute path. The file extension can be omitted.')

    parser.add_argument('--gpu', type=int, default=False,
                        help='Id of GPU to use.')

    parser.add_argument('--cpu', action='store_true', default=True,
                        help='Use only CPU for inference.')

    parser.add_argument('--limit-longest-size', type=int, default=800,
                        help='If the largest side of an image exceeds this value, '
                             'it is resized so that its largest side is equal to this value.')

    parser.add_argument('--cfg', type=str, default="config.yml",
                        help='The path to the config file.')

    args = parser.parse_args()
    if args.cpu:
        args.device = torch.device('cpu')
    else:
        args.device = torch.device(f'cuda:{args.gpu}')
    cfg = exp.load_config_file(args.cfg, return_edict=True)

    return args, cfg


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    args, cfg = parse_args()
    checkpoint_path = utils.find_checkpoint(cfg.INTERACTIVE_MODELS_PATH, args.checkpoint)
    model = utils.load_is_model(checkpoint_path, args.device, cpu_dist_maps=True)
    window = InteractiveDemoApp(args, model)
    window.show()
    sys.exit(app.exec_())
