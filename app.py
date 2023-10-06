import argparse

import cv2
import numpy as np
import torch
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QAction
from PyQt5.QtWidgets import QHBoxLayout, QWidget, QVBoxLayout, QGroupBox, QGraphicsScene, QMessageBox

from interactive_demo.canvas import CanvasImage
from interactive_demo.controller import InteractiveController
from isegm.utils import exp


class InteractiveDemoApp(QtWidgets.QMainWindow):
    def __init__(self, args, model):
        super().__init__()
        self.setWindowTitle("Reviving Iterative Training with Mask Guidance for Interactive Segmentation")
        self.setGeometry(100, 100, 800, 600)
        self.brs_modes = ['NoBRS', 'RGB-BRS', 'DistMap-BRS', 'f-BRS-A', 'f-BRS-B', 'f-BRS-C']
        self.limit_longest_size = args.limit_longest_size

        # 把更新图像的方法也传入构造函数
        self.controller = InteractiveController(model, args.device,
                                                predictor_params={'brs_mode': 'NoBRS'},
                                                update_image_callback=self._update_image)

        self.scene = QGraphicsScene()
        self._init_state()
        self._add_menu()
        self._add_window()
        self._add_canvas()
        self._add_buttons()
        self.show()
        print("self.show()")

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
        self.menubar = self.menuBar()

        self.load_image_action = QAction('Load image', self)
        self.load_image_action.triggered.connect(self._load_image_callback)
        self.menubar.addAction(self.load_image_action)

        self.save_mask_action = QAction('Save mask', self)
        self.save_mask_action.triggered.connect(self._save_mask_callback)
        self.save_mask_action.setEnabled(False)  # Disable initially
        self.menubar.addAction(self.save_mask_action)

        self.load_mask_action = QAction('Load mask', self)
        self.load_mask_action.triggered.connect(self._load_mask_callback)
        self.load_mask_action.setEnabled(False)  # Disable initially
        self.menubar.addAction(self.load_mask_action)

        about_action = QAction('About', self)
        about_action.triggered.connect(self._about_callback)
        self.menubar.addAction(about_action)

        exit_action = QAction('Exit', self)
        exit_action.triggered.connect(self.close)
        self.menubar.addAction(exit_action)

    # 这是整个窗口的布局
    def _add_window(self):
        self.setWindowTitle("Interactive Demo App")
        self.setGeometry(0, 0, 1300, 1000)

        # 创建一个名为central_widget的QWidget，设置中央部件的大小为800x600像素 它将作为主窗口的中央部件，也就是主要的可见区域，设置为主窗口的中央部件
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.central_widget.setFixedSize(1000, 1000)

    # 建一个包含图像显示的画布，并将其添加到用户界面中。
    def _add_canvas(self):

        # 画布控件
        self.canvas_frame = QtWidgets.QGroupBox("Image", self)

        self.canvas_frame_layout = QtWidgets.QGridLayout(self.canvas_frame)
        self.canvas_frame_layout.setContentsMargins(5, 5, 5, 5)

        self.canvas = QtWidgets.QGraphicsView(self)

        # 取消禁用了滚动
        self.canvas.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.canvas.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)

        # 将canvas添加到canvas_frame_layout的第0行、第0列的位置
        self.canvas_frame_layout.addWidget(self.canvas, 0, 0)

        # canvas_frame中的所有小部件将按照canvas_frame_layout定义的规则进行排列和布局。
        self.canvas_frame.setLayout(self.canvas_frame_layout)

        self.image_on_canvas = None

    def _add_buttons(self):
        # 创建控制台部件
        self.control_frame = QtWidgets.QGroupBox("Controls", self)
        self.control_frame.setFixedSize(400, 1000)

        # 水平布局管理器 设置为 central_widget 的布局管理器
        horizontal_layout_H = QHBoxLayout()
        self.central_widget.setLayout(horizontal_layout_H)

        horizontal_layout_H.addWidget(self.canvas_frame)
        horizontal_layout_H.addWidget(self.control_frame)

        self.control_frame_layout = QtWidgets.QGridLayout(self.control_frame)
        # 左上右下
        self.control_frame_layout.setContentsMargins(5, 5, 5, 5)

        # 垂直布局管理器 设置为 control_frame 的布局管理器
        horizontal_layout_V = QVBoxLayout()
        self.control_frame.setLayout(horizontal_layout_V)

        # control_frame的第一个区域---->点击区域
        self.clicks_options_frame = QGroupBox("Clicks management", self.control_frame)
        self.clicks_options_frame.setFixedSize(375, 200)
        self.clicks_options_frame.setGeometry(10, 20, 375, 175)

        # 在布局的顶部插入 clicks_options_frame
        horizontal_layout_V.insertWidget(0, self.clicks_options_frame)

        Clickmanagement_shorizontal_layout_H = QHBoxLayout()
        self.clicks_options_frame.setLayout(Clickmanagement_shorizontal_layout_H)

        self.finish_object_button = QtWidgets.QPushButton("Finish\nobject", self.clicks_options_frame)
        self.finish_object_button.setGeometry(10, 20, 100, 150)
        self.finish_object_button.setEnabled(False)

        self.undo_click_button = QtWidgets.QPushButton("Undo click", self.clicks_options_frame)
        self.undo_click_button.setGeometry(10, 20, 100, 150)
        self.undo_click_button.setEnabled(False)

        self.reset_clicks_button = QtWidgets.QPushButton("Reset clicks", self.clicks_options_frame)
        self.reset_clicks_button.setGeometry(10, 20, 100, 150)
        self.reset_clicks_button.setEnabled(False)

        Clickmanagement_shorizontal_layout_H.insertWidget(0, self.finish_object_button)
        Clickmanagement_shorizontal_layout_H.insertWidget(1, self.undo_click_button)
        Clickmanagement_shorizontal_layout_H.insertWidget(2, self.reset_clicks_button)

        # ZoomIn Options
        self.zoomin_options_frame = QtWidgets.QGroupBox("ZoomIn options", self.control_frame)
        self.zoomin_options_frame.setFixedSize(375, 200)
        self.zoomin_options_frame.setGeometry(10, 20 + 175 + 35, 375, 175)

        # 在布局的第二个索引插入 zoomin_options_frame
        horizontal_layout_V.insertWidget(1, self.zoomin_options_frame)

        ZoomInoptions_shorizontal_layout_V = QVBoxLayout()
        self.zoomin_options_frame.setLayout(ZoomInoptions_shorizontal_layout_V)

        self.use_zoomin_checkbox = QtWidgets.QCheckBox("Use ZoomIn", self.zoomin_options_frame)
        self.use_zoomin_checkbox.setGeometry(10, 20, 150, 20)

        self.fixed_crop_checkbox = QtWidgets.QCheckBox("Fixed crop", self.zoomin_options_frame)
        self.fixed_crop_checkbox.setGeometry(10, 50, 150, 20)

        ZoomInoptions_shorizontal_layout_V.insertWidget(0, self.use_zoomin_checkbox)
        ZoomInoptions_shorizontal_layout_V.insertWidget(1, self.fixed_crop_checkbox)

        # BRS Options
        self.brs_options_frame = QtWidgets.QGroupBox("BRS options", self.control_frame)
        self.brs_options_frame.setGeometry(10, 460, 375, 100)

        # 在布局的第三个索引插入 brs_options_frame
        horizontal_layout_V.insertWidget(2, self.brs_options_frame)

        self.brs_mode_combo = QtWidgets.QComboBox(self.brs_options_frame)
        self.brs_mode_combo.setGeometry(10, 38, 100, 30)
        self.brs_mode_combo.addItems(self.brs_modes)

        self.network_clicks_label = QtWidgets.QLabel("Network\nclicks", self.brs_options_frame)
        self.network_clicks_label.setGeometry(130, 10, 150, 45)

        self.network_clicks_spinbox = QtWidgets.QSpinBox(self.brs_options_frame)
        self.network_clicks_spinbox.setGeometry(130, 55, 60, 30)

        self.lbfgs_iters_label = QtWidgets.QLabel("L-BFGS Max\nIterations", self.brs_options_frame)
        self.lbfgs_iters_label.setGeometry(220, 10, 150, 45)

        self.lbfgs_iters_spinbox = QtWidgets.QSpinBox(self.brs_options_frame)
        self.lbfgs_iters_spinbox.setGeometry(220, 55, 60, 30)

        # Predictions Threshold
        self.prob_thresh_frame = QtWidgets.QGroupBox("Predictions threshold", self.control_frame)
        self.prob_thresh_frame.setGeometry(10, 460 + 110, 375, 100)

        self.prob_thresh_slider = QtWidgets.QSlider(Qt.Horizontal, self.prob_thresh_frame)
        self.prob_thresh_slider.setGeometry(10, 30, 350, 30)
        self.prob_thresh_slider.setRange(0, 100)
        self.prob_thresh_slider.setValue(50)
        # self.prob_thresh_slider.valueChanged.connect(self._update_prob_thresh)

        # Alpha Blending Coefficient
        self.alpha_blend_frame = QtWidgets.QGroupBox("Alpha blending coefficient", self.control_frame)
        self.alpha_blend_frame.setGeometry(10, 460 + 110 + 110, 375, 100)

        self.alpha_blend_slider = QtWidgets.QSlider(Qt.Horizontal, self.alpha_blend_frame)
        self.alpha_blend_slider.setGeometry(10, 30, 350, 30)
        self.alpha_blend_slider.setRange(0, 100)
        self.alpha_blend_slider.setValue(50)
        # self.alpha_blend_slider.valueChanged.connect(self._update_blend_alpha)

        # Visualisation Click Radius
        self.click_radius_frame = QtWidgets.QGroupBox("Visualisation click radius", self.control_frame)
        self.click_radius_frame.setGeometry(10, 460 + 110 + 110 + 110, 375, 100)

        self.click_radius_slider = QtWidgets.QSlider(Qt.Horizontal, self.click_radius_frame)
        self.click_radius_slider.setGeometry(10, 30, 350, 30)
        self.click_radius_slider.setRange(0, 7)
        self.click_radius_slider.setValue(3)
        # self.click_radius_slider.valueChanged.connect(self._update_click_radius)

    def _load_image_callback(self):
        # 打开一个文件对话框，允许用户选择图像文件。用户可以在文件对话框中浏览文件系统，并选择符合指定文件类型的图像文件（例如，jpg、jpeg、png、bmp、tiff）。选择的文件名存储在变量filename中
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Choose an image", "",
                                                            "Images (*.jpg *.jpeg *.png *.bmp *.tiff);;All files (*)")

        if len(filename) > 0:
            # 通过OpenCV（cv2）库加载图像文件并将其转换为RGB颜色空间。加载的图像存储在变量image中
            image = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
            # 将新的图像设置为应用程序中的当前图像  self.image = image
            self.controller.set_image(image)
            # 目的是将这两个按钮从禁用状态切换到正常状态，使用户可以点击它们执行相应的操作，例如保存或加载遮罩
            self.save_mask_action.setEnabled(True)
            self.load_mask_action.setEnabled(True)

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
        # 这个方法用于将新的图像更新到应用程序的图像视图，根据control中的self.image生成可视化image
        image = self.controller.get_visualization(
            alpha_blend=self.state['alpha_blend'],
            click_radius=self.state['click_radius']
        )
        if self.image_on_canvas is None:
            self.image_on_canvas = CanvasImage(self.canvas_frame, self.canvas)
            self.image_on_canvas.register_click_callback(self._click_callback)

        self._set_click_dependent_widgets_state()

        if image is not None:
            # 展示图片
            self.image_on_canvas.reload_image(image, reset_canvas=True)

    # 当用户在图像上点击时，会触发_click_callback方法。
    def _click_callback(self, is_positive, x, y):
        # 使self.canvas获取焦点  "获取焦点" 是指用户界面中的某个可交互的元素（通常是输入框、按钮、小部件等）成为用户当前操作的目标
        self.canvas.setFocus()

        if self.image_on_canvas is None:
            QMessageBox.warning(self, "Warning", "Please load an image first")
            return

        # if self._check_entry(self)
        #  True，它会将用户的点击信息传递给 self.controller.add_click(x, y, is_positive)
        self.controller.add_click(x, y, is_positive)

    def _set_click_dependent_widgets_state(self):

        if self.controller.is_incomplete_mask:
            after_1st_click_state = True
        else:
            after_1st_click_state = False

        if self.controller.is_incomplete_mask:
            before_1st_click_state = True
        else:
            before_1st_click_state = False

        self.finish_object_button.setEnabled(after_1st_click_state)
        self.undo_click_button.setEnabled(after_1st_click_state)
        self.reset_clicks_button.setEnabled(after_1st_click_state)
        self.zoomin_options_frame.setEnabled(before_1st_click_state)
        self.brs_options_frame.setEnabled(before_1st_click_state)

        if self.state['brs_mode'] == 'NoBRS':
            self.network_clicks_label.setEnabled(False)
            self.network_clicks_spinbox.setEnabled(False)
            self.lbfgs_iters_label.setEnabled(False)
            self.lbfgs_iters_spinbox.setEnabled(False)

    def _update_prob_thresh(self, value):
        if self.controller.is_incomplete_mask:
            self.controller.prob_thresh = self.state['prob_thresh']
            self._update_image()

    def _update_blend_alpha(self, value):
        self._update_image()

    def _update_click_radius(self, *args):
        if self.image_on_canvas is None:
            return

        self._update_image()

    # 当用户选择"NoBRS"模式时，文本框net_clicks_entry和lbfgs_iters_entry会被禁用，同时它们的标签也会被禁用。
    # 当用户选"BRS"模式时，如果net_clicks_entry中的文本是"INF"，它会被设置为"8"，然后这些文本框和标签会被启用。最后，调用reset_predictor方法以重置预测器的状态。
    def change_brs_mode(self):
        selected_mode = self.brs_mode_combobox.currentText()

        if selected_mode == 'NoBRS':
            self.net_clicks_entry.setText('INF')
            self.net_clicks_entry.setDisabled(True)
            self.net_clicks_label.setDisabled(True)
            self.lbfgs_iters_entry.setDisabled(True)
            self.lbfgs_iters_label.setDisabled(True)
        else:
            if self.net_clicks_entry.text() == 'INF':
                self.net_clicks_entry.setText('8')
            self.net_clicks_entry.setEnabled(True)
            self.net_clicks_label.setEnabled(True)
            self.lbfgs_iters_entry.setEnabled(True)
            self.lbfgs_iters_label.setEnabled(True)

        self.reset_predictor()  # 重置预测器的状态

    def _reset_predictor(self, *args, **kwargs):
        brs_mode = self.state['brs_mode'].get()
        prob_thresh = self.state['prob_thresh'].get()
        net_clicks_limit = None if brs_mode == 'NoBRS' else self.state['predictor_params']['net_clicks_limit'].get()

        if self.state['zoomin_params']['use_zoom_in'].get():
            zoomin_params = {
                'skip_clicks': self.state['zoomin_params']['skip_clicks'].get(),
                'target_size': self.state['zoomin_params']['target_size'].get(),
                'expansion_ratio': self.state['zoomin_params']['expansion_ratio'].get()
            }
            if self.state['zoomin_params']['fixed_crop'].get():
                zoomin_params['target_size'] = (zoomin_params['target_size'], zoomin_params['target_size'])
        else:
            zoomin_params = None

        predictor_params = {
            'brs_mode': brs_mode,
            'prob_thresh': prob_thresh,
            'zoom_in_params': zoomin_params,
            'predictor_params': {
                'net_clicks_limit': net_clicks_limit,
                'max_size': self.limit_longest_size
            },
            'brs_opt_func_params': {'min_iou_diff': 1e-3},
            'lbfgs_params': {'maxfun': self.state['lbfgs_max_iters'].get()}
        }
        self.controller.reset_predictor(predictor_params)

    def set_click_dependent_widgets_state(self):
        after_1st_click_state = Qt.Normal if self.controller.is_incomplete_mask else Qt.Disabled
        before_1st_click_state = Qt.Disabled if self.controller.is_incomplete_mask else Qt.Normal

        self.finish_object_button.setEnabled(after_1st_click_state)
        self.undo_click_button.setEnabled(after_1st_click_state)
        self.reset_clicks_button.setEnabled(after_1st_click_state)
        self.zoomin_options_frame.setEnabled(before_1st_click_state)
        self.brs_options_frame.setEnabled(before_1st_click_state)

        if self.state['brs_mode'] == 'NoBRS':
            self.net_clicks_entry.setEnabled(False)
            self.net_clicks_label.setEnabled(False)
            self.lbfgs_iters_entry.setEnabled(False)
            self.lbfgs_iters_label.setEnabled(False)

    # 递归函数，用于检查窗口中的所有子组件（widget）以确保它们都通过某种条件检查。
    def _check_entry(self, widget):
        all_checked = True
        for child_widget in widget.findChildren(QWidget):
            all_checked = all_checked and self.check_entries(child_widget)

        if hasattr(widget, "check_bounds"):
            all_checked = all_checked and widget.check_bounds(widget.text())

        return all_checked


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
