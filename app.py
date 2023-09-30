import argparse
import sys

import cv2
import numpy as np
import torch
from PyQt5 import QtWidgets, QtGui, QtCore

from interactive_demo.canvas import CanvasImage
from interactive_demo.controller import InteractiveController
from interactive_demo.wrappers import FocusButton, FocusLabelFrame
from isegm.inference import utils
from isegm.utils import exp


class InteractiveDemoApp(QtWidgets.QWidget):
    def __init__(self, args, model):
        super().__init__()
        self.setWindowTitle("Reviving Iterative Training with Mask Guidance for Interactive Segmentation")
        self.setGeometry(100, 100, 800, 600)
        self.brs_modes = ['NoBRS', 'RGB-BRS', 'DistMap-BRS', 'f-BRS-A', 'f-BRS-B', 'f-BRS-C']
        self.limit_longest_size = args.limit_longest_size

        self.controller = InteractiveController(model, args.device,
                                                predictor_params={'brs_mode': 'NoBRS'},
                                                update_image_callback=self._update_image)

        # self._init_state()
        # self._add_menu()
        self.setupUi(self)
        # self._add_canvas()
        # self._add_buttons()

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

    # 根据pyqtdesigner生成的代码
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(500, 700)
        self.verticalLayout = QtWidgets.QVBoxLayout(Form)
        self.verticalLayout.setObjectName("verticalLayout")
        self.groupBox = QtWidgets.QGroupBox(Form)
        self.groupBox.setObjectName("groupBox")

        self.frame = QtWidgets.QFrame(self.groupBox)
        self.frame.setGeometry(QtCore.QRect(20, 60, 321, 371))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        # 设置自动扩展   如果父部件的大小增加，self.canvas_frame 会尽量扩展以填充更多的空间  也适用于高度
        self.frame.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        self.frame_2 = QtWidgets.QFrame(self.groupBox)
        self.frame_2.setGeometry(QtCore.QRect(600, 60, 257, 43))
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")

        self.horizontalLayout = QtWidgets.QHBoxLayout(self.frame_2)
        self.horizontalLayout.setObjectName("horizontalLayout")

        self.FinishObject = QtWidgets.QPushButton(self.groupBox)
        self.FinishObject.setGeometry(QtCore.QRect(40, 450, 181, 23))
        self.FinishObject.setObjectName("FinishObject")

        self.pushButton = QtWidgets.QPushButton(self.groupBox)
        self.pushButton.setGeometry(QtCore.QRect(40, 490, 181, 23))
        self.pushButton.setObjectName("pushButton")

        self.undo_clicks = QtWidgets.QPushButton(self.groupBox)
        self.undo_clicks.setGeometry(QtCore.QRect(40, 530, 181, 23))
        self.undo_clicks.setObjectName("undo_clicks")

        self.loadpicture = QtWidgets.QPushButton(self.groupBox, text='Load picture')
        self.loadpicture.clicked.connect(lambda: self._load_image_callback())

        self.loadpicture.setGeometry(QtCore.QRect(240, 20, 111, 31))
        self.loadpicture.setObjectName("loadpicture")

        self.savemask = QtWidgets.QPushButton(self.groupBox)
        self.savemask.setGeometry(QtCore.QRect(120, 20, 111, 31))
        self.savemask.setObjectName("savemask")

        self.exit = QtWidgets.QPushButton(self.groupBox)
        self.exit.setGeometry(QtCore.QRect(0, 20, 111, 31))
        self.exit.setObjectName("exit")

        self.verticalLayout.addWidget(self.groupBox)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

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

    def _add_menu(self):
        self.menubar_layout = QtWidgets.QHBoxLayout(self)

        self.load_image_btn = FocusButton(self)
        self.menubar_layout.addWidget(self.load_image_btn)

        self.setLayout(self.menubar_layout)

    def _add_canvas(self):
        self.canvas_frame = FocusLabelFrame(self)
        self.canvas = QtWidgets.QWidget(self.canvas_frame)
        self.canvas.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.canvas.setFixedSize(400, 400)
        self.image_on_canvas = None
        self.canvas_frame_layout = QtWidgets.QVBoxLayout(self.canvas_frame)
        label = QtWidgets.QLabel("Image Canvas")  # 替换成您想要的组件名称
        self.canvas_frame_layout.addWidget(label)
        # 将 self.canvas 组件添加到名为 self.canvas_frame_layout 的布局管理器中
        self.canvas_frame_layout.addWidget(self.canvas)
        # self.canvas_frame_layout分配给组件self.canvas_frame
        self.canvas_frame.setLayout(self.canvas_frame_layout)
        # 子组件的对齐方式。在这里，对齐方式被设置为左对齐和顶部对齐
        self.canvas_frame_layout.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.canvas_frame_layout.setContentsMargins(5, 5, 5, 5)
        # 将 self.canvas_frame 放入父窗口的布局中
        self.layout().addWidget(self.canvas_frame)
        # 设置自动扩展   如果父部件的大小增加，self.canvas_frame 会尽量扩展以填充更多的空间  也适用于高度
        self.canvas_frame.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)


    def _add_buttons(self):
        self.control_frame = FocusLabelFrame(self)
        label = QtWidgets.QLabel("Controls")
        self.control_frame_layout = QtWidgets.QVBoxLayout(self.control_frame)
        self.control_frame_layout.addWidget(label)
        self.control_frame_layout.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTop)
        self.control_frame_layout.setContentsMargins(5, 5, 5, 5)

        clicks_options_frame = QtWidgets.QGroupBox("Clicks management")

        self.finish_object_button = QtWidgets.QPushButton('Finish\nobject')
        self.finish_object_button.setStyleSheet('background-color: #b6d7a8; color: black;')
        self.finish_object_button.setFixedSize(100, 40)
        self.finish_object_button.setDisabled(True)
        self.finish_object_button.clicked.connect(self.controller.finish_object)

        self.undo_click_button = QtWidgets.QPushButton('Undo click')
        self.undo_click_button.setStyleSheet('background-color: #ffe599; color: black;')
        self.undo_click_button.setFixedSize(100, 40)
        self.undo_click_button.setDisabled(True)
        self.undo_click_button.clicked.connect(self.controller.undo_click)

        self.reset_clicks_button = QtWidgets.QPushButton('Reset clicks')
        self.reset_clicks_button.setStyleSheet('background-color: #ea9999; color: black;')
        self.reset_clicks_button.setFixedSize(100, 40)
        self.reset_clicks_button.setDisabled(True)
        self.reset_clicks_button.clicked.connect(self._reset_last_object)

        self.clicks_layout = QtWidgets.QHBoxLayout()
        self.clicks_layout.setContentsMargins(5, 5, 5, 5)
        self.clicks_layout.addWidget(self.finish_object_button)
        self.clicks_layout.addWidget(self.undo_click_button)
        self.clicks_layout.addWidget(self.reset_clicks_button)

        clicks_options_frame.setLayout(self.clicks_layout)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(clicks_options_frame)

        clicks_widget = QtWidgets.QWidget()
        clicks_widget.setLayout(self.clicks_layout)

        # Add the QWidget to the QGroupBox
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(clicks_widget)
        clicks_options_frame.setLayout(layout)

        # Add the QGroupBox to the control_frame
        control_layout = QtWidgets.QVBoxLayout()
        control_layout.addWidget(clicks_options_frame)
        self.control_frame.setLayout(control_layout)

    def _load_image_callback(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Choose an image", "",
                                                            "Images (*.jpg *.jpeg *.png *.bmp *.tiff);;All files (*)")
        if filename:
            image = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
            self.controller.set_image(image)
            self.save_mask_btn.setEnabled(True)
            self.load_mask_btn.setEnabled(True)

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

    def _update_image(self, reset_canvas=False):
        image = self.controller.get_visualization(alpha_blend=self.state['alpha_blend'],
                                                  click_radius=self.state['click_radius'])
        if self.image_on_canvas is None:
            self.image_on_canvas = CanvasImage(self.canvas_frame, self.canvas)
            self.image_on_canvas.register_click_callback(self._click_callback)

        self._set_click_dependent_widgets_state()
        if image is not None:
            height, width, channel = image.shape
            bytes_per_line = 3 * width
            q_image = QtGui.QImage(image.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(q_image)
            self.canvas.setPixmap(pixmap)

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
