import torch
import numpy as np
from tkinter import messagebox

from isegm.inference import clicker
from isegm.inference.predictors import get_predictor
from isegm.utils.vis import draw_with_blend_and_clicks


class InteractiveController:
    def __init__(self, net, device, predictor_params, update_image_callback, prob_thresh=0.5):
        self.net = net
        self.prob_thresh = prob_thresh
        self.clicker = clicker.Clicker()
        self.states = []
        self.probs_history = []
        self.object_count = 0
        self._result_mask = None
        self._init_mask = None
        # 当前图像
        self.image = None
        self.predictor = None
        self.device = device
        # 将一个函数或方法的引用存储在类的属性中，
        self.update_image_callback = update_image_callback
        self.predictor_params = predictor_params
        self.reset_predictor()

    # 将新的图像设置为应用程序中的当前图像，并进行一些初始化工作，以便应用程序可以开始处理和显示新图像
    def set_image(self, image):
        self.image = image
        # 创建一个与输入图像相同大小的全零掩码
        self._result_mask = np.zeros(image.shape[:2], dtype=np.uint16)
        # 对象计数器初始化为零，用于跟踪当前识别的对象数量
        self.object_count = 0
        # 清除之前的交互状态，以准备开始下一个对象的交互
        self.reset_last_object(update_image=False)
        # 可能用于更新图像显示
        self.update_image_callback(reset_canvas=True)

    def set_mask(self, mask):
        if self.image.shape[:2] != mask.shape[:2]:
            messagebox.showwarning("Warning", "A segmentation mask must have the same sizes as the current image!")
            return

        if len(self.probs_history) > 0:
            self.reset_last_object()

        self._init_mask = mask.astype(np.float32)
        self.probs_history.append((np.zeros_like(self._init_mask), self._init_mask))
        self._init_mask = torch.tensor(self._init_mask, device=self.device).unsqueeze(0).unsqueeze(0)
        self.clicker.click_indx_offset = 1

    def add_click(self, x, y, is_positive):
        self.states.append({
            'clicker': self.clicker.get_state(),
            'predictor': self.predictor.get_states()
        })

        click = clicker.Click(is_positive=is_positive, coords=(y, x))
        self.clicker.add_click(click)
        pred = self.predictor.get_prediction(self.clicker, prev_mask=self._init_mask)
        if self._init_mask is not None and len(self.clicker) == 1:
            pred = self.predictor.get_prediction(self.clicker, prev_mask=self._init_mask)

        torch.cuda.empty_cache()

        if self.probs_history:
            self.probs_history.append((self.probs_history[-1][0], pred))
        else:
            self.probs_history.append((np.zeros_like(pred), pred))

        self.update_image_callback()

    def undo_click(self):
        if not self.states:
            return

        prev_state = self.states.pop()
        self.clicker.set_state(prev_state['clicker'])
        self.predictor.set_states(prev_state['predictor'])
        self.probs_history.pop()
        if not self.probs_history:
            self.reset_init_mask()
        self.update_image_callback()

    def partially_finish_object(self):
        object_prob = self.current_object_prob
        if object_prob is None:
            return

        self.probs_history.append((object_prob, np.zeros_like(object_prob)))
        self.states.append(self.states[-1])

        self.clicker.reset_clicks()
        self.reset_predictor()
        self.reset_init_mask()
        self.update_image_callback()

    def finish_object(self):
        if self.current_object_prob is None:
            return

        self._result_mask = self.result_mask
        self.object_count += 1
        self.reset_last_object()

    # 用于重置应用程序中与上一个对象交互相关的状态和数据，是在每次与对象的交互结束后，清除之前的交互状态，以准备开始下一个对象的交互
    def reset_last_object(self, update_image=True):
        # 存储交互状态和概率历史的数据结构  设置为空列表来清空之前的数据
        self.states = []
        self.probs_history = []
        # 除与点击相关的数据
        self.clicker.reset_clicks()
        # 重置与预测器相关的状态
        self.reset_predictor()
        # 重置初始遮罩
        self.reset_init_mask()
        if update_image:
            self.update_image_callback()

    def reset_predictor(self, predictor_params=None):
        if predictor_params is not None:
            self.predictor_params = predictor_params
        self.predictor = get_predictor(self.net, device=self.device,
                                       **self.predictor_params)
        if self.image is not None:
            self.predictor.set_input_image(self.image)

    def reset_init_mask(self):
        self._init_mask = None
        self.clicker.click_indx_offset = 0

    @property
    def current_object_prob(self):
        if self.probs_history:
            current_prob_total, current_prob_additive = self.probs_history[-1]
            return np.maximum(current_prob_total, current_prob_additive)
        else:
            return None

    @property
    def is_incomplete_mask(self):
        return len(self.probs_history) > 0

    @property
    def result_mask(self):
        result_mask = self._result_mask.copy()
        if self.probs_history:
            result_mask[self.current_object_prob > self.prob_thresh] = self.object_count + 1
        return result_mask

    # get_visualization方法用于生成图像的可视化，以便在用户界面中显示
    def get_visualization(self, alpha_blend, click_radius):
        if self.image is None:
            return None

        # 将当前结果掩码（self.result_mask）存储在
        # results_mask_for_vis变量中。这个掩码可能包含了已识别的对象的信息。
        results_mask_for_vis = self.result_mask

        # 使用draw_with_blend_and_clicks函数，将图像、掩码、透明度（alpha_blend）、
        # 点击列表（self.clicker.clicks_list）和点击半径（click_radius）传递给该函数，以生成带有混合效果和点击标记的图像
        vis = draw_with_blend_and_clicks(self.image, mask=results_mask_for_vis, alpha=alpha_blend,
                                         clicks_list=self.clicker.clicks_list, radius=click_radius)

        if self.probs_history:
            total_mask = self.probs_history[-1][0] > self.prob_thresh
            results_mask_for_vis[np.logical_not(total_mask)] = 0
            # 将更新后的results_mask_for_vis和透明度（alpha_blend）传递给该函数，以生成包含总体掩码和点击标记的最终可视化图像
            vis = draw_with_blend_and_clicks(vis, mask=results_mask_for_vis, alpha=alpha_blend)

        return vis
