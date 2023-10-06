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
        # 存储最终的分割掩码的NumPy数组  数据类型np.uint16
        self._result_mask = None
        # 存储分割掩码（segmentationmask）的初始值的NumPy数组。 数据类型np.float32
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
        # 创建一个与输入图像相同大小的全零掩码 在构造函数中，它被初始化为一个全零的数组
        self._result_mask = np.zeros(image.shape[:2], dtype=np.uint16)
        # 对象计数器初始化为零，用于跟踪当前识别的对象数量
        self.object_count = 0

        self.reset_last_object(update_image=False)
        # 清除之前的交互状态，以准备开始下一个对象的交互
        self.update_image_callback(reset_canvas=True)

    def set_mask(self, mask):
        if self.image.shape[:2] != mask.shape[:2]:
            messagebox.showwarning("Warning", "A segmentation mask must have the same sizes as the current image!")
            return

        if len(self.probs_history) > 0:
            self.reset_last_object()

        # 将名为 mask 的 NumPy 数组转换为 np.float32 数据类型，并将结果存储在 self._init_mask 变量中
        self._init_mask = mask.astype(np.float32)

        self.probs_history.append((np.zeros_like(self._init_mask), self._init_mask))
        #  _init_mask 转化为 PyTorch 张量
        self._init_mask = torch.tensor(self._init_mask, device=self.device).unsqueeze(0).unsqueeze(0)

        self.clicker.click_indx_offset = 1

    # 处理用户在图像上的点击操作，用于进行交互式图像分割。
    def add_click(self, x, y, is_positive):
        # 将当前的状态保存到self.states列表中。这个状态包括点击器（clicker）和预测器（predictor）的当前状态
        self.states.append({
            'clicker': self.clicker.get_state(),
            'predictor': self.predictor.get_states()
        })

        # 创建一个clicker.Click对象，表示用户的点击操作。is_positive参数表示点击是否是正面（positive）,coords 参数表示点击的坐标位置(x, y)
        click = clicker.Click(is_positive=is_positive, coords=(x, y))
        # 将用户的点击操作添加到点击器（clicker）中
        self.clicker.add_click(click)
        pred = self.predictor.get_prediction(self.clicker, prev_mask=self._init_mask)

        # 获取预测结果。预测的结果存储在pred变量中。预测器（predictor）根据用户的点击和初始掩码（self._init_mask）来生成预测
        if self._init_mask is not None and len(self.clicker) == 1:
            # 这段代码的作用是将用户的点击操作以及先前的分割结果传递给分割模型，以获取最终的图像分割预测结果。这对于交互式图像分割应用程序非常重要，因为用户可以不断添加点击以改进分割结果
            pred = self.predictor.get_prediction(self.clicker, prev_mask=self._init_mask)

        # 清空CUDA内存缓存，以释放GPU内存。这通常用于优化内存使用，确保程序在GPU上运行时不会耗尽内存。
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

    # 这个属性用于获取当前对象的概率分布，它会根据self.probs_history中的数据计算并返回相应的值
    @property
    def current_object_prob(self):
        if self.probs_history:
            # 如果self.probs_history存在，就从self.probs_history[-1]中获取当前的概率分布。
            # 通常，self.probs_history[-1]包含两个部分的概率信息：current_prob_total和current_prob_additive
            current_prob_total, current_prob_additive = self.probs_history[-1]
            # 个元素进行比较，返回一个新的数组，其中每个元素是两者中相应位置上的较大值。
            return np.maximum(current_prob_total, current_prob_additive)
        else:
            return None


    @property
    def is_incomplete_mask(self):
        return len(self.probs_history) > 0

    @property
    def result_mask(self):
        result_mask = self._result_mask.copy()
        # 存储了模型的历史预测概率
        if self.probs_history:
            # 每个像素是否大于self.prob_thresh在self.current_object_prob中的像素值
            # 大于self.prob_thresh的位置上将result_mask中的像素标记为self.object_count + 1
            result_mask[self.current_object_prob > self.prob_thresh] = self.object_count + 1
        return result_mask

    # 生成一个可视化图像，该图像反映了原始图像、点击点和模型结果的混合。这可以用于可视化模型的效果以及用户交互的结果。
    def get_visualization(self, alpha_blend, click_radius):
        if self.image is None:
            return None

        # 将当前结果掩码（self.result_mask）存储在results_mask_for_vis变量中。这个掩码可能包含了已识别的对象的信息。
        # result_mask被更新过
        results_mask_for_vis = self.result_mask
        # print("get_visualization中")
        # print("self.image",end="")
        # print(self.image)
        # print("results_mask_for_vis", end="")
        # print(results_mask_for_vis)
        # print("alpha_blend", end="")
        # print(alpha_blend)
        # print("self.clicker.clicks_list", end="")
        # print(self.clicker.clicks_list)
        # print("click_radius", end="")
        # print(click_radius)
        # 使用draw_with_blend_and_clicks函数，将图像、掩码、透明度（alpha_blend）、点击列表（self.clicker.clicks_list）和点击半径（click_radius）传递给该函数，以生成带有混合效果和点击标记的图像
        vis = draw_with_blend_and_clicks(self.image, mask=results_mask_for_vis, alpha=alpha_blend,
                                         clicks_list=self.clicker.clicks_list, radius=click_radius)
        # print("self.probs_history", end="")
        # print(self.probs_history)

        if self.probs_history:
            total_mask = self.probs_history[-1][0] > self.prob_thresh
            # print("total_mask",end="")
            # print(total_mask)
            results_mask_for_vis[np.logical_not(total_mask)] = 0
            # 将更新后的results_mask_for_vis和透明度（alpha_blend）传递给该函数，以生成包含总体掩码和点击标记的最终可视化图像
            vis = draw_with_blend_and_clicks(vis, mask=results_mask_for_vis, alpha=alpha_blend)
        # print("vis",end="")
        # print(vis)
        return vis
