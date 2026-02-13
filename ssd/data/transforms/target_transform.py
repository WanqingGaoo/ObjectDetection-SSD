import numpy as np
import torch

from ssd.utils import box_utils


class SSDTargetTransform:
    """
    SSD 模型的目标转换类：用于将原始的真实标签（gt_boxes、gt_labels）转换为 SSD 模型训练所需的目标数据（位置偏移量、匹配后的标签）。
    核心作用：
    1.  将原始真实边框（gt_boxes）与 SSD 模型预定义的先验框（priors）进行匹配。
    2.  转换边框格式（从角点格式 ↔ 中心格式），适配模型训练需求。
    3.  计算真实边框相对于匹配先验框的位置偏移量（而非直接使用原始边框坐标），作为模型位置回归的目标。
    """

    def __init__(self, center_form_priors, center_variance, size_variance, iou_threshold):
        """
        构造方法：初始化 SSD 目标转换所需的核心参数（先验框、方差、IOU 匹配阈值）
        Args:
            center_form_priors (torch.Tensor): 中心格式的先验框（default boxes），形状通常为 [num_priors, 4]。
                每个先验框的格式为 (cx, cy, w, h)：cx/cy 是框的中心坐标，w/h 是框的宽和高。
            center_variance (float): 中心坐标（cx, cy）转换时使用的方差，用于归一化位置偏移量，提升训练稳定性。
            size_variance (float): 尺寸（w, h）转换时使用的方差，作用同上（通常和 center_variance 取值相同，如 0.1）。
            iou_threshold (float): IOU 匹配阈值，用于判断真实边框与先验框是否匹配（如 0.5）。
                若两者 IOU 大于该阈值，认为先验框匹配到了真实目标，否则标记为背景。
        """
        # 保存中心格式的先验框（后续用于计算位置偏移量）
        self.center_form_priors = center_form_priors
        # 将先验框从中心格式转换为角点格式，用于后续与真实边框（通常为角点格式）进行 IOU 匹配
        # 角点格式：(xmin, ymin, xmax, ymax)，分别对应框的左上角和右下角坐标
        self.corner_form_priors = box_utils.center_form_to_corner_form(center_form_priors)
        # 保存中心坐标对应的方差
        self.center_variance = center_variance
        # 保存尺寸对应的方差
        self.size_variance = size_variance
        # 保存 IOU 匹配阈值
        self.iou_threshold = iou_threshold

    def __call__(self, gt_boxes, gt_labels):
        """
        可调用方法：核心逻辑实现，将原始真实标签转换为 SSD 模型训练所需的目标数据。
        Args:
            gt_boxes (np.ndarray / torch.Tensor): 原始真实边框，形状 [num_gt, 4]，角点格式 (xmin, ymin, xmax, ymax)
            gt_labels (np.ndarray / torch.Tensor): 原始真实类别标签，形状 [num_gt]，非背景类别
        Returns:
            locations (torch.Tensor): 先验框对应的位置偏移量，形状 [num_priors, 4]，中心格式偏移 (cx_off, cy_off, w_off, h_off)
            labels (torch.Tensor): 先验框对应的类别标签，形状 [num_priors]，背景标记为 0
        """
        if type(gt_boxes) is np.ndarray:
            gt_boxes = torch.from_numpy(gt_boxes)
        if type(gt_labels) is np.ndarray:
            gt_labels = torch.from_numpy(gt_labels)

        # 维度层面：将真实边框/标签与先验框进行匹配，分配每个先验框对应的目标（计算交并比-角点模式）
        # 输入：原始真实框、真实标签、角点格式先验框、IOU 匹配阈值
        # 输出：boxes（每个先验框匹配后的真实框，角点格式，[num_priors, 4]）、labels（每个先验框的类别，[num_priors]）
        boxes, labels = box_utils.assign_priors(gt_boxes, gt_labels,
                                                self.corner_form_priors, self.iou_threshold)

        # 格式层面：将匹配后的角点格式边框，转为中心格式（适配后续偏移量计算）
        # 转换后格式：(cx, cy, w, h)，形状仍为 [num_priors, 4]
        boxes = box_utils.corner_form_to_center_form(boxes)

        # 数值层面：计算真实框相对于先验框的位置偏移量（模型位置回归的训练目标）
        # 输入：匹配后的中心格式框、原始中心格式先验框、中心/尺寸方差
        # 输出：归一化后的位置偏移量，形状 [num_priors, 4]
        locations = box_utils.convert_boxes_to_locations(boxes, self.center_form_priors, self.center_variance, self.size_variance)
       
        return locations, labels


