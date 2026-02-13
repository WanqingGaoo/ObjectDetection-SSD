import torch.nn as nn
import torch.nn.functional as F
import torch

from ssd.utils import box_utils

#
# class MultiBoxLoss(nn.Module):
#     def __init__(self, neg_pos_ratio):
#         """Implement SSD MultiBox Loss.
#
#         Basically, MultiBox loss combines classification loss
#          and Smooth L1 regression loss.
#         """
#         super(MultiBoxLoss, self).__init__()
#         self.neg_pos_ratio = neg_pos_ratio
#
#     def forward(self, confidence, predicted_locations, labels, gt_locations):
#         """Compute classification loss and smooth l1 loss.
#
#         Args:
#             confidence (batch_size, num_priors, num_classes): class predictions.
#             predicted_locations (batch_size, num_priors, 4): predicted locations.
#             labels (batch_size, num_priors): real labels of all the priors.
#             gt_locations (batch_size, num_priors, 4): real boxes corresponding all the priors.
#         """
#         num_classes = confidence.size(2)
#         with torch.no_grad():
#             # derived from cross_entropy=sum(log(p))
#             loss = -F.log_softmax(confidence, dim=2)[:, :, 0]
#             mask = box_utils.hard_negative_mining(loss, labels, self.neg_pos_ratio)
#
#         confidence = confidence[mask, :]
#         classification_loss = F.cross_entropy(confidence.view(-1, num_classes), labels[mask], reduction='sum')
#
#         pos_mask = labels > 0
#         predicted_locations = predicted_locations[pos_mask, :].view(-1, 4)
#         gt_locations = gt_locations[pos_mask, :].view(-1, 4)
#         smooth_l1_loss = F.smooth_l1_loss(predicted_locations, gt_locations, reduction='sum')
#         num_pos = gt_locations.size(0)
#         return smooth_l1_loss / num_pos, classification_loss / num_pos

class MultiBoxLoss(nn.Module):
    """
    SSD 模型的 MultiBox 损失类：整合了「类别分类损失」和「边框回归 Smooth L1 损失」。
    核心特点：
    1.  分类损失：采用交叉熵损失，针对先验框的类别预测（背景/各类目标）。
    2.  回归损失：采用 Smooth L1 损失，仅针对正样本先验框（匹配到真实框的先验框）。
    3.  负样本处理：采用「硬负样本挖掘（Hard Negative Mining）」，平衡正负样本比例（解决正负样本失衡问题）。
    """

    def __init__(self, neg_pos_ratio):
        """
        初始化 MultiBox 损失函数
        Args:
            neg_pos_ratio (int/float): 负样本与正样本的比例（通常设为3），用于硬负样本挖掘。
            例如：neg_pos_ratio=3 表示每1个正样本，保留3个最难的负样本参与分类损失计算。
        """
        # 调用父类 nn.Module 的初始化方法，这是自定义PyTorch模块的必做步骤
        super(MultiBoxLoss, self).__init__()
        # 保存负正样本比例，后续硬负样本挖掘时使用
        self.neg_pos_ratio = neg_pos_ratio

    def forward(self, confidence, predicted_locations, labels, gt_locations):
        """
        前向传播：计算分类损失和边框回归损失（核心逻辑）。
        Args:
            confidence (torch.Tensor): 模型预测的类别置信度，形状 [batch_size, num_priors, num_classes]
                - batch_size：批次大小（一次训练的图片数量）
                - num_priors：先验框总数（如8732）
                - num_classes：类别总数（包含背景，如VOC数据集为21类：背景+20类目标）
            predicted_locations (torch.Tensor): 模型预测的边框偏移量，形状 [batch_size, num_priors, 4]
                - 4：对应 cx/cy/w/h 四个维度的偏移量
            labels (torch.Tensor): 每个先验框的真实类别标签，形状 [batch_size, num_priors]
                - 数值含义：0=背景，>0=对应目标类别ID（如1=飞机，2=自行车）
            gt_locations (torch.Tensor): 每个先验框对应的真实边框偏移量（正样本有效），形状 [batch_size, num_priors, 4]
        Returns:
            smooth_l1_loss (torch.Tensor): 归一化后的边框回归损失（仅正样本参与）
            classification_loss (torch.Tensor): 归一化后的类别分类损失（正负样本按比例参与）
        """
        # 从置信度张量中获取类别总数（包含背景类）
        num_classes = confidence.size(2)

        # 上下文管理器：禁用梯度计算（此部分仅用于筛选负样本，无需反向传播，节省内存、提升速度）
        with torch.no_grad():
            # 步骤1：计算所有先验框的「背景类负样本损失」，用于后续硬负样本挖掘
            # F.log_softmax(confidence, dim=2)：对类别维度做log softmax，得到每个类别的对数概率
            # [:, :, 0]：提取背景类（第0类）的对数概率
            # 加负号：将对数概率转为「损失值」（损失越大，说明模型越认为这个先验框不是背景，即越可能是难分负样本）
            loss = -F.log_softmax(confidence, dim=2)[:, :, 0]

            # 步骤2：硬负样本挖掘（核心：平衡正负样本比例，只保留最难的负样本）
            # 输入：背景类损失值、真实标签、负正样本比例
            # 输出：一个布尔型掩码（mask），形状 [batch_size, num_priors]，值为True表示该先验框被选中参与分类损失计算
            # 逻辑：1. 筛选出所有正样本（labels>0），全部保留；2. 筛选出负样本中损失最大的部分，数量为正样本的neg_pos_ratio倍；3. 其余负样本丢弃
            mask = box_utils.hard_negative_mining(loss, labels, self.neg_pos_ratio)

        # 步骤3：计算类别分类损失（交叉熵损失）
        # 1. 用掩码筛选出参与计算的先验框的置信度，形状变为 [selected_priors, num_classes]
        confidence = confidence[mask, :]
        # 2. 计算交叉熵损失：
        #    - confidence.view(-1, num_classes)：将置信度展平为 [selected_priors_total, num_classes]，适配交叉熵输入格式
        #    - labels[mask]：筛选出参与计算的先验框的真实标签，展平为 [selected_priors_total]
        #    - reduction='sum'：损失结果求和（后续除以正样本数量归一化）
        classification_loss = F.cross_entropy(confidence.view(-1, num_classes), labels[mask], reduction='sum')

        # 步骤4：计算边框回归损失（Smooth L1 损失）
        # 1. 构建正样本掩码：形状 [batch_size, num_priors]，值为True表示该先验框是正样本（匹配到真实框）
        pos_mask = labels > 0
        # 2. 筛选出正样本的预测偏移量，展平为 [num_pos, 4]（num_pos为正样本总数，仅正样本参与回归损失计算）
        predicted_locations = predicted_locations[pos_mask, :].view(-1, 4)
        # 3. 筛选出正样本的真实偏移量，展平为 [num_pos, 4]
        gt_locations = gt_locations[pos_mask, :].view(-1, 4)
        # 4. 计算Smooth L1损失：对离群值鲁棒（避免大误差样本主导损失），结果求和
        smooth_l1_loss = F.smooth_l1_loss(predicted_locations, gt_locations, reduction='sum')

        # 步骤5：损失归一化（除以正样本数量，消除批次中正样本数量差异的影响）
        num_pos = gt_locations.size(0)  # 获取正样本总数

        # 返回归一化后的两个损失：边框回归损失、分类损失
        return smooth_l1_loss / num_pos, classification_loss / num_pos