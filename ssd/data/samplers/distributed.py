# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Code is copy-pasted exactly as in torch.utils.data.distributed.
# FIXME remove this once c10d fixes the bug it has
import math
import torch
import torch.distributed as dist
from torch.utils.data.sampler import Sampler


class DistributedSampler(Sampler):
    """
    分布式采样器：在 DDP（分布式数据并行）训练中，为每个进程（通常对应单个 GPU）分配**专属且不重叠**的数据集子集。
    核心作用：避免不同进程训练相同数据，保证训练效率和结果一致性，同时支持数据打乱（不同 epoch 打乱规则一致）。

    注意事项：
        1. 假设数据集大小是固定不变的，不支持动态增减样本的数据集。
        2. 需配合 torch.nn.parallel.DistributedDataParallel 使用，单卡训练无需使用。

    参数说明：
        dataset (torch.utils.data.Dataset): 待采样的原始数据集（必填）。
        num_replicas (int, 可选): 参与分布式训练的进程总数（通常等于 GPU 数量）。
            若不指定，会自动从分布式环境中获取（dist.get_world_size()）。
        rank (int, 可选): 当前进程在所有进程中的唯一编号（范围：0 ~ num_replicas-1）。
            若不指定，会自动从分布式环境中获取（dist.get_rank()）。
        shuffle (bool, 可选): 是否在每个 epoch 打乱数据顺序，默认 True（推荐训练时开启）。
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        # 1. 校验并获取分布式训练的进程总数（num_replicas）
        if num_replicas is None:
            # 先判断分布式包是否可用
            if not dist.is_available():
                raise RuntimeError("分布式训练包不可用，请确保已正确初始化分布式环境")
            # 自动获取分布式环境中的进程总数
            num_replicas = dist.get_world_size()

        # 2. 校验并获取当前进程的编号（rank）
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("分布式训练包不可用，请确保已正确初始化分布式环境")
            # 自动获取当前进程的唯一编号
            rank = dist.get_rank()

        # 3. 初始化实例属性
        self.dataset = dataset  # 原始数据集
        self.num_replicas = num_replicas  # 进程总数（GPU 数）
        self.rank = rank  # 当前进程编号
        self.epoch = 0  # 当前训练轮次，用于控制数据打乱的随机性（保证不同进程同一 epoch 打乱规则一致）
        self.shuffle = shuffle  # 是否开启数据打乱

        # 4. 计算当前进程应加载的样本数（向上取整，保证每个进程样本数一致）
        # math.ceil: 向上取整，避免数据集大小无法被进程总数整除时，部分进程缺少样本
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))

        # 5. 计算补全后的总样本数（保证能被进程总数整除）
        # 若原始数据集大小无法被 num_replicas 整除，会补充部分样本（重复前面的样本）
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        """
        核心迭代方法：生成当前进程专属的样本索引列表（迭代器）。
        执行流程：1. 生成全量样本索引（可选打乱）；2. 补全样本至 total_size；3. 截取当前进程专属子集。
        """
        if self.shuffle:
            # 若开启打乱：基于当前 epoch 生成固定随机种子，保证不同进程同一 epoch 打乱结果一致
            # 原因：不同进程使用相同 epoch 作为种子，生成的随机排列完全相同，后续截取子集才不会重叠
            g = torch.Generator()  # PyTorch 随机数生成器
            g.manual_seed(self.epoch)  # 设置随机种子（绑定 epoch，每个 epoch 种子不同，实现不同 epoch 打乱不同）
            # 生成原始数据集的随机排列索引，并转换为 Python 列表
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            # 若不开启打乱：生成有序的连续索引列表
            indices = torch.arange(len(self.dataset)).tolist()

        # 步骤 2：补全样本索引，使总长度等于 total_size（解决数据集大小无法被进程总数整除的问题）
        # 补充逻辑：重复截取原始索引列表的前 N 个元素（N = total_size - 原始长度）
        indices += indices[: (self.total_size - len(indices))]
        # 断言校验：补全后的索引长度必须等于 total_size（防止逻辑错误）
        assert len(indices) == self.total_size

        # 步骤 3：截取当前进程专属的样本索引子集（不重叠、专属）
        # 计算当前进程的索引偏移量：每个进程截取的起始位置 = 进程编号 * 每个进程的样本数
        offset = self.num_samples * self.rank
        # 截取：从偏移量开始，截取 num_samples 个元素
        indices = indices[offset: offset + self.num_samples]
        # 断言校验：截取后的索引长度必须等于 num_samples（保证每个进程样本数一致）
        assert len(indices) == self.num_samples

        # 返回索引列表的迭代器，供 DataLoader 迭代获取样本
        return iter(indices)

    def __len__(self):
        """
        返回当前进程加载的样本数（DataLoader 会通过该方法获取批次迭代次数）。
        """
        return self.num_samples

    def set_epoch(self, epoch):
        """
        设置当前训练轮次，用于控制打乱时的随机种子。
        必须在每个 epoch 开始前调用（训练循环中），否则所有 epoch 的数据顺序一致，无法实现有效打乱。

        参数：
            epoch (int): 当前训练轮次（从 0 开始递增）。
        """
        self.epoch = epoch