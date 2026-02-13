from torch.utils.data.sampler import BatchSampler


class IterationBasedBatchSampler(BatchSampler):
    """
    基于迭代次数的批次采样器：对一个普通 BatchSampler 进行包装，核心作用是**按「迭代次数」停止采样，而非按数据集样本数/批次总数**。

    普通 BatchSampler 会遍历完数据集的所有批次后停止（以「数据量」为终止条件），
    而该采样器会在达到指定的迭代次数后停止（以「迭代步数」为终止条件），
    适合那些需要固定训练步数（而非固定 epoch 数）的训练场景（如一些检测、分割模型训练）。

    额外特性：
        若底层包装的采样器（如 DistributedSampler）拥有 `set_epoch()` 方法，
        会在每个迭代步骤更新 epoch，保证分布式训练的数据打乱一致性。
    """

    def __init__(self, batch_sampler, num_iterations, start_iter=0):
        self.batch_sampler = batch_sampler
        self.num_iterations = num_iterations
        self.start_iter = start_iter

    def __iter__(self):
        iteration = self.start_iter
        while iteration <= self.num_iterations:
            # if the underlying sampler has a set_epoch method, like
            # DistributedSampler, used for making each process see
            # a different split of the dataset, then set it
            if hasattr(self.batch_sampler.sampler, "set_epoch"):
                self.batch_sampler.sampler.set_epoch(iteration)
            for batch in self.batch_sampler:
                iteration += 1
                if iteration > self.num_iterations:
                    break
                yield batch

    def __len__(self):
        return self.num_iterations
