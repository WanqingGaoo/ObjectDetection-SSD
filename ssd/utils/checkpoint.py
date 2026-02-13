import logging
import os

import torch
from torch.nn.parallel import DistributedDataParallel

from ssd.utils.model_zoo import cache_url


class CheckPointer:
    '''
    保存模型训练过程中的检查点（包括模型权重、优化器状态、调度器状态等），
    以及后续加载这些检查点恢复训练或用于推理，避免训练中断（比如断电、宕机）后需要从头再来。
    '''
    _last_checkpoint_name = 'last_checkpoint.txt'

    def __init__(self,
                 model,               # 待保存/加载的模型实例（单GPU/DDP封装的分布式模型）
                 optimizer=None,      # 优化器实例（如SGD/Adam），可选，保存/加载其学习率、动量等状态
                 scheduler=None,      # 学习率调度器实例（如StepLR），可选，保存/加载其衰减状态
                 save_dir="",         # 检查点文件的保存目录，为空则不执行保存操作
                 save_to_disk=None,   # 是否将检查点保存到磁盘，分布式训练中仅主进程设为True
                 logger=None):        # 日志实例，用于打印保存/加载日志，可选

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.save_to_disk = save_to_disk
        if logger is None:
            logger = logging.getLogger(__name__)  # 初始化日志器：若未传入则创建当前类名的日志器，否则使用传入的日志器
        self.logger = logger

    def save(self, name, **kwargs):
        ''' 把训练的当前进度打包成一个 .pth 文件，并存好档，同时标记这是最新存档 '''
        # 若未指定保存目录，直接返回，不执行保存
        if not self.save_dir:
            return
        # 若磁盘保存开关为False（如分布式训练的非主进程），直接返回，避免重复保存
        if not self.save_to_disk:
            return

        data = {}
        # 兼容DDP分布式模型：DDP封装的模型需通过.module获取原始模型，再取状态字典
        # 单GPU模型直接取state_dict()即可
        if isinstance(self.model, DistributedDataParallel):
            data['model'] = self.model.module.state_dict()
        else:
            data['model'] = self.model.state_dict()
        if self.optimizer is not None:
            data["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            data["scheduler"] = self.scheduler.state_dict()
        # 合并自定义训练参数（如当前迭代数、epoch、损失值等）到检查点数据
        data.update(kwargs)

        save_file = os.path.join(self.save_dir, "{}.pth".format(name))
        self.logger.info("Saving checkpoint to {}".format(save_file))
        torch.save(data, save_file)

        # 标记当前保存的文件为最新检查点：将文件路径写入last_checkpoint.txt
        self.tag_last_checkpoint(save_file)

    def load(self, f=None, use_latest=True):
        if self.has_checkpoint() and use_latest:
            # override argument with existing checkpoint
            f = self.get_checkpoint_file()
        if not f:
            # no checkpoint could be found
            self.logger.info("No checkpoint found.")
            return {}

        self.logger.info("Loading checkpoint from {}".format(f))
        checkpoint = self._load_file(f)
        model = self.model
        if isinstance(model, DistributedDataParallel):
            model = self.model.module

        model.load_state_dict(checkpoint.pop("model"))
        if "optimizer" in checkpoint and self.optimizer:
            self.logger.info("Loading optimizer from {}".format(f))
            self.optimizer.load_state_dict(checkpoint.pop("optimizer"))
        if "scheduler" in checkpoint and self.scheduler:
            self.logger.info("Loading scheduler from {}".format(f))
            self.scheduler.load_state_dict(checkpoint.pop("scheduler"))

        # return any further checkpoint data
        return checkpoint

    def get_checkpoint_file(self):
        save_file = os.path.join(self.save_dir, self._last_checkpoint_name)
        try:
            with open(save_file, "r") as f:
                last_saved = f.read()
                last_saved = last_saved.strip()
        except IOError:
            # if file doesn't exist, maybe because it has just been
            # deleted by a separate process
            last_saved = ""
        return last_saved

    def has_checkpoint(self):
        save_file = os.path.join(self.save_dir, self._last_checkpoint_name)
        return os.path.exists(save_file)

    def tag_last_checkpoint(self, last_filename):
        ''' ：把最新保存的检查点文件路径，写入 last_checkpoint.txt 文件中，做「最新检查点」标记 '''
        save_file = os.path.join(self.save_dir, self._last_checkpoint_name)
        with open(save_file, "w") as f:
            f.write(last_filename)

    def _load_file(self, f):
        # download url files
        if f.startswith("http"):
            # if the file is a url path, download it and cache it
            cached_f = cache_url(f)
            self.logger.info("url {} cached in {}".format(f, cached_f))
            f = cached_f
        return torch.load(f, map_location=torch.device("cpu"))