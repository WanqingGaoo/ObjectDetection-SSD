'''
轻量级的数据容器，用于统一管理目标检测中的 boxes、labels 等相关数据，同时提供便捷的批量操作方法
不继承自 Python 内置 dict 类——因为 PyTorch 的 `default_collate` 函数在处理数据批次时，
会将 dict 的子类强制转换为普通 dict，丢失自定义的方法（如 to、numpy、resize 等），因此采用「内部封装 dict」的方式实现。
'''


class Container:
    def __init__(self, *args, **kwargs):
        """
        初始化容器：内部创建一个普通 dict 来存储所有数据（键值对形式）。
        Args:
            *args: 可变位置参数，支持传入与 dict 初始化兼容的参数（如另一个 dict、键值对元组列表等）
            **kwargs: 可变关键字参数，支持直接传入键值对（如 boxes=xxx, labels=xxx）
        """
        self._data_dict = dict(*args, **kwargs)

    def __setattr__(self, key, value):
        """
        重写属性赋值方法：保证「实例属性赋值」和「内部 _data_dict 赋值」互不干扰。
        （默认情况下，self.key = value 会调用此方法，这里直接调用 object 的原生实现，不修改 _data_dict）
        Args:
            key: 属性名
            value: 属性值
        """
        # 调用 Python 原生 object 类的 __setattr__，实现普通的实例属性赋值（如 self.img_width = 640）
        object.__setattr__(self, key, value)

    def __getitem__(self, key):
        """
        实现「下标取值」功能（如 container['boxes']），直接从内部 _data_dict 中取值。
        Args:
            key: 要获取的数据对应的键（如 'boxes'、'labels'）
        Returns:
            该键对应的值（如 boxes 对应的张量/数组）
        """
        return self._data_dict[key]

    def __iter__(self):
        """
        实现迭代功能（如 for key in container），直接迭代内部 _data_dict 的键。
        Returns:
            _data_dict 的迭代器
        """
        return self._data_dict.__iter__()

    def __setitem__(self, key, value):
        """
        实现「下标赋值」功能（如 container['scores'] = xxx），直接给内部 _data_dict 赋值。
        Args:
            key: 要赋值的数据对应的键
            value: 要赋值的数据值
        """
        self._data_dict[key] = value

    def _call(self, name, *args, **kwargs):
        """
        内部核心辅助方法：批量对 _data_dict 中所有值，调用指定名称的方法（如 to、numpy）。
        逻辑：遍历 _data_dict 中的所有值，若该值拥有指定名称的方法，则调用该方法并更新值，最终返回自身（支持链式调用）。
        Args:
            name (str): 要调用的方法名称（如 'to'、'numpy'）
            *args: 传递给目标方法的位置参数
            **kwargs: 传递给目标方法的关键字参数
        Returns:
            self: 容器自身，支持链式调用（如 container.to('cpu').numpy()）
        """
        # 获取内部 _data_dict 的所有键，转为列表用于遍历
        keys = list(self._data_dict.keys())
        for key in keys:
            # 获取当前键对应的值
            value = self._data_dict[key]
            # 判断该值是否拥有指定名称的方法（如判断是否有 'to' 方法）
            if hasattr(value, name):
                # 反射获取该方法，并调用（传入参数），更新 _data_dict 中的值
                self._data_dict[key] = getattr(value, name)(*args, **kwargs)
        # 返回自身，支持链式调用
        return self

    def to(self, *args, **kwargs):
        """
        批量调用内部所有数据的 `to()` 方法（主要用于 PyTorch Tensor 设备迁移，如 cpu → cuda）。
        依赖内部 _call 方法实现，支持链式调用。
        Args:
            *args: 传递给 Tensor.to() 的参数（如 torch.device('cpu')、'cuda'）
            **kwargs: 传递给 Tensor.to() 的关键字参数
        Returns:
            self: 容器自身
        """
        return self._call('to', *args, **kwargs)

    def numpy(self):
        """
        批量调用内部所有数据的 `numpy()` 方法（主要用于将 PyTorch Tensor 转换为 NumPy 数组）。
        依赖内部 _call 方法实现，支持链式调用。
        Returns:
            self: 容器自身
        """
        return self._call('numpy')

    def resize(self, size):
        """
        专属方法：将容器中的 `boxes` 边框坐标，从原始图片尺寸缩放到目标尺寸（反归一化/尺寸适配）。
        仅处理 `boxes` 键对应的数据，其他数据不做修改。
        Args:
            size (tuple): 目标图片尺寸，格式为 (new_width, new_height)（宽度在前，高度在后）
        Returns:
            self: 容器自身，支持链式调用
        """
        # 获取容器的实例属性 img_width、img_height（原始图片的宽和高，需提前赋值）
        # 若未赋值，默认值为 -1
        img_width = getattr(self, 'img_width', -1)
        img_height = getattr(self, 'img_height', -1)

        # 断言检查：原始图片宽高必须大于 0（确保已提前赋值，避免缩放错误）
        assert img_width > 0 and img_height > 0
        # 断言检查：容器中必须包含 'boxes' 键（确保有需要缩放的边框数据）
        assert 'boxes' in self._data_dict

        # 从容器中获取边框数据（格式：[N, 4]，坐标为 (xmin, ymin, xmax, ymax)）
        boxes = self._data_dict['boxes']
        # 解析目标尺寸的宽和高
        new_width, new_height = size

        # 边框坐标缩放计算（逐列缩放，保证 x 坐标对应宽度，y 坐标对应高度）
        # boxes[:, 0::2]：选取所有行的第 0、2 列（xmin、xmax，水平方向坐标），乘以 新宽/原宽 的缩放比例
        boxes[:, 0::2] *= (new_width / img_width)
        # boxes[:, 1::2]：选取所有行的第 1、3 列（ymin、ymax，垂直方向坐标），乘以 新高/原高 的缩放比例
        boxes[:, 1::2] *= (new_height / img_height)

        # 返回自身，支持链式调用
        return self

    def __repr__(self):
        """
        重写容器的字符串表示形式，直接返回内部 _data_dict 的字符串表示（方便打印调试）。
        Returns:
            str: _data_dict 的字符串描述
        """
        return self._data_dict.__repr__()





if __name__ == '__main__':
    import numpy as np
    import torch

    print("=" * 80)
    print("开始测试 Container 类")
    print("=" * 80)

    # ---------------------- 步骤1：初始化 Container，存储测试数据 ----------------------
    # boxes：[N, 4]，格式 (xmin, ymin, xmax, ymax)，原始坐标为归一化值（0~1 之间）
    boxes_tensor = torch.tensor([[0.05, 0.1, 0.2, 0.35], [0.3, 0.25, 0.55, 0.32]], dtype=torch.float32)
    # labels：[N,]，类别 ID
    labels_tensor = torch.tensor([12, 7], dtype=torch.int64)
    # scores：[N,]，置信度
    scores_tensor = torch.tensor([0.98, 0.85], dtype=torch.float32)

    # 初始化 Container，传入键值对数据
    det_container = Container(
        boxes=boxes_tensor,
        labels=labels_tensor,
        scores=scores_tensor
    )

    # 给 Container 赋值原始图片尺寸（用于后续 resize）
    det_container.img_width = 1.0  # 原始归一化宽度（0~1）
    det_container.img_height = 1.0  # 原始归一化高度（0~1）

    print("初始化后的 Container：")
    print(det_container)
    print(f"boxes 数据类型：{type(det_container['boxes'])}")
    print(f"labels 数据类型：{type(det_container['labels'])}")

    # ---------------------- 步骤2：基础操作测试（下标、迭代） ----------------------
    print("\n" + "-" * 60)
    print("【步骤2】基础操作测试（下标取值/赋值、迭代）")

    # 下标取值
    print(f"\n通过下标获取 boxes：")
    print(det_container['boxes'])

    # 下标赋值（新增一个键值对）
    det_container['image_name'] = "test.jpg"
    print(f"\n新增 'image_name' 后的 Container：")
    print(det_container)

    # 迭代遍历所有键
    print(f"\nContainer 中的所有键：")
    for key in det_container:
        print(f"  - {key}")

    # ---------------------- 步骤3：批量操作测试（to()、numpy()） ----------------------
    print("\n" + "-" * 60)
    print("【步骤3】批量操作测试（to() 设备迁移、numpy() 格式转换）")

    # 测试 to() 方法（若有 GPU 可改为 'cuda'，这里默认用 'cpu' 兼容所有环境）
    det_container.to(torch.device('cpu'))
    print(f"to('cpu') 后，boxes 设备：{det_container['boxes'].device}（兼容无 GPU 环境）")

    # 测试 numpy() 方法（批量将 Tensor 转为 NumPy 数组）
    det_container.numpy()
    print(f"\nnumpy() 转换后：")
    print(f"boxes 数据类型：{type(det_container['boxes'])}")
    print(f"labels 数据类型：{type(det_container['labels'])}")
    print(f"boxes 数值（NumPy 数组）：")
    print(det_container['boxes'])

    # ---------------------- 步骤4：专属操作测试（resize() 边框缩放） ----------------------
    print("\n" + "-" * 60)
    print("【步骤4】专属操作测试（resize() 边框缩放）")

    # 目标尺寸：640x480（宽x高），将归一化坐标（0~1）转换为像素坐标
    target_size = (640, 480)
    det_container.resize(target_size)

    print(f"resize 到 {target_size} 后的 boxes（像素坐标）：")
    print(det_container['boxes'])
    print(f"resize 后的 labels（无变化）：")
    print(det_container['labels'])

    # ---------------------- 步骤5：链式调用测试 ----------------------
    print("\n" + "-" * 60)
    print("【步骤5】链式调用测试（新建 Container 演示）")

    # 新建一个 Tensor 格式的 Container，演示链式调用
    new_container = Container(
        boxes=torch.tensor([[0.1, 0.1, 0.2, 0.2]], dtype=torch.float32),
        labels=torch.tensor([1], dtype=torch.int64)
    )
    new_container.img_width = 1.0
    new_container.img_height = 1.0

    # 链式调用：to(cpu) → numpy() → resize((320, 240))
    result_container = new_container.to('cpu').numpy().resize((320, 240))

    print(f"链式调用后的结果：")
    print(f"boxes 数据类型：{type(result_container['boxes'])}")
    print(f"boxes 数值（320x240 像素坐标）：")
    print(result_container['boxes'])