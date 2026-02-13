'''
这个函数会创建 / 获取一个指定名称的 logger，为其设置日志级别、输出格式，
并且只让主进程（distributed_rank=0）输出日志到控制台和文件，其他子进程不做额外配置，
避免日志重复打印
'''

import logging
import os
import sys


def setup_logger(name, distributed_rank, save_dir=None):
    # 1. 获取/创建指定名称的logger（如果不存在则新建）
    logger = logging.getLogger(name)

    # 2. 设置logger的基础日志级别为DEBUG（会捕获所有级别≥DEBUG的日志）
    logger.setLevel(logging.DEBUG)

    # 3. 分布式训练中，非主进程（rank>0）直接返回logger，不添加任何处理器
    #    目的：避免多个进程同时打印日志，造成日志混乱/重复
    if distributed_rank > 0:
        return logger

    # 4. 创建控制台输出处理器（StreamHandler），日志会打印到stdout（终端）
    stream_handler = logging.StreamHandler(stream=sys.stdout)

    # 5. 设置控制台处理器的日志级别为DEBUG（只输出≥DEBUG的日志）
    stream_handler.setLevel(logging.DEBUG)

    # 6. 定义日志格式：时间 + logger名称 + 日志级别 + 日志内容
    #    格式示例：2026-02-06 10:00:00 my_logger INFO: Training started
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")

    # 7. 给控制台处理器绑定格式
    stream_handler.setFormatter(formatter)

    # 8. 将控制台处理器添加到logger中，logger会通过这个处理器输出日志到终端
    logger.addHandler(stream_handler)

    # 9. 如果传入了保存目录，创建文件处理器（FileHandler），将日志写入文件
    if save_dir:
        # 拼接日志文件路径：save_dir/log.txt
        fh = logging.FileHandler(os.path.join(save_dir, 'log.txt'))
        # 设置文件处理器的日志级别
        fh.setLevel(logging.DEBUG)
        # 绑定相同的日志格式
        fh.setFormatter(formatter)
        # 将文件处理器添加到logger中，日志会同时写入文件
        logger.addHandler(fh)

    # 10. 返回配置好的logger
    return logger


def main():
    """
    主函数：测试setup_logger的不同使用场景
    """
    # ========== 场景1：模拟主进程（distributed_rank=0） ==========
    print("===== 主进程日志输出 =====")
    # 创建主进程logger，指定保存目录为./logs
    main_logger = setup_logger("train_main", distributed_rank=0, save_dir="./logs")
    # 输出不同级别的日志
    main_logger.debug("调试信息：初始化模型，学习率设置为0.001")
    main_logger.info("普通信息：训练开始，共10个epoch")
    main_logger.warning("警告信息：验证集准确率下降了3%")
    main_logger.error("错误信息：第5个epoch数据加载失败，已重试")
    main_logger.critical("严重错误：GPU显存不足，无法继续训练")

    # ========== 场景2：模拟子进程（distributed_rank=1） ==========
    print("\n===== 子进程日志输出（无任何输出） =====")
    worker_logger = setup_logger("train_worker", distributed_rank=1)
    # 子进程的logger没有添加处理器，所以这些日志不会输出到控制台/文件
    worker_logger.info("子进程1：开始处理第2批次数据")
    worker_logger.error("子进程1：数据处理出错")

    # ========== 场景3：不保存日志到文件（仅控制台输出） ==========
    print("\n===== 仅控制台输出日志 =====")
    no_file_logger = setup_logger("test_no_file", distributed_rank=0)
    no_file_logger.info("这个日志只会输出到控制台，不会保存到文件")

if __name__ == "__main__":
    # 运行主函数
    main()