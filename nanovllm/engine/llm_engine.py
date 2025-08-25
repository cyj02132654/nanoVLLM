import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


# 核心功能:
# - 进程管理: 启动多个ModelRunner进程实现张量并行
# - 请求管理: 统一的生成接口，支持批量处理
# - 性能监控: 实时显示prefill和decode吞吐量

# 生成流程:
# 1. add_request(): 添加请求到调度器
# 2. step(): 循环执行推理步骤
# 3. generate(): 返回解码后的文本结果

# 监控指标:
# - Prefill吞吐量(tok/s)
# - Decode吞吐量(tok/s)
# - 进度条显示

class LLMEngine:
    """
    LLM推理引擎，统一管理模型推理、调度和多进程协调。
    
    核心特性：
    - 张量并行：支持多进程分布式推理，提高大模型处理能力
    - 请求管理：统一的接口管理批量文本生成请求
    - 调度优化：与Scheduler集成，实现高效的资源管理
    - 性能监控：实时显示prefill和decode阶段的吞吐量
    - 生命周期管理：自动处理进程启动和清理
    """

    def __init__(self, model, **kwargs):
        """
        初始化LLM推理引擎
        
        初始化流程：
        1. 解析和验证配置参数
        2. 初始化张量并行进程组
        3. 设置tokenizer和调度器
        4. 注册退出清理函数
        
        Args:
            model: 模型路径或名称
            **kwargs: 其他配置参数
        """
        # 从 Config 类中提取所有字段名，用于参数验证
        config_fields = {field.name for field in fields(Config)}
        # 筛选出属于Config类的参数，避免无关参数污染配置
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        
        # 初始化进程管理的数据结构
        self.ps = []      # 存储子进程的列表
        self.events = []  # 存储进程间同步事件的列表
        
        # 获取多进程上下文（"spawn"方式适用于跨平台，尤其Windows）
        ctx = mp.get_context("spawn")
        
        # 创建张量并行子进程（索引从1开始）
        # 每个子进程对应模型的一个分片（张量并行的核心）
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()  # 创建进程间事件（用于同步）
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()  # 启动子进程
            self.ps.append(process)
            self.events.append(event)
        
        # 初始化主进程的ModelRunner（索引0）
        self.model_runner = ModelRunner(config, 0, self.events)
        
        # 初始化tokenizer并更新配置
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id  # 设置结束token ID
        
        # 初始化调度器用于任务调度和资源管理
        self.scheduler = Scheduler(config)
        
        # 注册程序退出时的清理方法
        atexit.register(self.exit)

    def exit(self):
        """
        清理资源和退出所有进程
        
        清理流程：
        1. 通知主进程ModelRunner退出
        2. 释放主进程资源
        3. 等待所有子进程结束
        """
        self.model_runner.call("exit")  # 通知主进程ModelRunner退出
        del self.model_runner           # 释放主进程资源
        for p in self.ps:               # 等待所有子进程结束
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        """
        添加新的生成请求到调度器
        
        处理流程：
        1. 将文本编码为token ID序列（如果需要）
        2. 创建Sequence对象封装请求
        3. 将序列添加到调度器的等待队列
        
        Args:
            prompt: 输入提示词（文本或token ID序列）
            sampling_params: 采样参数配置
        """
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)  # 将文本编码为token ID
        seq = Sequence(prompt, sampling_params)     # 创建Sequence对象
        self.scheduler.add(seq)                     # 添加到调度器

    def step(self):
        """
        执行一次推理步骤
        
        执行流程：
        1. 调度器选择待处理的序列
        2. ModelRunner执行推理计算
        3. 后处理结果并更新序列状态
        4. 返回完成的序列和性能统计
        
        Returns:
            tuple: (完成的序列列表, token数量统计)
                - outputs: [(seq_id, completion_token_ids), ...]
                - num_tokens: prefill时为正数，decode时为负数
        """
        # 1. 调度器选择待处理的序列（prefill或decode阶段）
        seqs, is_prefill = self.scheduler.schedule()
        
        # 2. 调用ModelRunner执行推理计算
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        
        # 3. 后处理：更新序列状态和检查完成条件
        self.scheduler.postprocess(seqs, token_ids)
        
        # 4. 收集完成的序列输出
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        
        # 5. 计算性能统计：prefill时为正数，decode时为负数
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        
        return outputs, num_tokens

    def is_finished(self):
        """
        检查所有请求是否已处理完成
        
        Returns:
            bool: 如果所有请求都已完成返回True，否则返回False
        """
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        """
        批量文本生成的主入口函数
        
        生成流程：
        1. 添加所有请求到调度器
        2. 循环执行推理步骤直到全部完成
        3. 实时监控和显示性能指标
        4. 返回解码后的文本结果
        
        Args:
            prompts: 输入提示词列表（文本或token ID序列）
            sampling_params: 采样参数（单个或列表）
            use_tqdm: 是否显示进度条和性能监控
            
        Returns:
            list[str]: 生成的文本结果列表，每个包含'text'和'token_ids'
        """
        # 初始化进度条（如果需要）
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        
        # 标准化采样参数为列表形式
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        
        # 添加所有请求到调度器
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        # 初始化输出存储和性能计数器
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        
        # 主推理循环：持续执行直到所有请求完成
        while not self.is_finished():
            t = perf_counter()  # 开始计时
            output, num_tokens = self.step()  # 执行一次推理步骤
            
            # 更新性能指标和进度条
            if use_tqdm:
                if num_tokens > 0:
                    # Prefill阶段：计算处理的token数/秒
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    # Decode阶段：计算处理的序列数/秒
                    decode_throughput = -num_tokens / (perf_counter() - t)
                # 更新进度条显示的性能信息
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            
            # 处理完成的序列输出
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids  # 存储完成的序列结果
                if use_tqdm:
                    pbar.update(1)  # 更新进度条
        # 按序列ID排序输出，保持原始顺序
        outputs = [outputs[seq_id] for seq_id in sorted(outputs)]
        
        # 将token ID解码为文本并构造返回结果
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} 
                  for token_ids in outputs]
        
        # 关闭进度条
        if use_tqdm:
            pbar.close()
        
        return outputs
