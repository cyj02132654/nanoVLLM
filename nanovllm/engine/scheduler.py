from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager

# 核心功能:
# - 双阶段调度策略:
# - Prefill阶段: 处理新请求的初始prompt
# - Decode阶段: 处理正在运行序列的token生成
# - 资源限制管理:
# - max_num_seqs: 最大并发序列数
# - max_num_batched_tokens: 最大批处理token数
# - 抢占机制: 显存不足时可抢占运行中的序列

# 调度流程:
# 1. 优先处理waiting队列中的新请求(prefill)
# 2. 无新请求时处理running队列(decode)
# 3. 显存不足时触发抢占机制

class Scheduler:
    """
    序列调度器，负责管理推理请求的调度和资源分配。
    
    核心特性：
    - 双阶段调度：Prefill（处理新请求）+ Decode（处理运行中请求）
    - 资源限制管理：控制并发序列数和批处理token数
    - 抢占机制：显存不足时自动抢占运行中的序列
    - KV缓存管理：与BlockManager集成管理显存分配
    """

    def __init__(self, config: Config):
        """
        初始化调度器
        
        Args:
            config: 包含调度参数的配置对象
        """
        self.max_num_seqs = config.max_num_seqs                    # 最大并发序列数
        self.max_num_batched_tokens = config.max_num_batched_tokens # 最大批处理token数
        self.eos = config.eos                                      # 结束token ID
        # 初始化块管理器，用于KV缓存的显存管理
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()                    # 等待处理的序列队列
        self.running: deque[Sequence] = deque()                    # 正在运行的序列队列

    def is_finished(self):
        """
        检查调度器是否已完成所有任务
        
        Returns:
            如果没有等待和运行中的序列返回True，否则返回False
        """
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        """
        添加新序列到等待队列
        
        Args:
            seq: 要添加的序列对象
        """
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        """
        调度序列进行推理，采用双阶段策略
        
        调度策略：
        1. Prefill阶段：优先处理等待队列中的新请求
        2. Decode阶段：处理运行队列中序列的token生成
        3. 资源检查：确保不超过最大序列数和token数限制
        4. 抢占机制：显存不足时抢占运行中的序列
        
        Returns:
            tuple[list[Sequence], bool]: (调度的序列列表, 是否为prefill阶段)
        """
        # Prefill阶段：处理新请求
        scheduled_seqs = []     # 本次调度的序列列表
        num_seqs = 0           # 已调度序列数
        num_batched_tokens = 0  # 已调度token数（不包括缓存命中部分）
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]  # 获取等待队列头部序列
            # 检查资源限制：token数量和显存分配能力
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break  # 资源不足，无法调度更多序列
            num_seqs += 1                                      # 增加序列计数
            self.block_manager.allocate(seq)                   # 为序列分配显存块
            num_batched_tokens += len(seq) - seq.num_cached_tokens  # 更新token计数（排除缓存命中部分）
            seq.status = SequenceStatus.RUNNING                # 更新序列状态
            self.waiting.popleft()                             # 从等待队列移除
            self.running.append(seq)                           # 加入运行队列
            scheduled_seqs.append(seq)                         # 加入本次调度列表
        if scheduled_seqs:
            return scheduled_seqs, True  # 返回prefill阶段的调度结果

        # Decode阶段：处理运行中序列的token生成
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()  # 从运行队列取出序列
            # 检查是否有足够显存追加新token
            while not self.block_manager.can_append(seq):
                if self.running:
                    # 抢占运行队列中的最后一个序列（LIFO策略）
                    self.preempt(self.running.pop())
                else:
                    # 无其他序列可抢占，抢占当前序列
                    self.preempt(seq)
                    break
            else:
                # 显存足够，可以处理该序列
                num_seqs += 1                        # 增加序列计数
                self.block_manager.may_append(seq)   # 为追加操作准备显存块
                scheduled_seqs.append(seq)           # 加入调度列表
        assert scheduled_seqs  # 确保decode阶段至少调度了一个序列
        # 将调度的序列重新放回运行队列头部，保持原有顺序
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False  # 返回decode阶段的调度结果

    def preempt(self, seq: Sequence):
        """
        抢占序列，释放其占用的显存资源
        
        抢占机制用于在显存不足时回收资源：
        1. 将序列状态改为等待
        2. 释放序列占用的显存块
        3. 将序列重新放入等待队列头部（优先处理）
        
        Args:
            seq: 要抢占的序列
        """
        seq.status = SequenceStatus.WAITING      # 更新状态为等待
        self.block_manager.deallocate(seq)       # 释放显存块
        self.waiting.appendleft(seq)             # 重新放入等待队列头部

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        """
        后处理推理结果，更新序列状态并检查完成条件
        
        处理流程：
        1. 将生成的token添加到序列中
        2. 检查序列是否达到结束条件（EOS token或达到最大长度）
        3. 完成的序列释放显存并从运行队列移除
        
        Args:
            seqs: 参与推理的序列列表
            token_ids: 对应生成的token ID列表
            
        Returns:
            list[bool]: 每个序列是否已完成的标志列表
        """
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)  # 将新生成的token添加到序列
            # 检查完成条件：遇到EOS token或达到最大token数
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED  # 标记为已完成
                self.block_manager.deallocate(seq)     # 释放占用的显存块
                self.running.remove(seq)               # 从运行队列移除
