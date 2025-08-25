import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model
# 核心功能:
#   - 多卡张量并行: 通过NCCL实现分布式推理
#   - 内存优化:
#     - 动态KV缓存分配基于GPU内存使用率
#     - CUDA Graph加速decode阶段推理
#   - 两阶段推理优化:
#     - Prefill: 处理变长序列，支持prefix缓存
#     - Decode: 单token生成，高度优化
#   - 进程间通信: 通过共享内存实现多进程协调

#   性能优化:
#   - warmup_model(): 模型预热
#   - capture_cudagraph(): 捕获CUDA计算图
#   - 支持eager模式和graph模式切换

class ModelRunner:
    """
    模型运行器 - 分布式LLM推理的核心组件
    
    主要职责：
    - 张量并行管理：在多卡间协调模型推理计算
    - 显存优化：智能分配KV缓存，提高GPU显存利用率
    - 推理加速：双阶段优化(Prefill/Decode)和CUDA Graph加速
    - 进程通信：通过共享内存实现多进程协调
    - 采样管理：控制文本生成的随机性和多样性
    
    技术亮点：
    - 支持Flash Attention变长序列优化
    - CUDA Graph捕获降低单token生成延迟
    - 前缀缓存避免重复计算
    - 异步内存传输提高吞吐量
    """

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        """
        初始化模型运行器
        
        初始化流程：
        1. 分布式环境设置（NCCL通信组）
        2. CUDA设备和数据类型配置
        3. 模型加载和采样器初始化
        4. 性能优化（预热、KV缓存、CUDA Graph）
        5. 多进程通信设置
        
        Args:
            config: 包含模型和运行时参数的配置对象
            rank: 当前进程在分布式组中的编号（0为主进程）
            event: 进程间同步事件（主进程传入列表，工作进程传入单个）
        """
        # 基本配置参数
        self.config = config                              # 存储全局配置
        hf_config = config.hf_config                      # HuggingFace模型配置
        self.block_size = config.kvcache_block_size       # KV缓存块大小
        self.enforce_eager = config.enforce_eager         # 是否强制使用eager模式（禁用CUDA Graph）
        self.world_size = config.tensor_parallel_size     # 张量并行大小（总进程数）
        self.rank = rank                                  # 当前进程编号
        self.event = event                                # 进程间同步事件

        # 分布式环境初始化
        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)              # 设置当前进程使用的GPU设备
        
        # 保存原始设置并更改为模型所需的配置
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)  # 设置模型数据类型（如fp16/bf16）
        torch.set_default_device("cuda")               # 设置默认设备为GPU
        
        # 模型和采样器初始化
        self.model = Qwen3ForCausalLM(hf_config)         # 创建Qwen3模型
        load_model(self.model, config.model)             # 加载预训练权重
        self.sampler = Sampler()                         # 初始化采样器
        
        # 性能优化步骤
        self.warmup_model()        # 模型预热，测试显存使用
        self.allocate_kv_cache()   # 分配KV缓存空间
        if not self.enforce_eager:
            self.capture_cudagraph()  # 捕获CUDA计算图（非强制eager模式）
        
        # 恢复原始设置
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)
        # 多进程通信初始化
        if self.world_size > 1:
            if rank == 0:  # 主进程：创建共享内存
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)  # 1MB共享内存
                dist.barrier()  # 等待所有进程就绪
            else:  # 工作进程：连接共享内存并进入消息循环
                dist.barrier()  # 等待主进程创建共享内存
                self.shm = SharedMemory(name="nanovllm")  # 连接已存在的共享内存
                self.loop()  # 进入消息循环，等待主进程指令

    def exit(self):
        """
        清理资源和退出进程
        
        清理流程：
        1. 关闭共享内存连接
        2. 主进程负责删除共享内存对象
        3. 释放CUDA Graph相关资源
        4. 同步GPU操作并销毁进程组
        """
        if self.world_size > 1:
            self.shm.close()      # 关闭共享内存连接
            dist.barrier()        # 等待所有进程关闭完成
            if self.rank == 0:
                self.shm.unlink() # 主进程负责删除共享内存对象
        
        if not self.enforce_eager:
            del self.graphs, self.graph_pool  # 释放CUDA Graph资源
        
        torch.cuda.synchronize()        # 同步所有CUDA操作
        dist.destroy_process_group()     # 销毁分布式进程组

    def loop(self):
        """
        工作进程的消息循环 - 等待和处理主进程指令
        
        工作流程：
        1. 从共享内存读取主进程的方法调用指令
        2. 执行对应的方法（如run、warmup_model等）
        3. 继续等待下一条指令，直到接收exit指令
        
        设计原理：
        - 主进程负责任务调度和结果收集
        - 工作进程专注于计算执行，保持高效的GPU利用率
        - 通过共享内存通信避免跨进程数据复制开销
        """
        while True:
            method_name, args = self.read_shm()  # 读取主进程发送的指令
            self.call(method_name, *args)        # 执行对应的方法
            if method_name == "exit":            # 接收退出指令则终止循环
                break

    def read_shm(self):
        """
        从共享内存读取主进程发送的指令
        
        数据格式：[4字节长度] + [序列化数据]
        - 前4字节：小端序整数，表示后续数据长度
        - 后续数据：pickle序列化的[method_name, *args]
        
        Returns:
            tuple: (method_name, args) - 方法名和参数列表
        """
        assert self.world_size > 1 and self.rank  # 仅工作进程调用
        
        self.event.wait()  # 等待主进程发送数据信号
        
        # 读取数据长度（前4字节）
        n = int.from_bytes(self.shm.buf[0:4], "little")
        
        # 读取并反序列化实际数据
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        
        # 清除事件信号，通知主进程数据已读取
        self.event.clear()
        
        return method_name, args

    def write_shm(self, method_name, *args):
        """
        向共享内存写入指令并通知所有工作进程
        
        数据序列化流程：
        1. 使用pickle序列化[method_name, *args]
        2. 先写入数据长度（前4字节）
        3. 再写入序列化后的数据
        4. 触发所有工作进程的事件信号
        
        Args:
            method_name: 要调用的方法名
            *args: 方法参数
        """
        assert self.world_size > 1 and not self.rank  # 仅主进程调用
        
        # 序列化指令和参数
        data = pickle.dumps([method_name, *args])
        n = len(data)
        
        # 先写入数据长度（前4字节，小端序）
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        
        # 再写入序列化数据
        self.shm.buf[4:n+4] = data
        
        # 通知所有工作进程数据已准备好
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        """
        核心的方法调用转发逻辑 - 实现分布式协调执行
        
        工作流程：
        1. 主进程发起调用：
           - 通过write_shm将指令广播给所有工作进程
           - 自己也执行相同的方法调用
        2. 工作进程响应：
           - 在loop中接收到指令后直接执行
           - 不再转发，避免无限循环
        
        设计优势：
        - 同步执行：所有进程同时执行相同操作
        - 零延迟：不需要等待结果返回
        - 统一协调：保证所有进程状态一致
        
        Args:
            method_name: 要调用的方法名
            *args: 方法参数
        
        Returns:
            方法执行结果（仅主进程有效）
        """
        # 主进程负责向工作进程广播指令
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        
        # 所有进程都执行相同的方法调用
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        """
            功能：初始化模型并测试显存使用，为后续推理做准备
            主要操作：
            清空 CUDA 缓存（torch.cuda.empty_cache()）并重置内存统计
            根据配置计算最大批处理 token 数和序列数
            创建一批测试序列（用 0 填充的最大长度序列）
            调用run函数执行一次推理（is_prefill=True表示预填充阶段）
            再次清空缓存，确保预热不影响实际推理显存
        """
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        """
        功能：计算并分配 KV 缓存空间（LLM 推理核心优化点）
        核心逻辑：
        计算当前 GPU 的内存使用情况（已用、峰值、当前分配）
        计算单个 KV 缓存块的字节大小（基于模型层数、头数、维度等）
        根据 GPU 总内存和内存利用率配置，计算可分配的 KV 缓存块数量
        创建一个大张量作为全局 KV 缓存，并分配给模型各层的k_cache和v_cache
        意义：KV 缓存存储历史 token 的键值对，避免重复计算，是提升生成式模型推理效率的关键
        """
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * hf_config.head_dim * hf_config.torch_dtype.itemsize
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.zeros(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, hf_config.head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        """
        功能：统一管理序列在 KV 缓存中的存储位置
        操作：
        对所有序列的块表（记录序列在 KV 缓存中的块索引）进行长度对齐（用 - 1 填充短序列）
        转换为 CUDA 张量，用于快速访问 KV 缓存
        背景：长序列通常被分割成固定大小的块存储在 KV 缓存中，块表记录了序列对应的块索引
        """
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        """
        预填充阶段数据准备 - LLM推理优化的核心函数
        
        功能概述:
        将多个序列的输入数据转换为模型可直接使用的张量格式，同时支持前缀缓存优化
        
        核心优化思想:
        1. 批量处理: 将多个序列合并成单个batch，提高GPU利用率
        2. 增量计算: 只计算未缓存的tokens，避免重复计算      
        3. 内存对齐: 通过slot mapping精确控制KV缓存存储位置  
        4. 前缀共享: 检测并利用序列间的公共前缀
        5. 异步传输: 使用pinned memory实现CPU-GPU并发传输

        与Flash Attention的配合:
        - cu_seqlens_*: 提供变长序列的边界信息
        - max_seqlen_*: 用于FlashAttention内核的内存分配
        - 支持不同长度序列在单个batch中高效处理

        参数:
        - seqs: 需要处理的序列列表

        返回值:
        - input_ids: 模型前向传播需要的token IDs
        - positions: 对应的位置编码
        注: 其他信息通过set_context存储在全局上下文中，供attention层使用
        """
        # 1. 初始化数据结构
        input_ids = []           # 存储需要计算的token IDs    
        positions = []           # 存储每个token在序列中的位置
        cu_seqlens_q = [0]       # Query的累积序列长度数组
        cu_seqlens_k = [0]       # Key/Value的累积序列长度数组
        max_seqlen_q = 0         # Query的最大序列长度
        max_seqlen_k = 0         # Key/Value的最大序列长度
        slot_mapping = []        # KV缓存槽位映射
        block_tables = None      # 块表(前缀缓存时需要)
        
        # 2. 遍历每个序列进行数据准备
        for seq in seqs:
            seqlen = len(seq)                           # 序列总长度
            # 只收集未缓存的tokens
            input_ids.extend(seq[seq.num_cached_tokens:])  # 跳过已缓存的tokens
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))  # 对应的位置编码
            
            # 计算序列长度
            seqlen_q = seqlen - seq.num_cached_tokens   # 需要计算query的长度  
            seqlen_k = seqlen                           # key/value的长度(包含缓存部分)
            
            # 构建累积长度数组 (Flash Attention需要)
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            
            # 更新最大长度
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            
            # 3. 构建slot mapping (KV缓存位置映射)
            if not seq.block_table:  # 序列还没有分配块表则跳过
                continue
            # 遍历需要存储KV的块(跳过已缓存的块)
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size  # 块的起始位置
                if i != seq.num_blocks - 1:  # 非最后一个块
                    end = start + self.block_size  # 填满整个块
                else:  # 最后一个块
                    end = start + seq.last_block_num_tokens  # 只到实际token数量
                slot_mapping.extend(list(range(start, end)))  # 生成连续的槽位索引
        
        # 4. 前缀缓存检测与优化
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # 当K长度 > Q长度时，存在前缀缓存
            block_tables = self.prepare_block_tables(seqs)
        
        # 5. 转换为GPU张量 (使用pinned memory + non_blocking优化传输)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        
        # 6. 设置全局上下文 (供attention层使用)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        """
        解码阶段数据准备 - 单token生成的高效优化
        
        功能概述:
        处理decode阶段的数据准备，每个序列只处理一个新token，追求极致的生成速度
        
        核心特点:
        1. 单token处理: 每个序列只处理最后一个token
        2. 固定批量: 批量大小等于序列数量，便于CUDA Graph优化  
        3. KV缓存追加: 新的KV值追加到现有缓存的末尾
        4. 延迟优化: 专为低延迟单token生成设计
        
        与prefill的区别:
        - prefill: 变长序列，复杂的attention计算
        - decode: 定长批次，简单的append操作
        
        参数:
        - seqs: 需要生成下一个token的序列列表
        
        返回值:
        - input_ids: 每个序列的最后一个token (用于生成下一个token)
        - positions: 对应的位置编码
        """
        input_ids = []      # 存储每个序列的最后一个token
        positions = []      # 存储对应的位置编码
        slot_mapping = []   # 新token在KV缓存中的存储位置  
        context_lens = []   # 每个序列的上下文长度
        
        for seq in seqs:
            # 1. 输入数据: 只处理最后一个token
            input_ids.append(seq.last_token)  # 当前序列的最后一个token
            positions.append(len(seq))        # 位置编码 = 序列当前长度
            
            # 2. 上下文长度: 用于attention mask计算
            context_lens.append(len(seq))     # 当前序列的总长度
            
            # 3. KV缓存位置计算: 新token存储在最后一个块的末尾
            # 公式: 最后块起始位置 + 块内偏移 - 1
            last_block_id = seq.block_table[-1]                    # 最后一个块的ID
            block_start = last_block_id * self.block_size          # 块的起始地址
            offset = seq.last_block_num_tokens - 1                 # 块内偏移
            slot_mapping.append(block_start + offset)              # 新token的存储位置
        
        # 4. 转换为GPU张量 (使用pinned memory优化传输)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        
        # 5. 准备块表并设置上下文
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        """
        采样参数准备 - 控制生成文本的随机性和多样性
        
        功能概述:
        收集所有序列的温度参数，用于控制token采样时的随机性程度
        
        温度参数的作用:
        - temperature = 0.0: 贪心解码，总是选择概率最高的token (确定性输出)
        - temperature = 1.0: 标准采样，按照模型输出的概率分布采样
        - temperature < 1.0: 降低随机性，输出更保守、更确定 (如0.7)
        - temperature > 1.0: 增加随机性，输出更创造性、更多样 (如1.2)
        
        实现原理:
        温度参数用于调整logits: logits_scaled = logits / temperature
        - 低温度使概率分布更尖锐 (概率向最大值集中)
        - 高温度使概率分布更平坦 (各选项概率更均匀)
        
        参数:
        - seqs: 序列列表，每个序列包含其采样参数
        
        返回值:
        - temperatures: 温度参数张量，与序列批次对应
        """
        temperatures = []  # 存储所有序列的温度参数
        
        # 收集每个序列的温度设置
        for seq in seqs:
            temperatures.append(seq.temperature)
        
        # 转换为GPU张量 (float32精度足够，使用pinned memory优化传输)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        """
        模型推理执行 - 智能选择eager模式或CUDA Graph加速
        
        模式选择策略：
        1. Prefill阶段：始终使用eager模式（支持变长输入）
        2. 强制eager模式：配置要求禁用CUDA Graph
        3. 大批量（>512）：使用eager模式避免内存问题
        4. 小批量 decode：使用CUDA Graph获得最优性能
        
        CUDA Graph模式优势：
        - 减少kernel启动开销
        - 减少CPU-GPU同步
        - 提高小批量推理吞吐量
        
        Args:
            input_ids: 输入token ID序列
            positions: 位置编码序列
            is_prefill: 是否为prefill阶段
            
        Returns:
            torch.Tensor: 模型输出logits
        """
        # 策略选择：prefill/强制eager/大批量 使用eager模式
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            # Eager模式：直接执行模型前向传播
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            # CUDA Graph模式：使用预先捕获的计算图
            bs = input_ids.size(0)  # 当前批量大小
            context = get_context()  # 获取当前推理上下文
            
            # 选择适合的计算图（找到第一个 >= 当前批量的图）
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars  # 预分配的变量缓冲区
            
            # 清零除输出外的所有变量（避免上次数据污染）
            for k, v in graph_vars.items():
                if k != "outputs":
                    v.zero_()
            
            # 将当前数据复制到图变量中
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            
            # 执行计算图（单次API调用）
            graph.replay()
            
            # 返回计算结果
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        """
        推理主流程 - 统一处理prefill和decode阶段
        
        执行流程：
        1. 数据准备：根据阶段选择不同的数据预处理
        2. 采样参数：仅主进程准备温度参数
        3. 模型推理：执行前向传播获得logits
        4. token采样：仅主进程执行采样并返回结果
        5. 上下文清理：释放当前推理的上下文资源
        
        分工设计：
        - 所有进程：同步执行模型推理
        - 主进程：额外负责采样参数和token采样
        - 工作进程：不需要采样，返回None
        
        Args:
            seqs: 需要处理的序列列表
            is_prefill: True为prefill阶段，False为decode阶段
            
        Returns:
            list[int] | None: 采样得到的token ID列表（仅主进程返回）
        """
        # 1. 根据阶段选择数据准备方法
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        
        # 2. 仅主进程准备采样参数（工作进程不需要）
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        
        # 3. 执行模型推理（所有进程同步执行）
        logits = self.run_model(input_ids, positions, is_prefill)
        
        # 4. 仅主进程执行token采样
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        
        # 5. 清理全局上下文状态
        reset_context()
        
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        """
        CUDA Graph捕获 - Decode阶段的极致性能优化
        
        功能概述:
        将模型的decode计算预先录制成CUDA计算图，后续通过graph.replay()直接执行
        避免重复的kernel启动开销，显著提升小批量推理的吞吐量和降低延迟
        
        CUDA Graph核心原理:
        1. 记录阶段: 捕获一次完整的GPU计算序列
        2. 重放阶段: 通过单次API调用执行整个计算序列
        3. 性能提升: 减少CPU-GPU同步，批量提交GPU操作
        
        优化策略:
        - 多批量大小支持: 覆盖1到512的各种批量大小
        - 内存池复用: 所有图共享内存池，减少内存碎片  
        - 预热机制: 确保kernel已优化后再捕获
        - 逆序捕获: 从大到小避免内存分配问题
        
        限制条件:
        - 仅适用于decode阶段 (固定计算模式)
        - 批量大小≤512 (内存和复杂度平衡)
        - 计算图结构必须完全相同
        """
        config = self.config
        hf_config = config.hf_config
        
        # 1. 配置参数计算
        max_bs = min(self.config.max_num_seqs, 512)  # 最大批量限制512(性能vs内存平衡)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size  # 最大块数
        
        # 2. 预分配所有必要张量 (使用最大尺寸，后续通过切片复用)
        input_ids = torch.zeros(max_bs, dtype=torch.int64)           # 输入token IDs
        positions = torch.zeros(max_bs, dtype=torch.int64)           # 位置编码
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)        # KV缓存槽位映射
        context_lens = torch.zeros(max_bs, dtype=torch.int32)        # 上下文长度
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)  # 块表
        outputs = torch.zeros(max_bs, hf_config.hidden_size)         # 模型输出缓冲区
        
        # 3. 批量大小策略设计
        # 小批量精细化: [1,2,4,8] 覆盖常见小批量场景
        # 大批量步进: [16,32,48...] 减少图数量，平衡内存使用
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}        # 存储不同批量大小的计算图
        self.graph_pool = None  # 共享内存池 (所有图复用同一内存空间)

        # 4. 逐个批量大小捕获计算图 (逆序：从大到小避免内存碎片)
        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()  # 创建新的计算图对象
            
            # 4.1 设置当前批量的上下文环境
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            
            # 4.2 预热阶段: 确保所有CUDA kernels已加载和优化
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
            
            # 4.3 捕获阶段: 记录完整的计算序列到图中
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
            
            # 4.4 内存池管理: 首次创建内存池，后续图复用
            if self.graph_pool is None:
                self.graph_pool = graph.pool()  # 创建共享内存池
            
            # 4.5 存储图并同步GPU操作
            self.graphs[bs] = graph              # 保存计算图供运行时使用
            torch.cuda.synchronize()             # 确保图捕获完全完成
            reset_context()                      # 清理上下文，避免状态污染

        # 5. 保存可复用的张量变量 (运行时直接修改这些张量，无需重新分配)
        self.graph_vars = dict(
            input_ids=input_ids,        # 输入token缓冲区
            positions=positions,        # 位置编码缓冲区
            slot_mapping=slot_mapping,  # KV缓存映射缓冲区
            context_lens=context_lens,  # 上下文长度缓冲区  
            block_tables=block_tables,  # 块表缓冲区
            outputs=outputs,            # 输出缓冲区
        )
