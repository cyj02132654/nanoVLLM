from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence

# 核心功能:
# - Block类: 单个内存块，包含引用计数和哈希值
# - BlockManager类: 实现基于哈希的缓存共享机制
# - 使用xxhash进行快速哈希计算
# - 引用计数管理，支持安全的内存回收
# - 增量扩展支持，适合流式生成
# - 前缀缓存优化，避免重复计算

# 关键方法:
# - allocate(): 为序列分配内存块，支持缓存命中检测
# - deallocate(): 释放内存块，自动处理引用计数
# - can_append(): 检查是否可以追加新token

class Block:
    """
    表示一个内存块，用于存储token序列的一部分。
    包含引用计数机制，支持多个序列共享同一个块。
    """

    def __init__(self, block_id):
        """初始化内存块
        
        Args:
            block_id: 块的唯一标识符
        """
        self.block_id = block_id  # 块的唯一ID
        self.ref_count = 0        # 引用计数，用于内存管理
        self.hash = -1           # 块内容的哈希值，-1表示未计算
        self.token_ids = []      # 存储的token ID列表

    def update(self, hash: int, token_ids: list[int]):
        """更新块的哈希值和token内容
        
        Args:
            hash: 新的哈希值
            token_ids: 新的token ID列表
        """
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        """重置块状态，将引用计数设为1，清空哈希和token内容"""
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:
    """
    块管理器，负责内存块的分配、释放和缓存管理。
    
    核心特性:
    - 基于哈希的缓存共享机制，避免重复存储相同内容
    - 引用计数管理，确保内存安全释放
    - 支持前缀缓存，提高缓存命中率
    """

    def __init__(self, num_blocks: int, block_size: int):
        """初始化块管理器
        
        Args:
            num_blocks: 总块数量
            block_size: 每个块的大小（token数量）
        """
        assert num_blocks > 0
        self.block_size = block_size                                    # 每个块的token容量
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]  # 所有内存块
        self.hash_to_block_id: dict[int, int] = dict()                 # 哈希值到块ID的映射，用于缓存查找
        self.free_block_ids: deque[int] = deque(range(num_blocks))     # 空闲块ID队列
        self.used_block_ids: set[int] = set()                          # 已使用块ID集合

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        """计算token序列的哈希值
        
        Args:
            token_ids: token ID列表
            prefix: 前缀哈希值，用于增量计算
            
        Returns:
            计算得到的64位哈希值
        """
        h = xxhash.xxh64()  # 创建64位哈希对象
        if prefix != -1:
            # 如果有前缀，先加入前缀哈希（用于增量计算）
            h.update(prefix.to_bytes(8, "little"))
        # 将token ID列表转为字节数组后加入哈希计算
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()  # 返回整数形式的哈希值

    def _allocate_block(self, block_id: int) -> Block:
        """分配指定的内存块
        
        Args:
            block_id: 要分配的块ID
            
        Returns:
            分配的块对象
        """
        block = self.blocks[block_id]
        assert block.ref_count == 0  # 确保块未被使用
        block.reset()                # 重置块状态
        self.free_block_ids.remove(block_id)  # 从空闲队列移除
        self.used_block_ids.add(block_id)     # 加入已使用集合
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> Block:
        """释放指定的内存块，将其重新加入空闲队列
        
        Args:
            block_id: 要释放的块ID
        """
        assert self.blocks[block_id].ref_count == 0  # 确保无引用
        self.used_block_ids.remove(block_id)         # 从已使用集合移除
        self.free_block_ids.append(block_id)         # 加入空闲队列

    def can_allocate(self, seq: Sequence) -> bool:
        """检查是否有足够的空闲块来分配给序列
        
        Args:
            seq: 需要分配内存的序列
            
        Returns:
            如果有足够空闲块返回True，否则返回False
        """
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        """为序列分配内存块，优先使用缓存
        
        该方法实现了智能缓存机制：
        1. 对每个块计算哈希值
        2. 检查哈希映射表寻找缓存命中
        3. 如果命中且内容匹配，复用现有块（增加引用计数）
        4. 如果未命中，分配新块
        
        Args:
            seq: 需要分配内存的序列对象
        """
        assert not seq.block_table  # 确保序列还没有分配块表
        h = -1                      # 前一个块的哈希值，用于增量计算
        cache_miss = False          # 缓存未命中标志
        for i in range(seq.num_blocks):  # 遍历序列需要的所有块
            token_ids = seq.block(i)    # 获取第i个块的token内容
            # 只有满块才计算哈希（用于缓存），不满的块哈希为-1
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            block_id = self.hash_to_block_id.get(h, -1)  # 查找缓存
            # 检查缓存未命中或内容不匹配的情况
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True
            if cache_miss:
                # 缓存未命中，分配新块
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                # 缓存命中，增加缓存token计数
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    # 块正在被其他序列使用，增加引用计数
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    # 块在缓存中但未被使用，重新分配
                    block = self._allocate_block(block_id)
            if h != -1:
                # 更新块的哈希和内容，加入缓存映射
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            seq.block_table.append(block_id)  # 将块ID加入序列的块表

    def deallocate(self, seq: Sequence):
        """释放序列占用的内存块
        
        采用引用计数机制：
        1. 减少每个块的引用计数
        2. 当引用计数为0时，释放块到空闲队列
        3. 清空序列的块表和缓存计数
        
        Args:
            seq: 要释放内存的序列对象
        """
        for block_id in reversed(seq.block_table):  # 倒序释放块
            block = self.blocks[block_id]
            block.ref_count -= 1                    # 减少引用计数
            if block.ref_count == 0:                # 无引用时释放块
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0                   # 重置缓存token计数
        seq.block_table.clear()                     # 清空块表

    def can_append(self, seq: Sequence) -> bool:
        """检查序列是否可以追加新的token
        
        当序列长度模块大小等于1时，需要新块来容纳后续token
        
        Args:
            seq: 要检查的序列
            
        Returns:
            如果可以追加返回True，否则返回False
        """
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        """为序列追加操作准备显存块
        
        根据序列当前长度的不同情况：
        1. 长度%block_size == 1: 需要分配新块
        2. 长度%block_size == 0: 完整块，更新哈希并加入缓存
        3. 其他情况: 块未满，无需操作
        
        Args:
            seq: 要追加token的序列
        """
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        if len(seq) % self.block_size == 1:
            # 需要新块：上一个块已满，当前块只有1个token
            assert last_block.hash != -1  # 上一个块应该已经有哈希
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        elif len(seq) % self.block_size == 0:
            # 块刚好填满：计算哈希并加入缓存
            assert last_block.hash == -1      # 当前块还没有哈希
            token_ids = seq.block(seq.num_blocks-1)  # 获取最后一个块的内容
            # 获取前一个块的哈希作为前缀，用于增量计算
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)     # 更新块信息
            self.hash_to_block_id[h] = last_block.block_id  # 加入缓存映射
        else:
            # 块未满：无需任何操作，等待更多token
            assert last_block.hash == -1      # 未满的块不应该有哈希
