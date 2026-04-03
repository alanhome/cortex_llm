import torch
from torch.utils.data import IterableDataset
from torch.nn.utils.rnn import pad_sequence
import glob
import os
import random
import numpy as np
from config import CONFIG

class StreamingDataset(IterableDataset):
    def __init__(self, data_dir, pad_token_id=151643, seed=42, chunk_size=2000):
        """
        Args:
            data_dir: 数据目录
            chunk_size: 预处理时设定的每个 chunk 的大小 (用于估算跳过步数)
        """
        self.pad_token_id = pad_token_id
        self.seed = seed
        self.epoch = 0
        self.skip_samples = 0  # 需要跳过的样本数
        
        # 1. 扫描所有文件并排序（保证基础顺序一致）
        self.files = sorted(glob.glob(os.path.join(data_dir, "*_chunk_*.pt")))
        if len(self.files) == 0:
            print(f"⚠️ Warning: No .pt files found in {data_dir}")
        else:
            print(f"📂 Found {len(self.files)} chunks. Mode: Streaming (Low RAM)")

        # 2. 估算总长度 (用于 tqdm 显示进度条)
        # 注意：这里是估算，假设每个 chunk 差不多满。如果需要精确，需要预先扫描一遍 metadata
        self.estimated_total_samples = len(self.files) * chunk_size
        self.max_length = CONFIG['max_length'] 

    def set_epoch(self, epoch):
        """每个 Epoch 开始前调用，改变 Shuffle 种子"""
        self.epoch = epoch

    def set_skip_samples(self, n):
        """断点续训用：设置需要跳过的样本数"""
        self.skip_samples = n

    def __len__(self):
        return self.estimated_total_samples

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        
        # 1. 确定性随机种子 (Base Seed + Epoch)
        # 保证每个 Epoch 的文件乱序都不一样，但重启后同一个 Epoch 顺序一致
        rng = random.Random(self.seed + self.epoch)
        
        # 2. 对文件列表进行全局乱序
        shuffled_files = self.files.copy()
        rng.shuffle(shuffled_files)

        # 3. 多进程切分 (Sharding)
        if worker_info is not None:
            # 如果有多个 worker，每个 worker 只处理一部分文件
            # 例如 worker 0 处理 [0, 2, 4], worker 1 处理 [1, 3, 5]
            shuffled_files = shuffled_files[worker_info.id :: worker_info.num_workers]
        
        # 4. 计算 Worker 内部需要跳过的数量 (Fast Forward)
        # 只有在 skip_samples > 0 时生效
        samples_processed = 0
        
        # 5. 流式加载主循环
        for f_path in shuffled_files:
            try:
                # 如果还需要跳过大量数据，且已知这个文件大概的大小，可以直接跳过文件读取 (优化 IO)
                # 这里为了精确性，我们还是读取文件，但在 yield 之前快速判断
                # 优化：如果 chunk 非常大，建议在这里加逻辑直接跳过文件
                
                # 加载一个 Chunk 到内存
                chunk_data = torch.load(f_path, map_location="cpu")
                
                # 文件内乱序 (Block Shuffle)
                # 使用同一个 rng 保证确定性
                rng.shuffle(chunk_data)

                for item in chunk_data:
                    # 断点续训逻辑：如果还没跳过足够的样本，就 continue
                    if self.skip_samples > 0:
                        self.skip_samples -= 1
                        continue
                    
                    yield item

                # 显式删除引用，帮助 GC 及时回收内存
                del chunk_data

            except Exception as e:
                print(f"⚠️ Error reading {f_path}: {e}")
                continue

    def collate(self, batch):
        """
        保持原有的 Padding 逻辑不变
        """
        input_ids = [item['input_ids'][:self.max_length] for item in batch]
        labels = [item['labels'][:self.max_length] for item in batch]
        attention_mask = [item['attention_mask'][:self.max_length] for item in batch]

        # Padding: 使用正确的 pad_token_id
        input_ids_pad = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        
        # Label 必须 pad 为 -100 (这是 PyTorch CrossEntropyLoss 的忽略索引，不需要改)
        labels_pad = pad_sequence(labels, batch_first=True, padding_value=-100)
        
        # Attention Mask pad 为 0 (表示不关注)
        attention_mask_pad = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        #if input_ids_pad.shape[1] > 1024 * 7:
        #    print("elba check: ", input_ids_pad.shape, labels_pad.shape, attention_mask_pad.shape, self.max_length)

        return {
            'input_ids': input_ids_pad,
            'labels': labels_pad,
            'attention_mask': attention_mask_pad
        }
