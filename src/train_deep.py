import torch
import torch._dynamo
torch.backends.cuda.matmul.fp32_precision = 'tf32'
torch.backends.cudnn.conv.fp32_precision = 'tf32'

from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import torch.nn.functional as F  # ✅ 补上了这个关键的包
import wandb
import os
import glob
import re
from tqdm import tqdm
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import traceback
import random           # <--- 新增
import numpy as np      # <--- 新增
from collections import defaultdict # <--- 新增
import gc

# ✅ 引入本地模块
from model_deep import CortexDeepModel, CortexDeepConfig
from config import CONFIG
from dataset import StreamingDataset # [修正] 必须引入 StreamingDataset

import bitsandbytes as bnb  # <--- 引入这个

import sys
import os
import ctypes

# 在 optimizer.step() 之后，插入这个强力检查：
# ==========================================
# 🕵️ Cortex 训练取证工具
# ==========================================
def debug_optimizer_status(model, optimizer, step):
    print(f"\n[Debug Step {step}] Analyzing Optimizer & Weights...")

    # 1. 检查 Latents 是否在优化器管辖范围内
    latents_in_optimizer = False
    for group in optimizer.param_groups:
        for p in group['params']:
            if p is model.latents: # 检查内存地址是否一致
                latents_in_optimizer = True
                print(f"✅ Latents found in optimizer. Current LR: {group['lr']:.2e}")
                break

    if not latents_in_optimizer:
        print("❌ CRITICAL: 'model.latents' is NOT in the optimizer! Weights will never update.")
        return

    # 2. 检查梯度
    if model.latents.grad is None:
        print("❌ CRITICAL: 'model.latents' has NO gradient (None). Backward broken.")
    else:
        grad_norm = model.latents.grad.norm().item()
        print(f"ℹ️ Latents Gradient Norm: {grad_norm:.6f}")

        # 3. 检查权重是否发生物理位移
        # 我们保存一个旧副本做对比 (利用函数属性做静态变量)
        if not hasattr(debug_optimizer_status, 'prev_latents'):
            debug_optimizer_status.prev_latents = model.latents.detach().clone()
            print("ℹ️ Reference weights saved. Wait for next step to compare.")
        else:
            diff = (model.latents - debug_optimizer_status.prev_latents).abs().sum().item()
            print(f"📉 Weight Delta (L1 Distance): {diff:.9f}")

            if diff == 0.0 and grad_norm > 0:
                print("🚨 ALARM: Gradient exists, but weights did NOT move! Check LR=0 or Scheduler.")
            elif diff > 0:
                print("✅ Weights are moving! Training is active.")

            # 更新引用
            debug_optimizer_status.prev_latents = model.latents.detach().clone()
    print("------------------------------------------\n")


def set_seed(seed=42):
    """固定随机种子，保证 DataLoader Shuffle 的顺序在恢复训练时一致"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_latest_checkpoint(output_dir):
    """自动查找目录下 Step 最大的 Checkpoint"""
    files = glob.glob(os.path.join(output_dir, "step_*.pt"))
    if not files:
        return None

    def extract_step(path):
        match = re.search(r'step_(\d+)\.pt', path)
        return int(match.group(1)) if match else -1

    latest_file = max(files, key=extract_step)
    return latest_file

def cleanup_checkpoints(output_dir, max_to_keep):
    """清理旧的 Checkpoint，只保留最新的几个"""
    files = glob.glob(os.path.join(output_dir, "step_*.pt"))

    def extract_step(path):
        match = re.search(r'step_(\d+)\.pt', path)
        return int(match.group(1)) if match else -1

    sorted_files = sorted(files, key=extract_step)

    if len(sorted_files) > max_to_keep:
        to_delete = sorted_files[: -max_to_keep]
        for f in to_delete:
            try:
                os.remove(f)
                print(f"🧹 Cleaned old checkpoint: {os.path.basename(f)}")
            except OSError as e:
                print(f"⚠️ Error deleting {f}: {e}")

def save_checkpoint(model, optimizer, scheduler, step, epoch, output_dir):
    save_path = os.path.join(output_dir, f"step_{step}.pt")

    # 尽可能只保存权重，不保存整个模型对象
    # 1. 获取完整的 state_dict
    raw_state_dict = model.state_dict()

    # 2. 🔥 过滤：只保存 requires_grad=True 的参数 (即 Cortex 部分)
    # 这样生成的 pt 文件非常小，只包含你训练的部分
    cortex_state_dict = {k: v for k, v in raw_state_dict.items() if "base_model" not in k}

    print(f"💾 Saving checkpoint (Cortex only, {len(cortex_state_dict)} tensors)...")
    torch.save({
        'step': step,
        'epoch': epoch,
        'model_state_dict': cortex_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, save_path)

    max_keep = CONFIG.get('max_checkpoints_to_keep', 3)
    cleanup_checkpoints(output_dir, max_keep)

def compute_diversity_loss(thought_vectors):
    """
    thought_vectors: [Batch, Num_Tokens, Dim]
    """
    # 1. 归一化，准备算余弦相似度
    thoughts_norm = F.normalize(thought_vectors, p=2, dim=-1)

    # 2. 计算相邻 Token 的相似度 (Token_i 与 Token_i+1)
    # 我们不希望相邻的 Token 一模一样
    # [B, N-1, D] * [B, N-1, D] -> [B, N-1]
    sim_neighbors = (thoughts_norm[:, :-1, :] * thoughts_norm[:, 1:, :]).sum(dim=-1)

    # 3. 计算所有 Token 与 "平均 Token" 的相似度 (防止整体坍塌到一个点)
    mean_thought = thoughts_norm.mean(dim=1, keepdim=True) # [B, 1, D]
    sim_mean = (thoughts_norm * mean_thought).sum(dim=-1)  # [B, N]

    # Loss = 相邻相似度 + 全局相似度 (越高越坏)
    loss = sim_neighbors.mean() + sim_mean.mean()
    return loss

class MemoryKoLeoLoss(nn.Module):
    def __init__(self, hidden_dim, memory_size=20480):
        super().__init__()
        self.memory_size = memory_size
        # 注册 buffer，不随梯度更新，但在 state_dict 中保存
        self.register_buffer("memory", F.normalize(torch.randn(memory_size, hidden_dim), p=2, dim=-1))
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))

    def forward(self, thought_vectors, epsilon=1e-8):
        x = thought_vectors.to(torch.float32)
        # 1. 当前数据处理: [1, N, D] -> [N, D]
        x = thought_vectors.reshape(-1, thought_vectors.shape[-1])
        x = F.normalize(x, p=2, dim=-1)
        N = x.shape[0]

        # ====================================================
        # 🔥 Part A: 内部互斥 (Intra-Sequence Diversity)
        # 解决: 防止同一句话里的 16 个 Token 长得一模一样
        # ====================================================
        # 计算 x 内部两两距离 [N, N]
        sim_matrix = torch.matmul(x, x.T)
        sim_matrix = torch.clamp(sim_matrix, max=1.0)
        dists_intra = torch.sqrt(2.0 - 2.0 * sim_matrix + 1e-4)

        # 屏蔽对角线 (自己到自己的距离是0，不应该算入 Loss)
        eye_mask = torch.eye(N, device=x.device, dtype=torch.bool)
        dists_intra_masked = dists_intra.masked_fill(eye_mask, float('inf'))
        min_dist_intra, _ = dists_intra_masked.min(dim=1)

        # Loss A: 强迫序列内部散开
        loss_intra = -torch.log(min_dist_intra + epsilon).mean()

        ## ====================================================
        ## 🧊 Part B: 历史互斥 (Inter-Sequence Diversity)
        ## 解决: 防止当前这句话跟以前说过的车轱辘话一样
        ## ====================================================
        #mem = self.memory.detach().to(x.dtype)

        ## 计算 x 到 memory 的距离 [N, M]
        #dists_inter = torch.cdist(x, mem, p=2)

        ## 找到离历史记录最近的距离
        #min_dist_inter, _ = dists_inter.min(dim=1)

        ## Loss B: 强迫当前 Token 占据新的语义空间
        #loss_inter = -torch.log(min_dist_inter + epsilon).mean()

        # ====================================================
        # ⚖️ 总 Loss & 更新记忆
        # ====================================================

        # 📝 策略：内部互斥权重通常要大一些，因为 Collapse 首先发生在内部
        total_loss = loss_intra 

        ## 更新记忆库 (保持不变)
        #with torch.no_grad():
        #    batch_num = x.shape[0]
        #    ptr = int(self.ptr.item())
        #    x_detach = x.detach()

        #    if ptr + batch_num <= self.memory_size:
        #        self.memory[ptr : ptr + batch_num] = x_detach
        #        self.ptr[0] = (ptr + batch_num) % self.memory_size
        #    else:
        #        tail = self.memory_size - ptr
        #        self.memory[ptr:] = x_detach[:tail]
        #        self.memory[:batch_num - tail] = x_detach[tail:]
        #        self.ptr[0] = batch_num - tail

        return total_loss

@torch.no_grad()
def visualize_dynamic_thoughts(model, tokenizer, thought_embeds, top_k=1):
    """
    [GB10 / Grace Blackwell 专用版]
    针对 LPDDR5x 统一内存优化：
    1. Zero-Copy: 绝不把词表权重搬运到 CPU，直接在原位(Unified Memory)读取。
    2. Tensor Core Compute: 利用 Blackwell 强大的算力在 GPU 上完成相似度计算。
    3. Low Bandwidth Mode: 仅传回极小的 indices 结果，最小化内存带宽压力。
    """
    try:
        tqdm.write("\n" + "="*20 + " 🧠 [Cortex Real-time Thoughts] " + "="*20)

        # 1. 准备数据 (全部保持在 GPU 上，不移动)
        # thought_embeds [1, Seq_Len, Dim] -> [Seq_Len, Dim]
        dtype = model.base_model.dtype
        target_vecs = thought_embeds[0].detach().squeeze(0).to(dtype=dtype)
        seq_len, dim = target_vecs.shape

        # --- [新增] 统计量监控 (Mean/Std/Drift) ---
        # 监控数值分布
        v_mean, v_std = target_vecs.mean().item(), target_vecs.std().item()
        v_norm = target_vecs.norm(dim=-1).mean().item()

        # 监控更新幅度 (与上一次 Debug 时的差异)
        if not hasattr(visualize_dynamic_thoughts, 'last_vecs'):
            visualize_dynamic_thoughts.last_vecs = None
            drift_msg = "First Step (No History)"
        else:
            # 确保形状一致才能相减
            if visualize_dynamic_thoughts.last_vecs.shape == target_vecs.shape:
                diff = (target_vecs - visualize_dynamic_thoughts.last_vecs).abs().mean().item()
                drift_msg = f"{diff:.9f}" if diff != 0 else "⚠️ 0.00000000 (STATIC!)"
            else:
                drift_msg = "Shape Mismatch"

        visualize_dynamic_thoughts.last_vecs = target_vecs.clone()
        tqdm.write(f"📊 Stats | Mean: {v_mean:.4f} | Std: {v_std:.4f} | Norm: {v_norm:.2f}")
        tqdm.write(f"📉 Drift | Delta vs Last: {drift_msg}")
        
        # --- [新增] 多样性检测 (Diversity Check) ---
        # 检测是否发生 Mode Collapse (所有 token 都一样)
        normed_vecs = F.normalize(target_vecs, p=2, dim=-1)
        if seq_len > 10:
            # 随机抽样 100 个点计算平均自相似度
            idx = torch.randperm(seq_len)[:100]
            sampled = normed_vecs[idx]
            sim_matrix = torch.matmul(sampled, sampled.T)
            # 去掉对角线
            mask = ~torch.eye(len(idx), dtype=torch.bool, device=sim_matrix.device)
            avg_self_sim = sim_matrix[mask].mean().item()
        else:
            avg_self_sim = 0.0
            
        tqdm.write(f"🧬 Diversity | Self-Sim(Avg): {avg_self_sim:.4f}")
        if avg_self_sim > 0.90:
             tqdm.write("⚠️ CRITICAL: Latent Collapse! All tokens are identical.")

        # ========================================================
        # 🔬 [新增] Embedding 数值透视镜
        # ========================================================
        tqdm.write("-" * 65)
        # 1. 打印前 3 个 Token 的前 8 个维度的数值
        num_show_tokens = min(16, seq_len)
        num_show_dims = 8
        tqdm.write("🔍 Raw Values (First 8 dims):")

        for i in range(num_show_tokens):
            # 转 float 避免 bfloat16 打印精度问题
            vals = target_vecs[i, :num_show_dims].float().cpu().numpy()
            # 格式化打印：保留3位小数，带符号
            vals_str = ", ".join([f"{v:+.2f}" for v in vals])
            tqdm.write(f"  Token {i}: [{vals_str}, ...]")

        # 2. 打印相邻 Token 的“物理距离” (验证 KoLeo Loss 是否生效)
        if seq_len >= 2:
            # 计算 Token 0 和 Token 1 的余弦相似度 (-1 ~ 1)
            sim_01 = F.cosine_similarity(target_vecs[0].unsqueeze(0), target_vecs[1].unsqueeze(0)).item()
            # 计算欧氏距离
            dist_01 = torch.norm(target_vecs[0] - target_vecs[1]).item()

            tqdm.write(f"📏 Separation (T0 vs T1): CosSim={sim_01:+.4f} | L2_Dist={dist_01:.4f}")

            if sim_01 > 0.99:
                tqdm.write("  ⚠️ Warning: T0 and T1 are almost identical! (Collapse Risk)")
            elif sim_01 < 0.5:
                tqdm.write("  ✅ Good: Tokens are well separated.")
        tqdm.write("-" * 65)
        # ========================================================

        # -------------------------------------------

        # 直接引用模型权重的指针 (Zero-Copy)
        # 在 GB10 上，这块内存既是显存也是内存，直接读即可
        vocab_weight = model.base_model.get_input_embeddings().weight.data

        # 2. 计算 (利用 GPU Tensor Cores)
        # 建议保持 bf16/fp16 精度以节省带宽 (GB10 对 BF16 优化极好)
        
        # [修改] 只采样前5和后5，显示置信度，避免刷屏
        num_samples = min(128, seq_len)
        indices_to_probe = list(range(num_samples))
        #if seq_len > 100: indices_to_probe += list(range(seq_len-50, seq_len))
        
        probe_vecs = target_vecs[indices_to_probe]
        probe_norm = F.normalize(probe_vecs, p=2, dim=-1)
        vocab_norm = F.normalize(vocab_weight, p=2, dim=-1)

        # Matrix Multiplication: [10, Dim] @ [Dim, Vocab] -> [10, Vocab]
        # 这一步是内存尖峰，但在 128GB 统一内存下通常是安全的
        similarity = torch.matmul(probe_norm, vocab_norm.T)

        # 3. 提取结果 (Top-K)
        # 计算依然在 GPU 完成
        scores, indices = torch.topk(similarity, k=top_k, dim=-1)

        # 4. 只有最后这几个整数才搬运回 CPU 用于打印
        # 数据量极小 (几百 Bytes)，不会造成带宽压力
        indices_cpu = indices.to("cpu")
        scores_cpu = scores.to("cpu")

        tqdm.write(f"🧐 Sampling {len(indices_to_probe)} tokens (Head & Tail) with Confidence:")
        line_parts = []

        for i, token_idx in enumerate(indices_to_probe):
            for k in range(top_k):
                idx_val = indices_cpu[i, k].item()
                score_val = scores_cpu[i, k].item()
                
                try:
                    w = tokenizer.decode([idx_val]).replace('\n', '↵').replace('\r', '').strip()
                    if not w: w = '░'
                except:
                    w = ""
                
                # 格式: Word(Score)
                line_parts.append(f"{w}({score_val:.2f})")

        tqdm.write(f" ".join(line_parts))
        tqdm.write("="*65 + "\n")

        # 5. 关键：手动释放大张量
        # 虽然 Python 有 GC，但在内存敏感环境下，显式删除引用是个好习惯
        del similarity, probe_norm, vocab_norm, target_vecs, normed_vecs

    except Exception as e:
        tqdm.write(f"⚠️ Vis Error: {e}")
        # 如果真的爆了内存，尝试清空缓存救一下
        if "out of memory" in str(e):
            print("⚠️ OOM detected in Vis! Skipping.")
            torch.cuda.empty_cache()

def train():
    set_seed(42)

    #os.environ["WANDB_MODE"] = "offline"
    # 1. 初始化 WandB
    wandb.init(project=CONFIG['project_name'], config=CONFIG)
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 🔥 [新增] 初始化 KoLeo Loss 模块
    print("🧠 Initializing Memory Bank for KoLeo Loss...")
    koleo_loss_fn = MemoryKoLeoLoss(
        hidden_dim=CONFIG['cortex_hidden_dim'],
        memory_size=4096 # 建议设为 2048 或 4096
    ).to(device)

    print(f"🚀 Initializing Model on {device}...")

    # 2. 模型初始化 (配置 -> 模型)
    model_config = CortexDeepConfig(
        base_model_path=CONFIG['base_model'],
        num_thought_tokens=CONFIG['num_thought_tokens'],
        cortex_hidden_dim=CONFIG['cortex_hidden_dim'],
        encoder_layers=CONFIG['encoder_layers'],
        encoder_heads=CONFIG['encoder_heads'],
        num_heavy_experts=CONFIG.get('num_heavy_experts', 4) # 确保读取配置
    )

    # 使用 bfloat16 节省显存
    #model = CortexDeepModel(model_config).to(device, dtype=torch.bfloat16)
    model = CortexDeepModel(model_config).to(device)

    # 在 train_deep.py 或单独的测试脚本中
    print(f"当前 Attention 实现模式: {model.base_model.config._attn_implementation}")

    # 4. 加载 Tokenizer 并修复 Padding ID
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['base_model'], trust_remote_code=True)

    # 🔥 关键修复：确保 pad_token_id 正确
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token_id = 151643 # Qwen 默认 <|endoftext|>
        print(f"✅ Set tokenizer.pad_token_id to {tokenizer.pad_token_id}")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🔥 System 2 Trainable Params: {trainable_params / 1e6:.2f} M")

    # 5. 数据加载 (传入修复后的 pad_token_id)
    # [修正] 必须使用 StreamingDataset
    ds = StreamingDataset(
        data_dir=CONFIG['raw_data_cache'], 
        pad_token_id=tokenizer.pad_token_id, 
        chunk_size=CONFIG.get('chunk_size'), 
        seed=42
    )
    print(f"📂 Dataset loaded: {len(ds)} samples (Estimated).")

    loader = DataLoader(
        ds,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        collate_fn=ds.collate, # 使用实例方法
        num_workers=1,         # 适当加速数据读取
        pin_memory=True,
        prefetch_factor=4
    )

    # 6. 优化器与调度器
    #optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=CONFIG['lr'])

    ## 👇👇👇 [修改] 分层学习率：Latents 需要更大的梯度才能移动 👇👇👇
    latents_params = [p for n, p in model.named_parameters() if 'latents' in n and p.requires_grad]
    other_params = [p for n, p in model.named_parameters() if 'latents' not in n and p.requires_grad]

    #print("🔧 Using Native Fused AdamW Optimizer (Max Speed)...")
    #optimizer = torch.optim.AdamW([
    #    {'params': other_params, 'lr': CONFIG['lr']},
    #    {'params': latents_params, 'lr': CONFIG['lr']}
    #], fused=True) # 👈 fused=True 会调用极速 C++ 内核

    # [新代码] 使用 8-bit 优化器
    print("🔧 Using 8-bit AdamW Optimizer (Saving massive VRAM)...")
    optimizer = bnb.optim.AdamW8bit([
        {'params': other_params, 'lr': CONFIG['lr']},
        {'params': latents_params, 'lr': CONFIG['lr']}
    ])
    print(f"🔧 Optimized LR: Base={CONFIG['lr']}, Latents={CONFIG['lr']}, Batch size={CONFIG['batch_size']}, grad accum = {CONFIG['grad_accum_steps']}")

    total_steps = len(ds) * CONFIG['epochs'] // (CONFIG['grad_accum_steps'] * CONFIG['batch_size'])
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    # 7. 断点续训逻辑
    start_epoch = 0
    global_step = 0
    resume_path = None

    if CONFIG.get('resume_from_checkpoint', False):
        if isinstance(CONFIG['resume_from_checkpoint'], str):
            resume_path = CONFIG['resume_from_checkpoint']
        else:
            resume_path = get_latest_checkpoint(CONFIG['output_dir'])

    if resume_path and os.path.exists(resume_path):
        print(f"🔄 Resuming from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)

        # ====================================================
        # 🛡️ 第一道防线：清理 Model 权重 (你之前已经做过的)
        # ====================================================
        state_dict_ref = checkpoint['model_state_dict']
        keys_to_delete = [k for k in state_dict_ref.keys() if "base_model" in k]

        if len(keys_to_delete) > 0:
            print(f"🧹 Detected {len(keys_to_delete)} frozen base_model weights. Deleting...")
            for k in keys_to_delete:
                del state_dict_ref[k]

        # 修复 torch.compile 前缀
        keys_to_rename = [k for k in state_dict_ref.keys() if "_orig_mod." in k]
        for k in keys_to_rename:
            val = state_dict_ref.pop(k)
            new_key = k.replace("_orig_mod.", "")
            state_dict_ref[new_key] = val

        # 立即回收内存
        gc.collect()

        # 加载净化后的权重进 GPU
        print(f"📉 Loading Model weights to GPU...")
        msg = model.load_state_dict(state_dict_ref, strict=False)
        print(f"✅ Model weights loaded. Missing keys: {len(msg.missing_keys)}")

        # ====================================================
        # 🛡️ 第二道防线：清理 Optimizer 状态 (OOM 的罪魁祸首)
        # ====================================================
        print("🔍 Checking Optimizer State health...")
        try:
            optim_state = checkpoint['optimizer_state_dict']

            # 1. 检查参数数量是否匹配
            # 这里的逻辑是：如果 Checkpoint 里的参数组数量比当前优化的参数组多太多，说明它存了 Base Model
            saved_param_groups = len(optim_state['param_groups'])
            current_param_groups = len(optimizer.param_groups)

            # 2. 检查状态字典的大小 (粗略估计)
            state_len = len(optim_state['state'])

            # 策略：如果保存的状态数量远大于我们现在要训练的参数量，说明它是“有毒”的
            # 我们直接丢弃它，从头初始化优化器 (这比 OOM 要好得多)
            total_cortex_params = sum(len(g['params']) for g in optimizer.param_groups) 

            print(f"   - Saved Optimizer has state for {state_len} tensors")
            print(f"   - Current Optimizer expects {total_cortex_params} tensors")

            if state_len > total_cortex_params * 1.5: # 留一点 buffer
                print(f"⚠️ [CRITICAL] Optimizer state in checkpoint is HUGE ({state_len} tensors vs expected {total_cortex_params}).")
                print(f"   It likely contains frozen Base Model states which causes OOM.")
                print(f"👉 ACTION: DISCARDING saved optimizer state and starting optimizer from scratch.")
                print(f"   (This is safe. Momentum will rebuild quickly.)")

                # 不加载优化器状态，直接跳过
                del optim_state
                checkpoint['optimizer_state_dict'] = None
            else:
                print("✅ Optimizer state looks healthy. Loading...")
                optimizer.load_state_dict(optim_state)
                # 恢复 Scheduler
                #if 'scheduler_state_dict' in checkpoint:
                #    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                del optim_state
        except Exception as e:
            print(f"⚠️ Optimizer Load Failed: {e}")
            print("   -> Continuing with fresh optimizer (Safe).")

        # ====================================================
        # 恢复步数 (必须做)
        # ====================================================
        global_step = checkpoint['step']
        start_epoch = checkpoint.get('epoch', 0)

        # 🔥 最后彻底删除 checkpoint 引用并 GC
        del checkpoint
        del state_dict_ref
        gc.collect()
        torch.cuda.empty_cache()

        print(f"✅ Successfully resumed from Step {global_step}, Epoch {start_epoch}")
    else:
        print("🆕 Starting training from scratch.")

    # 3. 编译加速 (System 2 核心)
    print("⚡️ Compiling Cortex Brain Sub-modules (JIT)...")
    try:
        # 防止万一有不支持的算子导致整个训练崩溃，直接 fallback 到普通 Python
        torch._dynamo.config.suppress_errors = True
        model.thinking_block = torch.compile(model.thinking_block)
        print("✅ Cortex Brain successfully compiled!")
    except Exception as e:
        print(f"⚠️ Compile failed (ignorable): {e}")

    # ... (原有代码：给 Base Model 开) ...
    #print("🛡️ Enabling Gradient Checkpointing for Base Model...") do not open this!!!!
    #model.base_model.gradient_checkpointing_enable(
    #    gradient_checkpointing_kwargs={"use_reentrant": False} # 👈 魔法开关！
    #)
    #model.base_model.config.use_cache = False

    model.train()

    is_first_batch = False
    
    # [MoE Stats] 统计缓冲区，用于梯度累积期间的汇总
    moe_buffer = {
        "indices": [],
    }

    if True: 
        # 🔥🔥🔥 [修正版] 显存预热 (Memory Warmup) 🔥🔥🔥
        print("\n🔥 Starting Memory Warmup (Dry Run with max_length)...")
        try:
            dummy_bs = CONFIG['batch_size'] 
            # 1. 造一个假的 8k 数据
            dummy_len = CONFIG['max_length']
            print(f"   -> Allocating dummy tensor: [{dummy_bs}, {dummy_len}]")

            # Input IDs
            dummy_input = torch.randint(
                0, 50000, (dummy_bs, dummy_len),
                device=device, dtype=torch.long
            )

            # Attention Mask (全 1)
            dummy_mask = torch.ones(
                (dummy_bs, dummy_len),
                device=device, dtype=torch.long
            )

            # Labels (为了计算 Loss，必须提供 Labels，通常与 input 一样)
            dummy_labels = dummy_input.clone()

            # 2. 假装跑一次前向传播 (Forward)
            print("   -> Running Dummy Forward...")
            # 修正警告：使用 torch.amp.autocast('cuda', ...)
            #with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            with torch.amp.autocast('cuda'):
                # 🔥🔥🔥 关键修正：必须传入 labels 和 attention_mask 🔥🔥🔥
                outputs = model(
                    input_ids=dummy_input,
                    attention_mask=dummy_mask,
                    labels=dummy_labels
                )

                # 确保拿到了 Loss
                if hasattr(outputs, "loss") and outputs.loss is not None:
                    loss = outputs.loss
                else:
                    # 兼容部分自定义模型的返回格式
                    loss = outputs[0] if isinstance(outputs, tuple) else outputs

            # 3. 假装跑一次反向传播 (Backward)
            print("   -> Running Dummy Backward...")
            if loss is None:
                raise RuntimeError("Model did not return loss! Check your forward() implementation.")

            loss.backward()

            # 4. 假装更新一次参数 (Optimizer Step)
            print("   -> Skipping Optimizer Step to prevent weight corruption...")
            model.zero_grad()
            optimizer.zero_grad()

            del dummy_input, dummy_mask, dummy_labels, outputs, loss
            print("✅ Memory Warmup Passed! VRAM is fully allocated and safe.")

        except RuntimeError as e:
            if "out of memory" in str(e):
                print("\n❌ [FATAL] VRAM OOM during Warmup!")
                print("   Conclusion: Your GPU absolutely CANNOT handle this length.")
                print("   Action: Reduce 'max_length' or enable 8-bit Optimizer immediately.")
                exit(1)
            else:
                raise e
        
    # 8. 训练主循环
    for epoch in range(start_epoch, CONFIG['epochs']):

        total_batches = len(ds) // CONFIG['batch_size']
        initial_batch = 0
 
        # [修正] Streaming 状态设置
        if hasattr(ds, 'set_epoch'):
            ds.set_epoch(epoch)

        # [修正] 断点续训使用 Dataset 级跳过
        if epoch == start_epoch and global_step > 0:
            samples_processed = global_step * CONFIG['grad_accum_steps'] * CONFIG['batch_size']
            #samples_processed= 20400
            initial_batch = global_step * CONFIG['grad_accum_steps'] 
            print(f"⏩ Resuming: Fast-forwarding dataset to {initial_batch} batch...")
            print(f"⏩ Resuming: Fast-forwarding dataset to {samples_processed} samples...")
            if hasattr(ds, 'set_skip_samples'):
                ds.set_skip_samples(samples_processed)
        else:
            if hasattr(ds, 'set_skip_samples'):
                ds.set_skip_samples(0)

        progress = tqdm(
            loader,
            desc=f"Epoch {epoch}",
            total=total_batches,
            initial=initial_batch  # <--- 关键修正：告诉 tqdm 我们是从这里开始的
        )

        accum_cortex = 0.0
        accum_koleo = 0.0
        accum_div = 0.0
        accum_latent_div = 0.0
        accum_base = 0.0

        for batch_idx, batch in enumerate(progress):
            #if batch_idx == 15:
            #    print("\n🔥 [ncu] Starting hardware profiling for Step 5...")
            #    torch.cuda.cudart().cudaProfilerStart()
            try:
                input_ids = batch['input_ids'].to(device).long()
                labels = batch['labels'].to(device).long()
                attention_mask = batch['attention_mask'].to(device)
                
                # [新增] 数据长度日志：帮助调试数据质量
                seq_len = input_ids.shape[1]
                valid_tokens = (labels != -100).sum().item()
                if batch_idx % 100 == 0:  # 每100个batch打印一次
                    tqdm.write(f"📊 Batch {batch_idx}: seq_len={seq_len}, valid_tokens={valid_tokens}")
                #print('elba lenth is ', input_ids.shape, labels.shape, attention_mask.shape)

                # --- Baseline Calculation (偶尔跑一次做对比) ---
                baseline_loss = 0.0
                if batch_idx % CONFIG['baseline_interval'] == 0:
                    with torch.no_grad():
                        base_out = model.base_model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                        baseline_loss = base_out.loss.item()
                        del base_out

                # --- Cortex Training ---
                #with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                with torch.amp.autocast('cuda'):
                    outputs, thought_embeds_vis, internal_thoughts = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        return_thoughts=True
                    )

                    div_loss = compute_diversity_loss(internal_thoughts)
                    latent_token_div_loss = compute_diversity_loss(thought_embeds_vis)
                    cortex_loss = outputs.loss
            
                    # 🔥 Compute KoLeo Loss
                    koleo_loss = koleo_loss_fn(thought_embeds_vis)
                    koleo_weight = CONFIG.get('koleo_loss_weight', 8.0)
     
                    total_loss = cortex_loss + koleo_weight * koleo_loss

                # [MoE Stats Collection] 收集本 Batch 的 MoE 统计数据
                # 这些数据已经在 Model 内部被 detach() 过了，可以直接转 CPU
                if hasattr(outputs, 'moe_indices'):
                    moe_buffer["indices"].extend(outputs.moe_indices.cpu().numpy().tolist())

                # 梯度累积
                loss = total_loss / CONFIG['grad_accum_steps']
                loss.backward()

                # 纯粹的累加，不要乘系数
                accum_cortex = 0.9 * accum_cortex + 0.1 * cortex_loss.item()
                accum_koleo  = 0.9 * accum_koleo  + 0.1 * koleo_loss.item()
                accum_div    = 0.9 * accum_div    + 0.1 * div_loss.item()
                accum_latent_div = 0.9 * accum_latent_div + 0.1 * latent_token_div_loss.item()
                if baseline_loss > 0:
                    accum_base = 0.9 * accum_base + 0.1 * baseline_loss

                # 更新权重
                # [修正] 增加 is_update_step 标记
                is_update_step = (batch_idx + 1) % CONFIG['grad_accum_steps'] == 0

                if is_update_step:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 128.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    gain = accum_base - accum_cortex
 
                    # --- [MoE Stats Analysis] ---
                    # 计算累积步内的统计信息
                    moe_log = {}
                    if len(moe_buffer["indices"]) > 0:
                        indices_arr = np.array(moe_buffer["indices"])
                        
                        # 1. 专家分布 (Counts)
                        # 0..N-1 是重型专家, N 是 System 1
                        total_samples = len(indices_arr)
                        num_experts = model_config.num_heavy_experts + 1
                        counts = np.bincount(indices_arr, minlength=num_experts)
                        
                        for i in range(model_config.num_heavy_experts):
                            moe_log[f"moe/exp_{i}_usage"] = counts[i] / total_samples
                        
                        sys1_idx = model_config.num_heavy_experts
                        moe_log[f"moe/sys1_usage"] = counts[sys1_idx] / total_samples

                        # 清空缓冲区
                        moe_buffer = {"indices": []}

                    # 合并 Log
                    wandb_log_dict = {
                        "loss/cortex": accum_cortex,
                        "loss/koleo": accum_koleo,
                        "loss/baseline": accum_base,
                        "loss/div": accum_div,
                        "loss/latent_token_div" : accum_latent_div,
                        "loss/gain": gain,
                        "lr": optimizer.param_groups[0]['lr'],
                        "grad_norm": grad_norm,
                        "step": global_step,
                        **moe_log # 解包 MoE 统计
                    }
                    wandb.log(wandb_log_dict)

                    progress.set_postfix({"Cortex": f"{accum_cortex:.3f}",
                                          "Grad": f"{grad_norm:.2f}",
                                          "KoLeo": f"{accum_koleo:.3f}",
                                          "Div_interal" : f"{accum_div:.3f}",
                                          "Div_latent_token" : f"{accum_latent_div:.3f}",
                                          "Gain": f"{gain:.3f}"})
                    if global_step % 100 == 0:
                        tqdm.write(str(progress))

                # --- 🛡️ 增强版 Debug 监控 (关键部分) ---
                save_interval = CONFIG.get('save_interval')
                # [修正] 增加 is_update_step 检查
                if (batch_idx + 1) % save_interval == 0:
                    save_checkpoint(model, optimizer, scheduler, global_step, epoch, CONFIG['output_dir'])

                debug_interval = CONFIG.get('debug_interval')
                # [修正] 增加 is_update_step 检查

                if ((batch_idx + 1) % debug_interval == 0) or is_first_batch:
                    is_first_batch = False
                    tqdm.write(f"\n{'='*20} 🔍 Debug Snapshot [Step {global_step}] {'='*20}")
                    try:
                        #torch.cuda.empty_cache()  #  epoch结束后清理碎片化显存
                        #print(f"Epoch {epoch} 显存清理完成，剩余显存：{torch.cuda.memory_reserved()/1024**3:.2f}GB")
                        with torch.no_grad():
                            # 采样第一条数据
                            test_in = input_ids[0:1]
                            test_mask = attention_mask[0:1]
                            test_labels = labels[0]

                            # 1. 文本解码
                            full_text = tokenizer.decode(test_in[0], skip_special_tokens=True)
                            tqdm.write(f"📝 Training instance:\n{full_text}")

                            # 2. 真实目标解码
                            valid_label_ids = test_labels[test_labels != -100]
                            if len(valid_label_ids) > 0:
                                target_text = tokenizer.decode(valid_label_ids, skip_special_tokens=True)
                                # 3. 手动高亮标签，防止它在控制台里隐身
                                target_text = target_text.replace("<|begin_of_thought|>", "\n🔥[THINK_START]\n")
                                target_text = target_text.replace("<|end_of_thought|>", "\n✅[THINK_END]\n")

                                tqdm.write(f"🎯 Ground Truth:\n{target_text}")
                            else:
                                tqdm.write("🎯 Ground Truth: [None / All Masked]")

			                # 🔥🔥🔥 3. Rollout Generation (完美 A/B Test + 强制唤醒版) 🔥🔥🔥
                            tqdm.write(f"📜 Rolling out (Generation A/B Test, comparing Base vs Cortex model):")
                            try:
                                # 1. 找到 User Prompt 的边界 (截断答案)
                                #prompt_mask_idx = (test_labels == -100)
                                #prompt_len = prompt_mask_idx.sum().item()

                                # 找到第一个不是 -100 的位置索引
                                valid_indices = (test_labels != -100).nonzero(as_tuple=True)[0]
                                
                                if len(valid_indices) > 0:
                                    prompt_len = valid_indices[0].item() # 取第一个有效 Label 的位置作为截断点
                                else:
                                    # 极少情况：全是被 Mask 的（比如全是 Prompt 或全是 Pad），这种数据应该跳过
                                    prompt_len = 0
                                
                                if prompt_len > 0:
                                    # 原始 Prompt (只包含 User 部分)
                                    raw_prompt_in = test_in[:, :prompt_len]
                                    raw_prompt_attn = test_mask[:, :prompt_len]

                                    ## ---------------------------------------------------------
                                    ## 🛠️ 构造最终输入 (Final Input Construction)
                                    ## ---------------------------------------------------------
                                    #suffix_text = "<|im_start|>assistant\n"
                                    #suffix_ids = tokenizer.encode(suffix_text, add_special_tokens=False)
                                    #suffix_tensor = torch.tensor([suffix_ids], device=device)
                                    #suffix_mask = torch.ones((1, len(suffix_ids)), device=device)

                                    ## 强制拼接 (不管之前有没有，先拼上看看效果，打印出来再说)
                                    #final_input_ids = torch.cat([raw_prompt_in, suffix_tensor], dim=1)
                                    #final_attn_mask = torch.cat([raw_prompt_attn, suffix_mask], dim=1)
                                    final_input_ids = raw_prompt_in
                                    final_attn_mask = raw_prompt_attn

                                    # ==========================================
                                    # 🅰️ Round A: Base Model (System 1 直觉模式)
                                    # ==========================================
                                    base_gen_ids = model.base_model.generate(
                                        input_ids=final_input_ids,       # <--- 就是上面打印的这个
                                        attention_mask=final_attn_mask,
                                        max_new_tokens=1500,
                                        do_sample=False,
                                        temperature=1.0,
                                        repetition_penalty=1.1,
                                        pad_token_id=tokenizer.pad_token_id,
                                        eos_token_id=tokenizer.eos_token_id
                                    )
                                    # 解码 (只解新生成的部分)
                                    # 注意：现在 input 变长了，所以切片位置也要变
                                    input_len = final_input_ids.shape[1]
                                    new_tokens_base = base_gen_ids[0][input_len:]
                                    base_text = tokenizer.decode(new_tokens_base, skip_special_tokens=False)
                                    tqdm.write(f"⚡ Base Says: \"{base_text.strip()}\"")

                                    # ==========================================
                                    # 🅱️ Round B: Cortex Model (System 2 思考模式)
                                    # ==========================================
                                    # (A) 构建 Embeddings
                                    # 注意：Base Embeddings 也要用拼接后的 final_input_ids 来生成

                                    model.eval() # 切换到 eval 模式，关闭 Dropout 和 Router 噪声
                                    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
                                        # 1. 闭眼思考：拿到 64 个 Thought Tokens
                                        outputs_debug, thought_vecs, internal_thoughts = model(
                                            input_ids=final_input_ids,
                                            attention_mask=final_attn_mask,
                                            labels=None,
                                            return_thoughts=True
                                        )
                                    visualize_dynamic_thoughts(model, tokenizer, thought_vecs)
                                    
                                    # 💥 核心重构：组装顺序改为 [User] -> [Latent]
                                    base_embeds = model.base_model.get_input_embeddings()(final_input_ids) # [1, L_prompt, D]
                                    combined_embeds = torch.cat([base_embeds, thought_vecs], dim=1)

                                    # 💥 核心重构：Mask 顺序同步改为 [User Mask] -> [Latent Mask]
                                    thought_mask = torch.ones((1, model.config.num_thought_tokens), dtype=torch.long, device=device)
                                    combined_mask = torch.cat([final_attn_mask, thought_mask], dim=1)

                                    # (C) 生成：利用 Base Model 的 generate 自动处理连续 KV Cache
                                    cortex_gen_ids = model.base_model.generate(
                                        inputs_embeds=combined_embeds,
                                        attention_mask=combined_mask,
                                        max_new_tokens=1500,
                                        do_sample=False,      
                                        temperature=1.0,
                                        repetition_penalty=1.1,
                                        pad_token_id=tokenizer.pad_token_id,
                                        eos_token_id=tokenizer.eos_token_id
                                    )
                                    
                                    # 💡 优雅之处：当使用 inputs_embeds 时，generate 默认只返回“新生成”的 IDs
                                    # 所以不需要再像 Base 测那样做 input_len 的切片了！直接解码！
                                    cortex_text = tokenizer.decode(cortex_gen_ids[0], skip_special_tokens=False)
                                    tqdm.write(f"🤖 Cortex Says: \"{cortex_text.strip()}\"")
                                    
                                    model.train() # 生成完毕后切回训练模式
                                else:
                                    tqdm.write("⚠️ Rollout Skipped: No prompt found (labels mismatch)")
                                
                            except Exception as e:
                                tqdm.write(f"⚠️ Gen Error: {e}")
                                if "out of memory" in str(e):
                                    print("⚠️ OOM detected! Skipping batch.")
                                    torch.cuda.empty_cache()
                                    raise e

                            # 5. 梯度健康检查
                            tqdm.write(f"🧠 Cortex Health Check:")

                            # Latents
                            if hasattr(model, 'latents'):
                                latent_norm = model.latents.data.norm().item()
                                latent_grad = model.latents.grad.norm().item() if model.latents.grad is not None else 0.0
                                tqdm.write(f"• Latents Norm: {latent_norm:.4f} | Grad: {latent_grad:.6f}")

                            # Projector
                            if hasattr(model, 'out_proj'):
                                w_norm = model.out_proj.weight.data.norm().item()
                                w_grad = model.out_proj.weight.grad.norm().item() if model.out_proj.weight.grad is not None else 0.0
                                tqdm.write(f"• Proj Weight Norm: {w_norm:.4f} | Grad: {w_grad:.6f}")
                            
                            # 2. Check System 1
                            if hasattr(model, 'system1_net'):
                                w = model.system1_net.weight
                                w_n = w.data.norm().item()
                                w_g = w.grad.norm().item() if w.grad is not None else 0.0
                                tqdm.write(f"  - Sys 1: W={w_n:.2f} | G={w_g:.5f}")

                            tqdm.write("="*60 + "\n")

                    except Exception as e:
                        tqdm.write(f"❌ Debug Error: {e}")
                        # traceback.print_exc()
                        if "out of memory" in str(e):
                            print("⚠️ OOM detected! Skipping batch.")
                            torch.cuda.empty_cache()
                            raise e
         

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("⚠️ OOM detected! Skipping batch.")
                    torch.cuda.empty_cache()
                    raise e
                else:
                    raise e
            #if batch_idx == 15:
            #    torch.cuda.cudart().cudaProfilerStop()
            #    print("✅ [ncu] Profiling finished! Exiting program to save time.")
            #    exit(0)  # 强制退出，别往下跑了，否则 ncu 会处理海量数据
        save_checkpoint(model, optimizer, scheduler, global_step, CONFIG['epochs'], CONFIG['output_dir'] + '/epoch/')
    save_checkpoint(model, optimizer, scheduler, global_step, CONFIG['epochs'], CONFIG['output_dir'] + '/epoch/')
    print("🎉 Training Done!")

if __name__ == "__main__":
    train()
