# src/config.py
import os

CONFIG = {
    # 基础模型路径 - 支持环境变量覆盖（容器内路径可能不同）
    "base_model": "path to base llm",
    
    # 路径配置
    "raw_data_cache": "./cortex_data_v2",
    "output_dir": "./checkpoints_v2",       
    "project_name": "cortex-deep-system2",
    
    # 数据处理 - 减小以适配 GB10 带宽
    "max_length": 2 * 1024,        # 8192 -> 2048: 省内存带宽
    "chunk_size": 8,
    
    # --- 🔥 Cortex 架构 (Small Cortex - 1.5B Base + 300M Cortex) ---
    "num_thought_tokens": 64,    # 1024 -> 64: O(N^2) attention cost
    "cortex_hidden_dim": 1536,   # 2048 -> 1536: 300M params for 1.5B base
    "encoder_layers": 12,        # 保持深度
    "encoder_heads": 12,         # 16 -> 12: 匹配 1536 dim (1536/12=128)
    
    # 训练参数
    # 参数变多了，Batch Size 保持 2 应该还能跑 (显存主要被 Base Model 占了)
    # 如果 OOM，就把 grad_accum_steps 翻倍，batch_size 改为 1
    "batch_size": 8,             
    "grad_accum_steps": 16,
    "lr": 1e-4,                  # 模型变大，学习率稍微调低一点点求稳
    "epochs": 1,
    "baseline_interval": 10,     
    "resume_from_checkpoint" : True,
    "koleo_loss_weight" : 8.0,

    "save_interval" : 1000,
    "debug_interval" : 501,
}
