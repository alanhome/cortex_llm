import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, PretrainedConfig

try:
    from liger_kernel.transformers import apply_liger_kernel_to_qwen2
    # 这一行必须在 AutoModelForCausalLM.from_pretrained 之前执行
    apply_liger_kernel_to_qwen2()
    print("🚀 Liger-Kernel successfully patched Qwen2 modules!")
except ImportError:
    print("⚠️ Liger-Kernel not found. Proceeding with standard HuggingFace modules.")

class CortexDeepConfig(PretrainedConfig):
    def __init__(
        self,
        base_model_path="TBD",
        num_thought_tokens=64,
        cortex_hidden_dim=2048,
        encoder_layers=12,
        encoder_heads=16,
        num_heavy_experts=4,   # [保留接口] 内部不再使用
        num_experts_per_tok=1, # [保留接口] 内部不再使用
        num_recurrent_steps=3, 
        **kwargs
    ):
        super().__init__(**kwargs)
        self.base_model_path = base_model_path
        self.num_thought_tokens = num_thought_tokens
        self.cortex_hidden_dim = cortex_hidden_dim
        self.encoder_layers = encoder_layers
        self.encoder_heads = encoder_heads
        self.num_heavy_experts = num_heavy_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.num_recurrent_steps = num_recurrent_steps 

class CortexDeepModel(nn.Module):
    def __init__(self, config: CortexDeepConfig):
        super().__init__()
        self.config = config

        # ======================================================
        # 🏗️ Architecture Overview (启动日志)
        # ======================================================
        print(f"\n{'='*20} Cortex Model Architecture {'='*20}")
        print(f"🧠 Mode:        Recurrent Dense Thinking (System 2 / Iterative Attention)")
        print(f"🏗️ Structure:   1 x Deep Transformer Block (Reused {config.num_recurrent_steps} times)")
        print(f"🎫 Tokens:      {config.num_thought_tokens} thought tokens")
        print(f"{'='*64}\n")
        print(f"Loading Base Model: {config.base_model_path}...")

        self.base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_path,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            #attn_implementation="sdpa" # 推荐使用 sdpa (Pytorch 2.0+) 或 flash_attention_2
            attn_implementation="flash_attention_2"
        )
        # 冻结 Base Model
        for param in self.base_model.parameters():
            param.requires_grad = False

        # 🔥🔥🔥 2. [关键修复] 开启输入层的梯度流 🔥🔥🔥
        self.base_model.enable_input_require_grads()

        self.base_hidden_dim = self.base_model.config.hidden_size

        # =================================================================
        # 🛡️ 预先计算并缓存 Padding ID (只算一次)
        # =================================================================
        pad_id = self.base_model.config.pad_token_id
        if pad_id is None:
            eos_id = getattr(self.base_model.config, "eos_token_id", None)
            if isinstance(eos_id, list) and len(eos_id) > 0:
                pad_id = eos_id[0]
            elif eos_id is not None:
                pad_id = eos_id
            else:
                pad_id = 151643  # Qwen 默认 <|endoftext|>
        self.pad_token_id = pad_id

        # --- System 2: Cortex Brain ---
        # 把 Base Model 的维度投影到 Cortex 维度
        self.in_proj = nn.Linear(self.base_hidden_dim, config.cortex_hidden_dim)

        # 思考种子 (Learnable Latents)
        self.latents = nn.Parameter(torch.empty(1, config.num_thought_tokens, config.cortex_hidden_dim))
        nn.init.orthogonal_(self.latents)
        self.latents.data.mul_(0.2) # 缩放一下幅度
        
        # 🔥 [关键修复] 恢复 Position Embeddings，否则无法感知顺序
        self.position_embeddings = nn.Parameter(torch.randn(1, config.num_thought_tokens, config.cortex_hidden_dim) * 0.2)

        # [必须加这个] 防止 Norm 爆炸
        #self.latent_norm = nn.LayerNorm(config.cortex_hidden_dim)

        # [新增] 步骤 Embedding，让模型知道当前是第几步思考
        self.step_embeddings = nn.Embedding(10, config.cortex_hidden_dim)
        
        # [新增] 循环内的稳定性归一化 (防止数值爆炸)
        self.loop_norm = nn.RMSNorm(config.cortex_hidden_dim, eps=1e-6)
        self.in_proj_norm = nn.RMSNorm(config.cortex_hidden_dim, eps=1e-6)

        # 🔥 [新增] Post-Norm：循环后的状态重置
        self.post_loop_norm = nn.RMSNorm(config.cortex_hidden_dim, eps=1e-6) 

        # 压缩器 (Cross Attention): 让种子去阅读题目
        self.compressor = nn.MultiheadAttention(
            embed_dim=config.cortex_hidden_dim, num_heads=config.encoder_heads, batch_first=True
        )
        self.compressor_norm =  nn.RMSNorm(config.cortex_hidden_dim, eps=1e-6)

        ## [新增] 整理层 (Shared Self-Attention) - 解决信息割裂问题
        #organizer_layer = nn.TransformerEncoderLayer(
        #    d_model=config.cortex_hidden_dim, nhead=config.encoder_heads,
        #    dim_feedforward=config.cortex_hidden_dim * 4, activation="gelu",
        #    batch_first=True, norm_first=True
        #)
        #self.organizer = nn.TransformerEncoder(organizer_layer, num_layers=2)

        # 推理器 (Deep Thinking): 这里发生真正的思考 (Dense 模式)
        # [修改] 不再是 ModuleList，而是单体 Transformer Encoder
        thinking_layer = nn.TransformerEncoderLayer(
            d_model=config.cortex_hidden_dim, nhead=config.encoder_heads,
            dim_feedforward=config.cortex_hidden_dim * 4, activation="gelu",
            batch_first=True, norm_first=True, dropout=0.0
        )
        self.thinking_block = nn.TransformerEncoder(thinking_layer, num_layers=config.encoder_layers)

        # 输出投影: 把思考向量变回 Base Model 能懂的维度
        self.out_proj = nn.Linear(config.cortex_hidden_dim, self.base_hidden_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, input_ids, attention_mask=None, labels=None, past_key_values=None, position_ids=None, return_thoughts=False, **kwargs):
        B, N = input_ids.shape
        K = self.config.num_thought_tokens
        device = input_ids.device
        pad_id = self.pad_token_id

        # =================================================================
        # 🚀 [提速核心] 自回归生成短路
        # =================================================================
        # 如果 KV Cache 已存在，说明是 model.generate() 吐出第 2 个字及以后的阶段
        # 直接跳过思考，按 1.5B 原生速度飞奔
        if past_key_values is not None:
            return self.base_model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **kwargs
            )

        # =================================================================
        # 1. 🔍 边界识别 (全程纯 GPU 向量化，0 Python 循环)
        # =================================================================
        if labels is not None:
            is_answer = (labels != -100)
            # 找到每个样本第一个 Answer Token 的索引
            prompt_lens = torch.where(
                is_answer.any(dim=1),
                is_answer.float().argmax(dim=1).to(torch.long),
                torch.tensor(N, device=device)
            )
            answer_lens = is_answer.long().sum(dim=1)
        else:
            # 预热/推理模式：全部视为 Prompt
            prompt_lens = torch.full((B,), N, device=device, dtype=torch.long)
            answer_lens = torch.zeros((B,), device=device, dtype=torch.long)

        # 防御性保护：确保长度合法
        max_p = max(1, prompt_lens.max().item())
        max_a = answer_lens.max().item()

        # =================================================================
        # 2. 🔥 Phase 1: 提取 User Prompt 并左对齐 (Left-Alignment)
        # 目的：让所有样本的 Prompt 都在 max_p 位置整齐结束，完美对齐 KV Cache
        # =================================================================
        p_range = torch.arange(max_p, device=device).unsqueeze(0)
        p_idx = p_range - (max_p - prompt_lens).unsqueeze(1)
        valid_p_mask = (p_idx >= 0)
        
        # 安全提取 Prompt 并用 pad_id 填充左侧空隙
        safe_p_ids = torch.gather(input_ids, 1, p_idx.clamp(min=0))
        safe_p_ids = torch.where(valid_p_mask, safe_p_ids, torch.tensor(pad_id, device=device))

        with torch.no_grad():
            base_outputs = self.base_model.model(
                input_ids=safe_p_ids,
                attention_mask=valid_p_mask.long(),
                return_dict=True,
                use_cache=True  # 👈 核心：拿到 User 阶段的 KV Cache
            )
            contextualized_features = base_outputs.last_hidden_state.detach()
            phase1_kv_cache = base_outputs.past_key_values
            del base_outputs

        # =================================================================
        # 3. 🧠 Phase 2: 深度思考 (Cortex Reasoning)
        # =================================================================
        cortex_in = self.in_proj(contextualized_features)
        cortex_in = self.in_proj_norm(cortex_in)
        
        # 告诉 64 个种子：不要看左边补的 Padding，只看真正的题目
        cortex_key_padding_mask = ~valid_p_mask 
        
        current_thoughts = self.latents.expand(B, -1, -1)

        for step in range(self.config.num_recurrent_steps):
            q_norm = self.compressor_norm(current_thoughts + self.position_embeddings)
            context_info, _ = self.compressor(
                query=q_norm,
                key=cortex_in,
                value=cortex_in,
                key_padding_mask=cortex_key_padding_mask
            )
            current_thoughts = current_thoughts + context_info
            step_embed = self.step_embeddings(torch.tensor(step, device=device))
            #current_thoughts = current_thoughts.to(torch.bfloat16)
            hidden_input = self.loop_norm(current_thoughts + step_embed.view(1, 1, -1) + self.position_embeddings)
            hidden_output = self.thinking_block(hidden_input)
            current_thoughts = current_thoughts + hidden_output

        current_thoughts = self.post_loop_norm(current_thoughts)
        thought_embeds = self.out_proj(current_thoughts)

        # =================================================================
        # 4. 🔥 Phase 3: 提取 Answer 并增量注入 (KV Cache Injection)
        # 组装结构: [Phase 1 Cache (隐含)] -> [Latent] -> [Answer]
        # =================================================================
        a_range = torch.arange(max_a, device=device).unsqueeze(0) if max_a > 0 else torch.empty((B, 0), device=device)
        a_idx = a_range + prompt_lens.unsqueeze(1)
        valid_a_mask = a_range < answer_lens.unsqueeze(1)

        # 组装 Embeddings: 此时只需要给基础模型提供 [Latent] + [Answer]
        if max_a > 0:
            safe_a_ids = torch.gather(input_ids, 1, a_idx.clamp(max=N-1))
            safe_a_ids = torch.where(valid_a_mask, safe_a_ids, torch.tensor(pad_id, device=device))
            answer_embeds = self.base_model.get_input_embeddings()(safe_a_ids)
            combined_embeds = torch.cat([thought_embeds, answer_embeds], dim=1)
        else:
            combined_embeds = thought_embeds

        # =================================================================
        # 🛰️ 终极修正：基于真实长度的动态 RoPE 偏移
        # =================================================================
        # 1. 生成一个纯粹的相对偏移量: [0, 1, 2, ..., K + max_a - 1]
        offset = torch.arange(0, K + max_a, device=device).unsqueeze(0).expand(B, -1)

        # 2. 🚀 让每个样本都从自己真实的 Prompt 尾部开始起跑！
        # prompt_lens 维度是 [B], offset 维度是 [B, K + max_a]
        # 广播相加后，每个样本的起点都是绝对正确的。
        combined_pos = prompt_lens.unsqueeze(1) + offset

        # 全量 Mask: [Prompt-Mask] + [Latent-Mask(全1)] + [Answer-Mask]
        thought_mask = torch.ones((B, K), device=device, dtype=torch.bool)
        combined_mask = torch.cat([valid_p_mask, thought_mask, valid_a_mask], dim=1).long()

        # Labels 组装: 长度等于 Phase 3 的增量长度 (K + max_a)
        combined_labels = None
        if labels is not None:
            if max_a > 0:
                safe_a_labels = torch.gather(labels, 1, a_idx.clamp(max=N-1))
                safe_a_labels = torch.where(valid_a_mask, safe_a_labels, torch.tensor(-100, device=device))
            else:
                safe_a_labels = torch.empty((B, 0), device=device, dtype=torch.long)
            
            thought_labels = torch.full((B, K), -100, device=device, dtype=torch.long)
            combined_labels = torch.cat([thought_labels, safe_a_labels], dim=1)

        # 🚀 发射！带有 KV Cache 的增量计算，速度极大提升
        outputs = self.base_model(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            position_ids=combined_pos,       # <--- 对齐 RoPE
            past_key_values=phase1_kv_cache, # <--- 注入 Prompt 缓存
            labels=combined_labels,
            return_dict=True,
            use_cache=False,
            **kwargs
        )

        outputs.aux_loss = 0.0

        if return_thoughts:
            # 返回三个值：outputs, 投影后的(用于可视化), 内部的(用于算多样性Loss)
            return outputs, thought_embeds, current_thoughts

        return outputs

