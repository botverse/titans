# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, Union

import torch
import torch.nn.functional as F
from torch import nn

# Import HF implementation
from transformers import LlamaForCausalLM, LlamaConfig
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000
    max_batch_size: int = 32
    max_seq_len: int = 2048

class MACModule(nn.Module):
    def __init__(self, dim: int, num_persistent: int = 16, memory_size: int = 1024, alpha: float = 0.1):
        super().__init__()
        self.persistent_memory = nn.Parameter(torch.randn(num_persistent, dim))  # (num_persistent, C)
        self.register_buffer("long_term_memory", torch.zeros(memory_size, dim))  # (M, C)
        self.mac_query = nn.Linear(dim, dim)
        self.alpha = alpha
    
    def retrieve(self, segment: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            # Calculate mean representation for the segment
            segment_mean = segment.mean(dim=1)  # (B, C)
            
            # Project segment mean to query space
            q = self.mac_query(segment_mean)  # (B, C)
            
            # Calculate attention scores
            attn_scores = torch.matmul(q, self.long_term_memory.T)  # (B, M)
            
            # Apply softmax to get attention weights
            attn_weights = torch.softmax(attn_scores, dim=-1)  # (B, M)
            
            # Retrieve memory via weighted sum
            h = torch.matmul(attn_weights, self.long_term_memory)  # (B, C)
        return h
    
    def update(self, segment: torch.Tensor):
        with torch.no_grad():
            # Compute segment mean
            new_info = segment.mean(dim=1)  # (B, C)
            
            # Create batch mean
            new_info = new_info.mean(dim=0, keepdim=True)  # (1, C)
            
            # Update memory with exponential moving average
            self.long_term_memory = (1 - self.alpha) * self.long_term_memory + self.alpha * new_info

class MACTransformer(nn.Module):
    """
    A wrapper around the HF LlamaForCausalLM that implements Memory as a Context (MAC).
    """
    def __init__(self, config: LlamaConfig, mac_module: MACModule):
        super().__init__()
        # Initialize the underlying LLaMA model
        self.llama = LlamaForCausalLM(config)
        self.mac_module = mac_module
        self.config = config
        
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory efficiency"""
        self.llama.gradient_checkpointing_enable()
        
    def forward(self, tokens: torch.Tensor, use_mac: bool = True):
        bsz, seqlen = tokens.shape  # (B, T)
        
        if not use_mac:
            # Standard forward with explicit 2D position_ids
            position_ids = torch.arange(seqlen, device=tokens.device)
            position_ids = position_ids.expand(bsz, -1)  # (B, T)
            return self.llama(input_ids=tokens, position_ids=position_ids).logits
        
        # MAC path remains mostly unchanged but simplified
        inputs_embeds = self.llama.model.embed_tokens(tokens)  # (B, T, C)
        
        # Memory operations
        h_mem = self.mac_module.retrieve(inputs_embeds).unsqueeze(1)  # (B, 1, C)
        p_mem = self.mac_module.persistent_memory.unsqueeze(0).expand(bsz, -1, -1)  # (B, P, C)
        
        # Concatenate memory components
        combined_embeds = torch.cat([p_mem, h_mem, inputs_embeds], dim=1)  # (B, P+1+T, C)
        seq_len = combined_embeds.shape[1]
        
        # Critical fix: Always create full 2D position_ids
        position_ids = torch.arange(seq_len, device=tokens.device).expand(bsz, -1)  # (B, S)
        
        outputs = self.llama(
            inputs_embeds=combined_embeds,
            position_ids=position_ids,
            attention_mask=torch.ones_like(position_ids, dtype=torch.bool)
        )
        
        # Memory update and logit extraction
        if outputs.hidden_states:
            self.mac_module.update(outputs.hidden_states[-1][:, p_mem.shape[1]+1:, :].detach())
        
        return outputs.logits[:, p_mem.shape[1]+1:, :]  # (B, T, V)