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
        self.dim = dim
        self.persistent_memory = nn.Parameter(torch.randn(num_persistent, dim) / math.sqrt(dim))  # (num_persistent, C)
        self.register_buffer("long_term_memory", torch.zeros(memory_size, dim))  # (M, C)
        self.mac_query = nn.Linear(dim, dim)
        self.alpha = alpha
    
    def retrieve(self, segment: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            # Calculate mean representation for the segment
            segment_mean = segment.mean(dim=1)  # (B, C)
            
            # Project segment mean to query space
            q = self.mac_query(segment_mean)  # (B, C)
            
            # Calculate attention scores with temperature scaling
            attn_scores = torch.matmul(q, self.long_term_memory.T)  # (B, M)
            attn_scores = attn_scores / math.sqrt(self.dim)
            
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
        
        return outputs # (B, T, V)

    def generate(
        self,
        tokens,
        max_new_tokens=20,
        temperature=0.7,
        top_p=0.95,
        do_sample=True,
        use_mac=True,
        pad_token_id=None,
        eos_token_id=None,
        repetition_penalty=1.0,
        **kwargs
    ):
        """Custom generation method supporting MAC functionality"""
        # Ensure model is in eval mode
        self.eval()
        
        # Store original batch size and sequence length
        batch_size, seq_length = tokens.shape
        device = tokens.device
        
        # Initialize sequence storage
        generated_tokens = tokens.clone()
        
        # Set default token IDs if not provided
        if pad_token_id is None:
            pad_token_id = self.config.pad_token_id
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id
        
        # Memory-efficient generation
        with torch.inference_mode(), torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            # Generation loop
            for _ in range(max_new_tokens):
                # Forward pass with MAC
                outputs = self.forward(
                    tokens=generated_tokens[:, -seq_length:],  # Limit context window
                    use_mac=use_mac
                )
                num_persistent = self.mac_module.persistent_memory.shape[0]  # (P)
                outputs = outputs.logits[:, num_persistent+1:, :]
                
                # Get logits for next token
                next_token_logits = outputs[:, -1, :]
                
                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    for i in range(batch_size):
                        for previous_token in generated_tokens[i]:
                            next_token_logits[i, previous_token] /= repetition_penalty
                
                # Temperature scaling
                if temperature > 0:
                    next_token_logits = next_token_logits / temperature
                
                # Top-p sampling
                if do_sample and top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    # Create mask and apply
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits = next_token_logits.masked_fill(indices_to_remove, float('-inf'))
                
                # Sample or greedy selection
                if do_sample:
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append new token
                generated_tokens = torch.cat([generated_tokens, next_token], dim=1)
                
                # Check for EOS or max length
                if (next_token == eos_token_id).all() or generated_tokens.shape[1] >= self.config.max_position_embeddings:
                    break
        
        return generated_tokens