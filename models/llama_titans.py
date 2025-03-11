# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import fairscale.nn.model_parallel.initialize as fs_init
import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from torch import nn


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000

    max_batch_size: int = 32
    max_seq_len: int = 2048


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # Calculate variance more carefully - handle edge cases
        variance = x.pow(2).mean(-1, keepdim=True)
        
        # Add safety checks
        if torch.any(variance < 0):
            print("[WARNING] Negative variance detected in RMSNorm!")
            variance = torch.clamp(variance, min=self.eps)
            
        # Add epsilon before sqrt for numerical stability
        return x * torch.rsqrt(variance + self.eps)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], new_tokens_length=None):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # Repeat KV heads if necessary
        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        # Transpose for attention computation
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # Limit attention to new tokens if specified
        if new_tokens_length is not None:
            xq = xq[:, :, -new_tokens_length:]

        # Calculate attention scores and apply mask
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        # Simple and memory-efficient safety clamp
        scores = torch.clamp(scores, min=-20.0, max=20.0)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores + mask
        
        # Apply softmax (using float for stability)
        scores = F.softmax(scores.float(), dim=-1).type_as(x)
        
        # Apply attention
        output = torch.matmul(scores, xv)
        
        # Reshape output
        output = output.transpose(1, 2).contiguous().view(bsz, -1, self.n_local_heads * self.head_dim)
        return self.wo(output)


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        model_parallel_size = fs_init.get_model_parallel_world_size()
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wk = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wv = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wo = RowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
        )

        self.cache_k = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        ).cuda()
        self.cache_v = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        ).cuda()

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        new_tokens_length: Optional[int] = None
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # Reshape for attention computation
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # Apply rotary embeddings
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # Repeat KV heads if necessary
        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        # Transpose for attention computation
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # Limit attention to new tokens if specified
        if new_tokens_length is not None:
            xq = xq[:, :, -new_tokens_length:]

        # Calculate attention scores and apply mask
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        # Safety check for NaNs or extreme values in scores
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            print(f"[WARNING] NaNs or Infs detected in attention scores: shape={scores.shape}")
            scores = torch.nan_to_num(scores, nan=0.0, posinf=20.0, neginf=-20.0)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores + mask
        
        # Constrain scores to reasonable range for numerical stability
        scores = torch.clamp(scores, min=-50.0, max=50.0)
        
        # Apply softmax (use float for better numerical stability)
        scores = F.softmax(scores.float(), dim=-1).type_as(x)
        
        # Safety check for NaNs in attention weights
        if torch.isnan(scores).any():
            print(f"[WARNING] NaNs detected in attention weights after softmax: shape={scores.shape}")
            scores = torch.nan_to_num(scores, nan=1.0/scores.size(-1))
        
        # Apply attention
        output = torch.matmul(scores, xv)
        
        # Reshape output
        output = output.transpose(1, 2).contiguous().view(bsz, -1, self.n_local_heads * self.head_dim)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )
        self.w2 = RowParallelLinear(
            hidden_dim, dim, bias=False, input_is_parallel=True, init_method=lambda x: x
        )
        self.w3 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        new_tokens_length: Optional[int] = None,
    ):
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask, new_tokens_length=new_tokens_length)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = VocabParallelEmbedding(
            params.vocab_size, params.dim, init_method=lambda x: x
        )

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = ColumnParallelLinear(
            params.dim, params.vocab_size, bias=False, init_method=lambda x: x
        )

        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads,
            params.max_seq_len * 2,
            params.rope_theta,
        )

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)

            mask = torch.triu(mask, diagonal=1)

            # When performing key-value caching, we compute the attention scores
            # only for the new sequence. Thus, the matrix of scores is of size
            # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
            # j > cache_len + i, since row i corresponds to token cache_len + i.
            mask = torch.hstack(
                [torch.zeros((seqlen, start_pos), device=tokens.device), mask]
            ).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h).float()
        return output

# --- MAC Implementation ---
# The following additions implement the Memory as a Context (MAC) variant.

class MACModule(nn.Module):
    """
    Implements a simplified Memory as a Context (MAC) module.
    It maintains a persistent memory (learnable tokens) and a long-term memory buffer.
    Given the current segment, it retrieves a memory summary and concatenates:
      [Persistent Memory || Retrieved Memory || Current Segment]
    before passing it to the transformer.
    """
    def __init__(self, dim: int, num_persistent: int = 16, memory_size: int = 1024, alpha: float = 0.1):
        super().__init__()
        # Learned persistent tokens
        self.persistent_memory = nn.Parameter(torch.randn(num_persistent, dim))
        # Long-term memory buffer (initialized to zeros)
        self.register_buffer("long_term_memory", torch.zeros(memory_size, dim))
        self.alpha = alpha
        self.memory_size = memory_size
        # Query projection for memory retrieval
        self.mac_query = nn.Linear(dim, dim)
    
    def retrieve(self, segment: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            q = self.mac_query(segment.mean(dim=1))
            print(f"[DEBUG] MAC query shape: {q.shape}, any NaN: {torch.isnan(q).any()}")
            print(f"[DEBUG] segment.mean range: {segment.mean(dim=1).min().item()} to {segment.mean(dim=1).max().item()}")
            
            attn_scores = torch.matmul(q, self.long_term_memory.T)
            print(f"[DEBUG] Attn scores shape: {attn_scores.shape}, any NaN: {torch.isnan(attn_scores).any()}")
            print(f"[DEBUG] Attn scores range: {attn_scores.min().item()} to {attn_scores.max().item()}")
            
            attn_weights = torch.softmax(attn_scores, dim=-1)
            print(f"[DEBUG] Attn weights shape: {attn_weights.shape}, any NaN: {torch.isnan(attn_weights).any()}")
            
            h = torch.matmul(attn_weights, self.long_term_memory)
            print(f"[DEBUG] Retrieved memory shape: {h.shape}, any NaN: {torch.isnan(h).any()}")
        return h
    
    def update(self, segment: torch.Tensor):
        with torch.no_grad():
            # Compute segment mean - add safety checks
            new_info = segment.mean(dim=1)
            
            # Check if new_info has NaNs before using it
            if torch.isnan(new_info).any():
                print("[WARNING] NaNs detected in segment mean, skipping memory update")
                return
            
            # Check if mean is all zeros (potential issue)
            if torch.all(new_info == 0):
                print("[WARNING] All-zero segment mean detected, skipping memory update")
                return
            
            # Create batch mean by safely averaging across batch dimension
            if new_info.shape[0] > 0:  # Make sure batch dimension isn't empty
                new_info = new_info.mean(dim=0, keepdim=True)
                
                # Safety check memory before updating
                if torch.isnan(self.long_term_memory).any():
                    print("[WARNING] NaNs detected in long_term_memory before update, resetting")
                    self.long_term_memory.zero_()
                    
                # Update with stable interpolation
                self.long_term_memory = (1 - self.alpha) * self.long_term_memory + self.alpha * new_info.expand_as(self.long_term_memory)
                
                # Final safety check
                if torch.isnan(self.long_term_memory).any():
                    print("[WARNING] NaNs detected in memory after update, resetting")
                    self.long_term_memory.zero_()

class MACTransformer(Transformer):
    """
    A wrapper around the base Transformer that implements Memory as a Context (MAC).
    It retrieves long-term memory using a MACModule, concatenates it with persistent tokens and the current segment,
    then processes the augmented input through the transformer layers. Finally, it updates the memory.
    """
    def __init__(self, params: ModelArgs, mac_module: MACModule):
        super().__init__(params)
        self.mac_module = mac_module

    def forward(self, tokens: torch.Tensor, start_pos: int, use_mac: bool = True):
        """Forward pass with targeted debugging"""
        _bsz, seqlen = tokens.shape
        
        # Get initial embeddings for the current segment
        h = self.tok_embeddings(tokens)  # (B, T, C)
        
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        if use_mac:
            # Retrieve long-term memory summary from current segment embeddings
            h_mem = self.mac_module.retrieve(h)  # (B, C)
            h_mem = h_mem.unsqueeze(1)  # (B, 1, C)
            
            # Expand persistent memory tokens to batch dimension
            p_mem = self.mac_module.persistent_memory.unsqueeze(0).expand(_bsz, -1, -1)  # (B, num_persistent, C)
            
            # Concatenate: persistent tokens, retrieved memory, then current segment embeddings
            h = torch.cat([p_mem, h_mem, h], dim=1)  # (B, num_persistent + 1 + T, C)
            
            # The extra tokens (prefix) count
            extra = p_mem.shape[1] + h_mem.shape[1]
            extra_freqs = torch.ones((extra, freqs_cis.shape[-1]), dtype=freqs_cis.dtype, device=freqs_cis.device)
            
            # Prepend dummy rotary embeddings for the extra tokens
            freqs_cis = torch.cat([extra_freqs, freqs_cis], dim=0)
            
            # Set new start position to the number of extra tokens
            new_start_pos = extra
            # Limit new_tokens_length 
            new_tokens_length = min(seqlen, self.params.max_seq_len - extra)
            mask = None
        else:
            new_start_pos = start_pos
            new_tokens_length = None
            mask = None
            if seqlen > 1:
                mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)
                mask = torch.triu(mask, diagonal=1)
                mask = torch.hstack([torch.zeros((seqlen, start_pos), device=tokens.device), mask]).type_as(h)

        # Run through each transformer layer without excessive checks
        for i, layer in enumerate(self.layers):
            h = layer(h, new_start_pos, freqs_cis, mask, new_tokens_length=new_tokens_length)
        
        h = self.norm(h)
        output = self.output(h).float()

        if use_mac:
            # Update long-term memory using the original current segment embeddings
            original_segment = h[:, -seqlen:, :].detach()
            # Only check for NaNs here since this is a critical point
            if not torch.isnan(original_segment).any():
                self.mac_module.update(original_segment)
            output = output[:, -seqlen:, :]
        return output