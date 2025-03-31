import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import LlamaForCausalLM, LlamaConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional, Tuple, Union

class MACModuleWithGradientUpdate(nn.Module):
    def __init__(self, dim: int, num_persistent: int = 16, memory_size: int = 1024, num_retrieved: int = 1):
        """
        MAC module where dynamic memory is a learnable parameter updated by task gradients.
        """
        super().__init__()
        self.dim = dim
        self.num_persistent = num_persistent
        self.memory_size = memory_size
        self.num_retrieved = num_retrieved # K

        # Persistent memory (learnable via standard backprop)
        self.persistent_memory = nn.Parameter(torch.randn(num_persistent, dim) / math.sqrt(dim)) # (P, C)

        # Dynamic memory bank (learnable via standard backprop using task gradients)
        self.dynamic_memory_bank = nn.Parameter(torch.randn(memory_size, dim) / math.sqrt(dim)) # (M, C)

        # Projection layer for memory retrieval query (learnable)
        self.mac_query = nn.Linear(dim, dim)
        # Optional: Layers for key/value projections if using attention for retrieval
        # self.key_proj = nn.Linear(dim, dim)
        # self.value_proj = nn.Linear(dim, dim)

    def retrieve(self, segment_embeds: torch.Tensor) -> torch.Tensor:
        """
        Retrieves relevant information from the dynamic memory bank.
        MUST be differentiable w.r.t. self.dynamic_memory_bank.

        Args:
            segment_embeds (torch.Tensor): Embeddings of the current input segment. Shape: (B, T, C)

        Returns:
            torch.Tensor: Retrieved memory context. Shape: (B, K, C)
        """
        # Calculate mean representation for the segment query
        segment_mean = segment_embeds.mean(dim=1) # (B, C)

        # Project segment mean to query space
        q = self.mac_query(segment_mean) # (B, C)

        # --- Differentiable Retrieval ---
        # Example: Simple attention-based retrieval (ensure no torch.no_grad())
        # Project memory bank to keys (optional, could use bank directly)
        # mem_keys = self.key_proj(self.dynamic_memory_bank) # (M, C)
        mem_keys = self.dynamic_memory_bank # (M, C)

        # Calculate attention scores with temperature scaling
        attn_scores = torch.matmul(q, mem_keys.T) # (B, M)
        attn_scores = attn_scores / math.sqrt(self.dim)

        # Apply softmax to get attention weights
        # Use top-k for retrieval if desired, otherwise weighted sum
        # Here, we retrieve top K slots based on attention scores
        top_k_scores, top_k_indices = torch.topk(attn_scores, self.num_retrieved, dim=-1) # (B, K)
        attn_weights = torch.softmax(top_k_scores, dim=-1) # (B, K) - Softmax over the top K scores

        # Retrieve memory via weighted sum of the TOP K slots from the dynamic_memory_bank
        # Gather the top K memory vectors: shape (B, K, C)
        # Need to handle batch indexing carefully. Expand indices for gather.
        batch_indices = torch.arange(q.shape[0]).unsqueeze(-1).expand(-1, self.num_retrieved) # (B, K)
        retrieved_vectors = self.dynamic_memory_bank[top_k_indices] # This might need gather for batch-specific indices if not directly indexable like this.
                                                                    # Let's assume simple indexing works for now, or use gather.
                                                                    # A simpler differentiable approach: Weighted sum over *all* memory slots
                                                                    # attn_weights_all = torch.softmax(attn_scores, dim=-1) # (B, M)
                                                                    # retrieved_mem_sum = torch.matmul(attn_weights_all, self.dynamic_memory_bank) # (B, C)
                                                                    # return retrieved_mem_sum.unsqueeze(1) # Shape (B, 1, C) if K=1

        # Weighted sum of the retrieved top-K vectors
        # attn_weights shape (B, K), retrieved_vectors shape (B, K, C)
        # We want output (B, K, C) where each of the K slots is a weighted combination?
        # Or just a single vector (B, 1, C)? Let's stick to K vectors for now.
        # If we want K distinct retrieved vectors weighted by their attention:
        # retrieved_mem = attn_weights.unsqueeze(-1) * retrieved_vectors # (B, K, C) - This might not be standard attention retrieval

        # Let's simplify: Retrieve K slots, take their mean or weighted mean based on top_k_scores
        # Weighted mean:
        retrieved_mem = torch.sum(attn_weights.unsqueeze(-1) * retrieved_vectors, dim=1, keepdim=True) # (B, 1, C) - If K=1 effectively
        # If we want K separate slots, maybe just return retrieved_vectors directly?
        # return retrieved_vectors # (B, K, C)

        # Let's return a single weighted vector for simplicity (K=1 case essentially)
        return retrieved_mem # Shape (B, 1, C)


class MACTransformerWithGradientUpdate(nn.Module):
    def __init__(self, config: LlamaConfig, mac_module: MACModuleWithGradientUpdate):
        super().__init__()
        self.llama = LlamaForCausalLM(config)
        self.mac_module = mac_module
        self.config = config

    def forward(
        self,
        tokens: torch.Tensor,
        use_mac: bool = True,
        attention_mask: Optional[torch.Tensor] = None, # Allow passing original mask
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is not None:
             raise ValueError("Cannot provide both tokens and inputs_embeds.")
        if not use_mac:
            # Standard Llama forward pass
            return self.llama(
                input_ids=tokens, attention_mask=attention_mask, position_ids=position_ids,
                past_key_values=past_key_values, inputs_embeds=inputs_embeds, labels=labels,
                use_cache=use_cache, output_attentions=output_attentions,
                output_hidden_states=output_hidden_states, return_dict=return_dict,
            )

        # --- MAC Path ---
        bsz, seqlen = tokens.shape # (B, T)
        device = tokens.device

        # 1. Embed Input Tokens
        inputs_embeds = self.llama.model.embed_tokens(tokens) # (B, T, C)

        # 2. Memory Operations (Differentiable Retrieval)
        retrieved_mem = self.mac_module.retrieve(inputs_embeds) # (B, K, C) K=num_retrieved
        p_mem = self.mac_module.persistent_memory.unsqueeze(0).expand(bsz, -1, -1) # (B, P, C) P=num_persistent

        # 3. Concatenate Memory and Input Embeddings
        combined_embeds = torch.cat([p_mem, retrieved_mem, inputs_embeds], dim=1) # (B, P + K + T, C)
        combined_seq_len = combined_embeds.shape[1]

        # 4. Create Position IDs for the Combined Sequence
        # If original position_ids are provided, offset them. Otherwise create fresh ones.
        if position_ids is None:
             mac_position_ids = torch.arange(combined_seq_len, device=device).unsqueeze(0).expand(bsz, -1) # (B, P+K+T)
        else:
             # Offset provided position_ids
             mem_len = p_mem.shape[1] + retrieved_mem.shape[1] # P + K
             mem_pos_ids = torch.arange(mem_len, device=device).unsqueeze(0).expand(bsz, -1) # (B, P+K)
             shifted_input_pos_ids = position_ids + mem_len # Shift original ids
             mac_position_ids = torch.cat([mem_pos_ids, shifted_input_pos_ids], dim=1) # (B, P+K+T)


        # 5. Create Attention Mask for Combined Sequence
        # Let Llama handle causal masking by passing None if no specific masking needed beyond causal
        mac_attention_mask = None
        # If original attention_mask is provided, prepend 1s for memory tokens
        if attention_mask is not None:
            mem_len = p_mem.shape[1] + retrieved_mem.shape[1] # P + K
            mem_mask = torch.ones((bsz, mem_len), dtype=attention_mask.dtype, device=device) # (B, P+K)
            mac_attention_mask = torch.cat([mem_mask, attention_mask], dim=1) # (B, P+K+T)


        # 6. Forward Pass through Llama
        outputs = self.llama(
            input_ids=None,
            attention_mask=mac_attention_mask, # Pass combined mask or None
            position_ids=mac_position_ids,
            past_key_values=past_key_values,
            inputs_embeds=combined_embeds,
            labels=None, # Handle labels below
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        # 7. Adjust Outputs (Logits and Loss)
        mem_len = p_mem.shape[1] + retrieved_mem.shape[1] # P + K
        final_logits = outputs.logits[:, mem_len:, :] # (B, T, V) - Slice to get logits for original tokens

        loss = None
        if labels is not None:
            # Shift logits and labels for standard causal LM loss calculation
            shift_logits = final_logits[..., :-1, :].contiguous() # (B, T-1, V)
            shift_labels = labels[..., 1:].contiguous() # (B, T-1)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            # Crucially, calling loss.backward() later will compute gradients for
            # self.mac_module.dynamic_memory_bank and self.mac_module.persistent_memory

        # Reconstruct output object
        if not return_dict:
             output_tuple = (final_logits,) + outputs[1:]
             return (loss,) + output_tuple if loss is not None else output_tuple

        return CausalLMOutputWithPast(
            loss=loss,
            logits=final_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states, # Contains states for the combined sequence
            attentions=outputs.attentions,
        )

    # Generation would need modification to handle the prepending and cache correctly,
    # similar to the challenges in the original MACTransformer.generate.
    # Using vLLM with custom ops or careful cache management is likely needed for efficiency.