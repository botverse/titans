from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaModel, LlamaForCausalLM
from models.llama_titans import MACModule

class MACLlamaForCausalLM(LlamaForCausalLM):
    """
    A vLLM-compatible version of the MACTransformer model
    """
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        
        # Initialize the MAC module with proper device placement
        self.mac_module = MACModule(
            dim=config.hidden_size,
            num_persistent=config.mac_module_config.get('num_persistent', 16),
            memory_size=config.mac_module_config.get('memory_size', 1024),
            alpha=config.mac_module_config.get('alpha', 0.1)
        )
        
        # Ensure memory buffers are initialized
        with torch.no_grad():
            self.mac_module.long_term_memory.zero_()  # Clear initial memory
        
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = True,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_mac: bool = True,
    ):
        """Forward pass with MAC module integration"""
        # Memory optimization: Use inference mode and mixed precision
        with torch.inference_mode(), torch.cuda.amp.autocast():
            if not use_mac:
                return super().forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    labels=labels,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict
                )
            
            # Get batch size and device
            batch_size = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            
            # Get input embeddings with memory optimization
            if inputs_embeds is None:
                inputs_embeds = self.model.embed_tokens(input_ids)  # (B, T, C)
            
            # Memory-efficient memory operations
            with torch.no_grad():  # No gradients needed for memory operations
                h_mem = self.mac_module.retrieve(inputs_embeds).unsqueeze(1)  # (B, 1, C)
                p_mem = self.mac_module.persistent_memory.unsqueeze(0).expand(batch_size, -1, -1)  # (B, P, C)
            
            # Concatenate with reduced memory usage
            combined_embeds = torch.cat([p_mem, h_mem, inputs_embeds], dim=1)  # (B, P+1+T, C)
            
            # Create position IDs efficiently
            seq_len = combined_embeds.shape[1]
            if position_ids is None:
                position_ids = torch.arange(seq_len, device=device).expand(batch_size, -1)  # (B, S)
            
            # Create attention mask efficiently
            if attention_mask is not None:
                # Convert to float32 for numerical stability
                attention_mask = attention_mask.to(dtype=torch.float32)
                
                # Create memory prefix mask
                prefix_mask = torch.ones(
                    (batch_size, p_mem.shape[1] + 1),
                    device=device,
                    dtype=torch.float32
                )
                
                # Combine masks
                attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
                
                # Create 4D attention mask directly
                attention_mask = attention_mask.view(batch_size, 1, 1, -1)
                
                # Apply causal mask
                causal_mask = torch.triu(
                    torch.ones((seq_len, seq_len), device=device, dtype=torch.bool),
                    diagonal=1
                )
                attention_mask = attention_mask.masked_fill(causal_mask, float("-inf"))
            
            # Forward pass with memory optimizations
            outputs = super().forward(
                input_ids=None,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=combined_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=False,  # Disable to save memory
                output_hidden_states=True,
                return_dict=True
            )
            
            # Update memory efficiently
            if outputs.hidden_states:
                with torch.no_grad():
                    self.mac_module.update(
                        outputs.hidden_states[-1][:, p_mem.shape[1]+1:, :].detach()
                    )
            
            # Extract logits efficiently
            logits = outputs.logits[:, p_mem.shape[1]+1:, :]
            outputs.logits = logits
            
            return outputs

    def generate(
        self,
        input_ids,
        attention_mask=None,
        max_new_tokens=20,
        temperature=0.7,
        top_p=0.95,
        do_sample=True,
        use_mac=True,
        use_cache=True,  # Enable KV cache by default
        pad_token_id=None,  # Add these parameters
        eos_token_id=None,  # Add these parameters
        repetition_penalty=1.0,  # Add this parameter
        **kwargs
    ):
        """Custom generation method supporting MAC functionality"""
        # Ensure model is in eval mode
        self.eval()
        
        # Store original batch size and sequence length
        batch_size, seq_length = input_ids.shape
        device = input_ids.device
        
        # Initialize sequence storage
        generated_tokens = input_ids.clone()
        
        # Set default token IDs if not provided
        if pad_token_id is None:
            pad_token_id = self.config.pad_token_id
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id
        
        # Memory-efficient settings
        with torch.inference_mode(), torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            # Generation loop
            for _ in range(max_new_tokens):
                # Forward pass with MAC
                outputs = self.forward(
                    input_ids=generated_tokens[:, -1024:],  # Limit context window
                    attention_mask=attention_mask,
                    use_cache=use_cache,
                    use_mac=use_mac,
                    output_attentions=False,
                    output_hidden_states=True,
                    return_dict=True
                )
                
                # Get logits for next token
                next_token_logits = outputs.logits[:, -1, :]
                
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
                
                # Update attention mask if needed
                if attention_mask is not None:
                    attention_mask = torch.cat([
                        attention_mask,
                        torch.ones((batch_size, 1), device=device, dtype=attention_mask.dtype)
                    ], dim=1)
                
                # Check for EOS or max length
                if (next_token == eos_token_id).all() or generated_tokens.shape[1] >= self.config.max_position_embeddings:
                    break
        
        return generated_tokens 