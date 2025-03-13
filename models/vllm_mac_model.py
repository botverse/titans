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
        
        # Initialize the MAC module
        self.mac_module = MACModule(
            dim=config.hidden_size,
            num_persistent=16,
            memory_size=1024,
            alpha=0.1
        )
        
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
        if not use_mac:
            # Standard LLaMA forward pass
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
        
        # MAC-enabled forward pass
        batch_size = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
        
        # Get input embeddings
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)  # (B, T, C)
        
        # Get memory embeddings
        h_mem = self.mac_module.retrieve(inputs_embeds).unsqueeze(1)  # (B, 1, C)
        p_mem = self.mac_module.persistent_memory.unsqueeze(0).expand(batch_size, -1, -1)  # (B, P, C)
        
        # Concatenate memory with input embeddings
        combined_embeds = torch.cat([p_mem, h_mem, inputs_embeds], dim=1)  # (B, P+1+T, C)
        
        # Create new position IDs
        seq_len = combined_embeds.shape[1]
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=combined_embeds.device).expand(batch_size, -1)  # (B, S)
        
        # Create attention mask for combined sequence
        if attention_mask is not None:
            # Create prefix mask (all ones)
            prefix_mask = torch.ones(batch_size, p_mem.shape[1] + 1, device=attention_mask.device, dtype=attention_mask.dtype)
            # Concatenate with original attention mask
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        else:
            # Use all ones if no mask provided
            attention_mask = torch.ones(batch_size, seq_len, device=combined_embeds.device, dtype=torch.bool)
            
        # Call model with combined embeddings
        outputs = super().forward(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=combined_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,  # Need hidden states for MAC update
            return_dict=True
        )
        
        # Update MAC module's memory
        if outputs.hidden_states:
            self.mac_module.update(outputs.hidden_states[-1][:, p_mem.shape[1]+1:, :].detach())
        
        # Extract the logits for the actual input sequence (excluding memory tokens)
        logits = outputs.logits[:, p_mem.shape[1]+1:, :]  # (B, T, V)
        
        # Adjust the outputs to match the input sequence length
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
        **kwargs
    ):
        """Custom generation method supporting MAC functionality"""
        # Store original batch size and sequence length
        batch_size = input_ids.shape[0]  # (B)
        seq_length = input_ids.shape[1]  # (T)
        
        # Create position IDs if not provided
        device = input_ids.device
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_length, dtype=torch.bool, device=device)
        
        past_key_values = None
        all_tokens = input_ids.clone()
        
        # Get initial embeddings - we'll update memory from these
        initial_inputs_embeds = self.model.embed_tokens(input_ids)  # (B, T, C)
        
        # Process input_ids to get MAC context-enhanced representation
        with torch.no_grad():
            self.mac_module.retrieve(initial_inputs_embeds)  # Retrieve once to set up memory
            
            # Generation loop
            for _ in range(max_new_tokens):
                # Forward pass with MAC
                outputs = self.forward(
                    input_ids=all_tokens,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_attentions=False,
                    output_hidden_states=True,
                    return_dict=True,
                    use_mac=use_mac
                )
                
                # Get next token logits (last token only)
                next_token_logits = outputs.logits[:, -1, :]  # (B, V)
                
                # Apply temperature scaling
                if temperature > 0:
                    next_token_logits = next_token_logits / temperature
                
                # Apply top_p sampling
                if do_sample:
                    # Apply top_p
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                        
                        # Remove tokens with cumulative probability above the threshold
                        sorted_indices_to_remove = cumulative_probs > top_p
                        # Shift the indices to the right to keep also the first token above the threshold
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        
                        # Scatter sorted tensors to original indexing
                        indices_to_remove = sorted_indices_to_remove.scatter(
                            1, sorted_indices, sorted_indices_to_remove
                        )
                        next_token_logits = next_token_logits.masked_fill(indices_to_remove, float('-inf'))
                    
                    # Sample from the filtered distribution
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    # Greedy decoding
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append the generated token to the sequence
                all_tokens = torch.cat([all_tokens, next_token], dim=1)
                
                # Check for EOS token
                if (next_token == self.config.eos_token_id).all():
                    break
                
                # Update the past key values
                past_key_values = outputs.past_key_values
        
        return all_tokens 