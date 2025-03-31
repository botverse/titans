import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import LlamaForCausalLM, LlamaConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.cache_utils import Cache
from typing import Optional, List, Tuple, Union
import os
from pathlib import Path

# Use the actual LlamaForCausalLM from the packaged 'models' dir if needed,
# but relying on the globally installed transformers version is usually fine.
# from .hf_llama.modeling_llama import LlamaForCausalLM, LlamaConfig

class InferenceMemoryWrapper(PreTrainedModel):
    # config_class = LlamaConfig # Keep if needed for saving config

    # --- REVERTED __init__ signature ---
    def __init__(self, llama_model: LlamaForCausalLM, memory_size: int = 4096, num_retrieved: int = 1, update_alpha: float = 0.1, surprise_momentum: float = 0.9, surprise_lr: float = 0.01):
        super().__init__(llama_model.config) # Use config from the passed model
        self.llama = llama_model # Store the pre-loaded model

        # --- Use passed parameters ---
        self.memory_size = memory_size
        self.num_retrieved = num_retrieved
        self.update_alpha = update_alpha
        self.surprise_momentum_eta = surprise_momentum
        self.surprise_lr_theta = surprise_lr
        self.dim = llama_model.config.hidden_size
        self._target_dtype = llama_model.dtype # Get dtype from the base model (should be float16)

        # --- Memory buffer is a Parameter ---
        # Create tensor directly with correct dtype on CPU initially
        init_buffer_data = torch.zeros(self.memory_size, self.dim, dtype=self._target_dtype)
        # Initialize in place
        nn.init.normal_(init_buffer_data, mean=0.0, std=1 / math.sqrt(self.dim))
        # Wrap in Parameter (Parameter itself doesn't change dtype)
        self.memory_buffer = nn.Parameter(init_buffer_data)


        # --- Surprise Update State ---
        # Create tensor directly with correct dtype on CPU initially
        init_surprise_state = torch.zeros_like(self.memory_buffer.data, dtype=self._target_dtype) # Use buffer's shape/dtype
        self.register_buffer("surprise_state", init_surprise_state)


        # --- Freeze the underlying Llama model ---
        for param in self.llama.parameters():
            param.requires_grad = False
        self.llama.eval() # Keep llama in eval mode

    # --- Keep existing methods (get_input_embeddings, set_input_embeddings, etc.) ---
    def get_input_embeddings(self):
        return self.llama.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.llama.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.llama.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.llama.set_output_embeddings(new_embeddings)

    # --- Differentiable Attention Retrieval ---
    def retrieve_memory(self, query_input: torch.Tensor) -> torch.Tensor:
        """
        Retrieves memory using differentiable attention based on query_input.
        Args:
            query_input (torch.Tensor): Query tensor. Shape (B, C).
        Returns:
            torch.Tensor: Retrieved memory embedding (weighted sum). Shape (B, 1, C)
        """
        # Ensure query is the correct dtype (should match memory buffer)
        q = query_input.to(self.memory_buffer.dtype) # Still check against buffer's actual dtype

        # Use memory_buffer directly as keys and values
        # self.memory_buffer should now consistently be self._target_dtype (float16)
        mem_keys = self.memory_buffer # (memory_size, C)
        mem_values = self.memory_buffer # (memory_size, C)

        # Matmul should now work as dtypes match
        attn_scores = torch.matmul(q, mem_keys.T) / math.sqrt(self.dim) # (B, memory_size)
        attn_weights = torch.softmax(attn_scores, dim=-1) # (B, memory_size)

        # Ensure retrieved mem is also the correct dtype before returning
        retrieved_mem = torch.matmul(attn_weights, mem_values) # (B, C)

        return retrieved_mem.unsqueeze(1) # (B, 1, C)

    # --- Surprise Update Application ---
    @torch.no_grad()
    def apply_surprise_update(self):
        """ Applies the TITANS-style surprise update rule using self.memory_buffer.grad """
        if self.memory_buffer.grad is None:
            return

        # Ensure surprise_state is on the same device and dtype
        self.surprise_state = self.surprise_state.to(device=self.memory_buffer.device, dtype=self.memory_buffer.dtype)

        # Grad should have the same dtype as the parameter
        surprise_update_val = -self.surprise_lr_theta * self.memory_buffer.grad.data
        self.surprise_state.mul_(self.surprise_momentum_eta).add_(surprise_update_val)

        self.memory_buffer.data.add_(self.surprise_state)
        self.memory_buffer.grad.zero_()


    # --- EMA Update (Alternative, No Gradients) ---
    @torch.no_grad()
    def update_memory_ema(self, new_context_embedding: torch.Tensor):
        """ Updates the memory buffer using EMA. """
        # Ensure update vector is the correct dtype
        update_vec_float = new_context_embedding.mean(dim=0, keepdim=True) if new_context_embedding.shape[0] > 1 else new_context_embedding # (1, C)
        update_vec = update_vec_float.to(self.memory_buffer.dtype)

        # Ensure buffer is on the correct device before update
        self.memory_buffer.data = self.memory_buffer.data.to(update_vec.device)
        self.memory_buffer.data.mul_(1 - self.update_alpha).add_(update_vec * self.update_alpha)


    # --- Forward Pass (Pass-through to Llama) ---
    # Overriding forward is needed if we want AutoModelForCausalLM(wrapper) to work directly
    # This now needs to call self.llama.forward
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs, # Pass any extra kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
         # Directly call the wrapped llama model's forward pass
         # Note: This basic forward doesn't include the memory prepending logic.
         # That logic is currently only in the custom generate method.
         # If you wanted to use model(input_ids) directly *with* memory,
         # you'd need to replicate the generate logic here.
        return self.llama(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

    # --- MODIFIED Generate Method with Inline Backward Pass ---
    # (Generate method remains largely the same as before, but ensure it uses self.llama correctly)
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 20,
        num_beams: int = 1,
        use_memory: bool = True,
        update_rule: str = 'ema',
        temperature: float = 0.7,
        top_p: float = 0.95,
        do_sample: bool = True,
        repetition_penalty: float = 1.0,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None, # Added attention_mask parameter
        **kwargs,
    ) -> torch.LongTensor:
        if num_beams != 1:
            raise NotImplementedError("Beam search not implemented.")
        if update_rule == 'surprise' and not use_memory:
            print("Warning: update_rule='surprise' requires use_memory=True.")
            update_rule = 'none'

        if update_rule == 'surprise':
             self.memory_buffer.requires_grad_(True)
        else:
             # Ensure no grads are computed if not needed
             # Note: Llama part is already frozen and in eval mode
             pass # No specific action needed if not surprise

        bsz, seq_len_start = input_ids.shape
        device = input_ids.device
        generated_ids = input_ids.clone()
        current_seq_len = seq_len_start
        # Determine the expected dtype from the buffer
        expected_dtype = self.memory_buffer.dtype # Use actual buffer dtype

        if eos_token_id is None: eos_token_id = self.config.eos_token_id
        if pad_token_id is None: pad_token_id = self.config.pad_token_id

        past_key_values = None # Initialize KV cache

        # Prepare initial attention mask if provided
        if attention_mask is None:
             attention_mask = torch.ones_like(input_ids)

        for step in range(max_new_tokens):
            # --- Prepare Inputs for this step ---
            # Use only the last token for generation if KV cache is active
            if past_key_values is not None:
                current_input_ids = generated_ids[:, -1:]
                # We need the hidden state/embedding of the *previous* token to query memory
                # Let's get the full embeddings first, then select the query basis
                # Use the full sequence length processed so far for embeddings
                full_embeds = self.llama.model.embed_tokens(generated_ids) # (B, T_cur, C)
                # Ensure query_basis has the expected dtype
                query_basis = full_embeds[:, -1, :].to(expected_dtype) # Query based on the last token generated *before* this step
            else:
                current_input_ids = generated_ids
                inputs_embeds_full = self.llama.model.embed_tokens(current_input_ids) # (B, T_cur, C)
                # Ensure query_basis has the expected dtype
                query_basis = inputs_embeds_full[:, -1, :].to(expected_dtype) # Query based on last token of the input prompt


            # --- Memory Retrieval ---
            retrieved_mem = None
            if use_memory:
                # query_basis should now match memory_buffer dtype
                retrieved_mem = self.retrieve_memory(query_basis) # (B, 1, C)

            # --- Combine Embeddings and Prepare Model Inputs ---
            # Manage attention mask and position IDs carefully
            current_mask = None
            mem_len = 0
            if retrieved_mem is not None:
                 retrieved_mem_casted = retrieved_mem.to(self.llama.dtype) # (B, 1, C_llama)
                 mem_len = retrieved_mem_casted.shape[1] # Should be 1

            if past_key_values is None: # First step
                inputs_embeds_full_casted = inputs_embeds_full.to(self.llama.dtype) # (B, T_cur, C_llama)
                if retrieved_mem is not None:
                    model_inputs_embeds = torch.cat([retrieved_mem_casted, inputs_embeds_full_casted], dim=1) # (B, 1 + T_cur, C)
                    # Create mask for memory + original input mask
                    mem_mask = torch.ones((bsz, mem_len), dtype=attention_mask.dtype, device=device)
                    current_mask = torch.cat([mem_mask, attention_mask], dim=1) # (B, 1 + T_cur)
                else:
                    model_inputs_embeds = inputs_embeds_full_casted # (B, T_cur, C)
                    current_mask = attention_mask # Use original mask

                effective_seq_len = model_inputs_embeds.shape[1]
                position_ids = torch.arange(effective_seq_len, device=device).unsqueeze(0) # (1, P+K+T)
                cur_input_ids_for_llama = None # Using embeds
            else: # Subsequent steps with KV cache
                current_input_embeds = self.llama.model.embed_tokens(current_input_ids).to(self.llama.dtype) # (B, 1, C_llama)
                if retrieved_mem is not None:
                     model_inputs_embeds = torch.cat([retrieved_mem_casted, current_input_embeds], dim=1) # (B, 1 + 1, C)
                     # Mask for memory + current token
                     current_mask = torch.ones((bsz, mem_len + 1), dtype=attention_mask.dtype, device=device) # (B, 1 + 1)
                else:
                     model_inputs_embeds = current_input_embeds # (B, 1, C)
                     # Mask for current token only
                     current_mask = torch.ones((bsz, 1), dtype=attention_mask.dtype, device=device) # (B, 1)

                # Position ID for the new token(s) relative to KV cache length + memory length
                # LlamaModel._update_causal_mask and cache handling expect position_ids to reflect the absolute position
                # cache_position (passed internally by generate if use_cache) handles this. We construct it manually here.
                # The position id for the *new token* is the current sequence length (including memory if prepended this step)
                past_len = past_key_values.get_seq_length() # Length stored in cache
                # The position_id should reflect where this new token/memory would be in the *full* sequence if no cache was used
                # Let's use current_seq_len derived from generated_ids, which doesn't include memory
                position_ids = torch.tensor([[current_seq_len -1 + i + mem_len for i in range(model_inputs_embeds.shape[1])]], device=device) # (1, M+1) or (1, 1)

                cur_input_ids_for_llama = None # Using embeds

            # --- Llama Forward Pass ---
            # Use KV caching if possible (update_rule != 'surprise')
            # We need past_key_values AND not be doing surprise update AND base model supports caching
            use_kv_cache_this_step = past_key_values is not None and update_rule != 'surprise' and self.llama.config.use_cache

            outputs = self.llama(
                input_ids=cur_input_ids_for_llama, # None if using embeds
                inputs_embeds=model_inputs_embeds,
                attention_mask=current_mask, # Pass the correctly shaped mask for this step
                position_ids=position_ids, # Pass adjusted position IDs
                past_key_values=past_key_values,
                use_cache=use_kv_cache_this_step,
                output_hidden_states=True, # Needed for query/target/update
                return_dict=True,
            )

            # --- Associative Loss Calculation (if surprise update) ---
            if update_rule == 'surprise' and use_memory and retrieved_mem is not None:
                 # Target: Final hidden state corresponding to the *last input token* before generation
                 # The index needs to account for the prepended memory.
                 # If mem_len=1, the target state corresponds to index -1 in the output sequence
                 target_repr = outputs.hidden_states[-1][:, -1, :].to(self.memory_buffer.dtype) # (B, C)

                 # pred_repr comes from retrieve_memory, should already match buffer dtype
                 pred_repr = retrieved_mem.squeeze(1) # (B, C)

                 assoc_loss = F.mse_loss(pred_repr, target_repr.detach())

                 if self.memory_buffer.grad is not None:
                      self.memory_buffer.grad.zero_()
                 assoc_loss.backward() # Compute grads for memory_buffer
                 self.apply_surprise_update() # Apply update and zero grad

            # --- Standard Generation Logic ---
            # Get logits for the very last position in the output sequence (corresponds to the token we just fed in)
            next_token_logits = outputs.logits[:, -1, :] # (B, V)

            # Update KV cache for next step
            if use_kv_cache_this_step:
                # The past_key_values returned by Llama should account for the memory prepended in this step
                past_key_values = outputs.past_key_values


            # Sampling (same as before)
            if repetition_penalty != 1.0:
                 # Simple loop for now:
                 for i in range(bsz):
                     # Penalize tokens in the *generated* sequence (excluding prompt if needed)
                     # Use generated_ids which tracks the full sequence
                     for token_id in generated_ids[i]:
                         # Avoid penalizing pad token if present
                         if token_id != pad_token_id:
                            next_token_logits[i, token_id] /= repetition_penalty

            if temperature > 0 and temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            if do_sample and top_p < 1.0:
                # Use Hugging Face's top_p implementation detail
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits = next_token_logits.masked_fill(indices_to_remove, float('-inf'))

            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # --- Update State ---
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            current_seq_len += 1
            # Update attention mask for the next iteration by appending 1
            attention_mask = torch.cat([attention_mask, torch.ones((bsz, 1), dtype=attention_mask.dtype, device=device)], dim=1)


            # --- EMA Memory Update ---
            if update_rule == 'ema' and use_memory and outputs.hidden_states is not None:
                 # Use hidden state corresponding to the newly generated token position (index -1)
                 # Cast state to buffer dtype before update
                 new_context_state = outputs.hidden_states[-1][:, -1, :].to(self.memory_buffer.dtype) # (B, C)
                 self.update_memory_ema(new_context_state.detach())

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        # self.eval() # Already in eval mode if llama is frozen

        return generated_ids


    # --- Save/Load ---
    # Keep the save_pretrained as is, it saves wrapper specific state.
    def save_pretrained(self, save_directory: Union[str, os.PathLike], **kwargs):
        """ Saves the wrapper's specific state (memory buffer, surprise state). """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        # Save the base model's config (important for PreTrainedModel compatibility)
        self.config.save_pretrained(save_directory)

        # Save the memory buffer parameter directly
        # Ensure saving in float32 for broader compatibility, can be cast back on load
        # Note: Saving the Parameter itself, not just its .data
        torch.save(self.memory_buffer.float(), save_directory / "memory_buffer.pt")
        # Save the surprise state buffer directly
        torch.save(self.surprise_state.float(), save_directory / "surprise_state.pt")

        print(f"InferenceMemoryWrapper state saved to {save_directory}")
        # Note: Base Llama model weights are assumed to be saved separately or loaded from source.

    # from_pretrained is complex with wrappers. For local testing/handler, load manually.
    # @classmethod
    # def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs):
    #      raise NotImplementedError(...)