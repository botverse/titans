import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_dim, ff_dim):
        super(FeedForwardNetwork, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, hidden_dim)
        self.activation = nn.SiLU()  # Changed to SiLU

    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))

class SlidingWindowAttention(nn.Module):
    def __init__(self, embed_dim, window_size):
        super(SlidingWindowAttention, self).__init__()
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.scale = embed_dim ** -0.5

    def forward(self, q, k, v):
        seq_len = q.size(1)
        output = []
        for i in range(seq_len):
            start = max(0, i - self.window_size)
            end = i + 1
            k_slice = k[:, start:end, :]
            v_slice = v[:, start:end, :]
            scores = torch.matmul(q[:, i:i+1, :], k_slice.transpose(-2, -1)) * self.scale
            attn_weights = torch.softmax(scores, dim=-1)
            attn_output = torch.matmul(attn_weights, v_slice)
            output.append(attn_output)
        return torch.cat(output, dim=1)

class GatedFusion(nn.Module):
    def __init__(self, hidden_dim):
        super(GatedFusion, self).__init__()
        self.gate = nn.Sigmoid()
        self.linear = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x1, x2):
        concat = torch.cat((x1, x2), dim=-1)
        gate = self.gate(self.linear(concat))
        return gate * x1 + (1 - gate) * x2

class NeuralMemory(nn.Module):
    def __init__(self, input_dim, memory_dim, memory_depth=2):
        super(NeuralMemory, self).__init__()
        self.memory_dim = memory_dim
        self.input_proj = nn.Linear(input_dim, memory_dim)
        self.memory_layers = nn.ModuleList(
            [nn.Linear(memory_dim, memory_dim) for _ in range(memory_depth)]
        )
        self.output_proj = nn.Linear(memory_dim, input_dim)
        self.forgetting_gate = nn.Parameter(torch.ones(1))
        self.learning_rate = nn.Parameter(torch.ones(1))

    def forward(self, memory, key, value):
        batch_size, seq_len, _ = key.shape
        key = F.normalize(key.view(batch_size * seq_len, -1), p=2, dim=-1)  # L2 normalization
        value = F.normalize(value.view(batch_size * seq_len, -1), p=2, dim=-1)

        key = self.input_proj(key)
        value_proj = self.input_proj(value)

        for layer in self.memory_layers:
            key = F.silu(layer(key))  # SiLU activation

        prediction = memory(key)
        loss = F.mse_loss(prediction, value_proj)
        surprise = torch.autograd.grad(loss, memory.parameters(), retain_graph=True)[0]

        momentum = self.learning_rate * surprise
        updated_memory = (1 - self.forgetting_gate) * memory.weight + momentum
        memory.weight.data.copy_(updated_memory)

        memory_output = memory(key)
        output = self.output_proj(memory_output)
        return output.view(batch_size, seq_len, -1)

class PersistentMemory(nn.Module):
    def __init__(self, memory_dim, num_slots):
        super(PersistentMemory, self).__init__()
        self.memory = nn.Parameter(torch.randn(num_slots, memory_dim))

    def forward(self, batch_size):
        return self.memory.unsqueeze(0).expand(batch_size, -1, -1)

class MemoryBlock(nn.Module):
    def __init__(self, hidden_dim, memory_dim, memory_depth=2, ff_dim=2048, window_size=16):
        super(MemoryBlock, self).__init__()
        self.short_term_memory = SlidingWindowAttention(hidden_dim, window_size)
        self.long_term_memory = NeuralMemory(hidden_dim, memory_dim, memory_depth)
        self.persistent_memory = PersistentMemory(memory_dim, num_slots=10)
        self.feed_forward = FeedForwardNetwork(hidden_dim, ff_dim)
        self.gated_fusion = GatedFusion(hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(self, x, memory):
        persistent_memory = self.persistent_memory(x.size(0))
        k = torch.cat((x, persistent_memory), dim=1)
        v = k.clone()
        attn_output = self.short_term_memory(x, k, v)
        x = self.norm1(x + attn_output)

        memory_output = self.long_term_memory(memory, k, v)
        combined = self.gated_fusion(x, memory_output)
        x = self.norm2(x + combined)

        ff_output = self.feed_forward(x)
        x = self.norm3(x + ff_output)
        return x

class TitansModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, memory_dim, memory_depth=2, ff_dim=2048, num_blocks=4):
        super(TitansModel, self).__init__()
        self.blocks = nn.ModuleList([
            MemoryBlock(hidden_dim, memory_dim, memory_depth, ff_dim)
            for _ in range(num_blocks)
        ])
        self.query_proj = nn.Linear(input_dim, hidden_dim)
        self.memory = nn.Linear(memory_dim, memory_dim)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        x = self.query_proj(x)
        memory = self.memory
        for block in self.blocks:
            x = block(x, memory)
        output = self.output_layer(x)
        return output
