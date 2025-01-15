import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralMemory(nn.Module):
    def __init__(self, input_dim, memory_dim, memory_depth=2):
        super(NeuralMemory, self).__init__()
        self.memory_dim = memory_dim
        self.memory_layers = nn.ModuleList(
            [nn.Linear(input_dim if i == 0 else memory_dim, memory_dim) for i in range(memory_depth)]
        )
        self.forgetting_gate = nn.Parameter(torch.ones(1))
        self.learning_rate = nn.Parameter(torch.ones(1))

    def forward(self, memory, key, value):
        # Pass key through memory layers (deep memory)
        for layer in self.memory_layers:
            key = F.relu(layer(key))

        # Compute prediction and surprise
        prediction = memory(key)
        loss = F.mse_loss(prediction, value)
        surprise = torch.autograd.grad(loss, memory.parameters(), retain_graph=True)[0]

        # Adaptive update with momentum (surprise-based learning)
        momentum = self.learning_rate * surprise
        updated_memory = (1 - self.forgetting_gate) * memory.weight + momentum
        memory.weight.data.copy_(updated_memory)

        return memory(key)

class TitansModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, memory_dim, memory_depth=2):
        super(TitansModel, self).__init__()
        self.short_term_memory = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8)
        self.long_term_memory = NeuralMemory(input_dim, memory_dim, memory_depth)
        self.query_proj = nn.Linear(input_dim, memory_dim)
        self.key_proj = nn.Linear(input_dim, memory_dim)
        self.value_proj = nn.Linear(input_dim, memory_dim)
        self.output_layer = nn.Linear(memory_dim, input_dim)

    def forward(self, x):
        # Short-term memory via attention
        q = self.query_proj(x)
        k = self.key_proj(x)
        v = self.value_proj(x)
        attn_output, _ = self.short_term_memory(q, k, v)

        # Long-term memory update and retrieval
        memory_output = self.long_term_memory(self.key_proj, k, v)

        # Combine short-term and long-term memory
        combined = attn_output + memory_output

        # Output projection
        output = self.output_layer(combined)
        return output

# Example usage
batch_size, seq_len, input_dim = 16, 50, 128
hidden_dim, memory_dim, memory_depth = 256, 128, 2

model = TitansModel(input_dim, hidden_dim, memory_dim, memory_depth)
input_data = torch.randn(batch_size, seq_len, input_dim)

output = model(input_data)
print(output.shape)