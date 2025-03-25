import torch
import torch.nn as nn

class Memory(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Memory, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.mlp(x)

# Hyperparameters
input_dim = 64
hidden_dim = 128
output_dim = 64
lr = 0.01            # base learning rate (θ_t)
momentum_eta = 0.9   # momentum factor (η_t)

# Modules
memory = Memory(input_dim, hidden_dim, output_dim)

# Dummy inputs (key-value pairs)
x_t = torch.randn(1, input_dim)
W_K = torch.randn(input_dim, input_dim)
W_V = torch.randn(input_dim, output_dim)

# Initial states
S_t = [torch.zeros_like(p) for p in memory.parameters()]

# Implement zero_grad function
def zero_grad(model):
    for param in model.parameters():
        if param.grad is not None:
            param.grad.zero_()

# Single-step Update Example:
k_t = x_t @ W_K
v_t = x_t @ W_V

# Forward pass
pred = memory(k_t)

# Compute associative memory loss (MSE)
loss = (pred - v_t).pow(2).mean()

# Compute gradient (surprise)
zero_grad(memory)  # Use the custom zero_grad function
loss.backward()

# Gradient-based surprise mechanism with momentum
with torch.no_grad():
    for idx, param in enumerate(memory.parameters()):
        if param.grad is None:
            continue
        # Surprise calculation
        surprise = -lr * param.grad
        # Momentum update for surprise
        S_t[idx] = momentum_eta * S_t[idx] + surprise
        # Update parameters (memory weights) with surprise
        param += S_t[idx]

# The memory parameters are now updated with the surprise mechanism