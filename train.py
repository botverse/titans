import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import argparse
import os
import yaml
from datetime import datetime

from models.naive_model import TitansModel  # Assuming TitansModel is defined in model.py

def setup_run():
    # Create runs directory if it doesn't exist
    runs_dir = "runs"
    if not os.path.exists(runs_dir):
        os.makedirs(runs_dir)
    
    # Create a unique run directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(runs_dir, f"run_{timestamp}")
    os.makedirs(run_dir)
    
    # Save hyperparameters
    hparams = vars(args)
    hparams_path = os.path.join(run_dir, "hparams.yaml")
    with open(hparams_path, "w") as f:
        yaml.dump(hparams, f)
    
    return run_dir

# Argument parser for hyperparameters
parser = argparse.ArgumentParser(description="Train Titans Model on OpenWebText Dataset")
parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--seq_length', type=int, default=128, help='Sequence length for input')
parser.add_argument('--input_dim', type=int, default=768, help='Input dimension')
parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension')
parser.add_argument('--memory_dim', type=int, default=512, help='Memory dimension')
parser.add_argument('--memory_depth', type=int, default=2, help='Memory depth')
args = parser.parse_args()

# Setup run directory and save paths
run_dir = setup_run()
model_save_path = os.path.join(run_dir, "model.pth")

# Load OpenWebText dataset
dataset = load_dataset("Skylion007/openwebtext", split="train")

# Tokenizer (using GPT-2 tokenizer for consistency with OpenWebText)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Set padding token to be the end of sequence token

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=args.seq_length)

# Tokenize dataset with caching
cache_dir = os.path.join(run_dir, "cache")
os.makedirs(cache_dir, exist_ok=True)
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
    cache_file_name=os.path.join(cache_dir, "tokenized_dataset_cache.arrow"),
    num_proc=os.cpu_count()  # Enable multiprocessing for faster tokenization
)
tokenized_dataset.set_format(type="torch", columns=["input_ids"])

# DataLoader
train_loader = DataLoader(tokenized_dataset, batch_size=args.batch_size, shuffle=True)

# Initialize the Titans model
model = TitansModel(args.input_dim, args.hidden_dim, args.memory_dim, args.memory_depth)
model = model.cuda() if torch.cuda.is_available() else model

# Optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
criterion = nn.CrossEntropyLoss()

# Training loop
def train():
    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            input_ids = batch["input_ids"].cuda() if torch.cuda.is_available() else batch["input_ids"]

            optimizer.zero_grad()

            # Shift inputs for next token prediction
            inputs = input_ids[:, :-1]
            targets = input_ids[:, 1:]

            # Forward pass
            outputs = model(inputs)

            # Reshape outputs and targets for CrossEntropyLoss
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.reshape(-1))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

        # Save the model
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    train()