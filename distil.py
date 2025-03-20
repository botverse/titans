import dotenv
dotenv.load_dotenv()

import os
import copy
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer, BatchEncoding, LlamaForCausalLM, LlamaConfig
from models.llama_titans import MACModule, MACTransformer, ModelArgs
from datasets import load_dataset
from tqdm import tqdm
import socket
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import fairscale.nn.model_parallel.initialize as fs_init
import wandb
import numpy as np
import json
from datetime import datetime
import yaml
import argparse

def is_datacenter():
    """Check if we're running in a datacenter environment"""
    hostname = socket.gethostname()
    return any(name in hostname.lower() for name in ['cluster', 'node', 'compute', 'gpu'])

def get_world_size():
    """Get number of GPUs to use based on environment"""
    if is_datacenter():
        return torch.cuda.device_count()
    else:
        return min(1, torch.cuda.device_count())

def setup_distributed():
    """Initialize distributed training and model parallel groups"""
    world_size = get_world_size()
    
    if world_size <= 1:
        # For single GPU, initialize process group first
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        dist.init_process_group(backend="nccl", rank=0, world_size=1)
        
        # Then initialize model parallel
        fs_init.initialize_model_parallel(1)
        return False
    
    # Multi-GPU mode
    dist.init_process_group(backend="nccl")
    
    # Initialize model parallel groups
    # Using 1 for model parallel size as we're doing data parallel training
    fs_init.initialize_model_parallel(1)
    
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    return True

def cleanup():
    """Cleanup distributed training resources"""
    # Cleanup model parallel groups first
    if hasattr(fs_init, "model_parallel_is_initialized") and fs_init.model_parallel_is_initialized():
        fs_init.destroy_model_parallel()
    
    # Then cleanup distributed process group
    if dist.is_initialized():
        dist.destroy_process_group()

# Log where the NaNs first appear
def debug_nans(tensor, name):
    tensor_np = tensor.detach().cpu().numpy()
    wandb.log({f"nan_check/{name}": np.isnan(tensor_np).any()})
    if np.isnan(tensor_np).any():
        wandb.log({f"nan_check/{name}_percentage": np.isnan(tensor_np).mean() * 100})

def find_latest_experiment():
    """Find the most recent experiment directory"""
    runs_dir = Path("runs")
    if not runs_dir.exists():
        return None
    
    # Find all experiment directories
    experiments = [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("distil_")]
    if not experiments:
        return None
    
    # Sort by creation time and return the latest
    return max(experiments, key=lambda x: x.stat().st_mtime)

def setup_experiment(args):
    """Setup experiment directory and save configurations"""
    if args.resume:
        # Find the latest experiment
        exp_dir = find_latest_experiment()
        if exp_dir is None:
            raise ValueError("No existing experiments found to resume from")
        print(f"Resuming from experiment: {exp_dir}")
        
        # Load existing config
        config_path = exp_dir / "config" / "config.yaml"
        if config_path.exists():
            with open(config_path, 'r') as f:
                existing_config = yaml.safe_load(f)
                # Update args with saved config while preserving new command line args
                for k, v in existing_config.items():
                    if not hasattr(args, k) or getattr(args, k) is None:
                        setattr(args, k, v)
        
        return exp_dir
    
    # Create new experiment as before
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"distil_{timestamp}"
    if not args.use_mac:
        exp_name += "_no_mac"
    
    # Create experiment directory structure
    exp_dir = Path("runs") / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    checkpoints_dir = exp_dir / "checkpoints"
    tensorboard_dir = exp_dir / "tensorboard"
    config_dir = exp_dir / "config"
    
    for d in [checkpoints_dir, tensorboard_dir, config_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Save experiment config
    config = vars(args)
    config['timestamp'] = timestamp
    with open(config_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f)
    
    return exp_dir

class DistillationTrainer:
    def __init__(
        self,
        teacher_model_id: str,
        exp_dir: Path,
        checkpoint_path: str = None,
        batch_size: int = 8,
        max_length: int = 256,
        temperature: float = 2.0,
        alpha: float = 0.5,
        distributed: bool = False,
        use_mac: bool = True,
    ):
        self.exp_dir = exp_dir
        self.use_mac = use_mac
        self.distributed = distributed
        self.world_size = get_world_size()
        
        # Define is_main_process based on distributed setup
        if distributed:
            self.device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
            self.is_main_process = int(os.environ['LOCAL_RANK']) == 0
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.is_main_process = True
        
        # Initialize wandb with experiment config
        if self.is_main_process:
            wandb.init(
                project="titans-distillation",
                name=exp_dir.name,
                config={
                    "batch_size": batch_size,
                    "max_length": max_length,
                    "temperature": temperature,
                    "alpha": alpha,
                    "use_mac": use_mac
                },
                dir=str(exp_dir)
            )
            
            # Initialize tensorboard writer with experiment directory
            self.writer = SummaryWriter(exp_dir / "tensorboard")
        
        self.temperature = temperature
        self.alpha = alpha
        self.max_length = max_length
        
        # Initialize tokenizer
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(teacher_model_id, token=os.getenv("HF_TOKEN"))
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize teacher model with reduced precision
        if distributed:
            device_map = {"": int(os.environ["LOCAL_RANK"])}
        else:
            device_map = "auto"
            
        self.teacher = AutoModelForCausalLM.from_pretrained(
            teacher_model_id,
            torch_dtype=torch.float16,
            device_map=device_map,
            token=os.getenv("HF_TOKEN")
        )
        self.teacher.gradient_checkpointing_enable()
        self.teacher.eval()
        
        # Initialize student model based on use_mac flag
        config = copy.deepcopy(self.teacher.config)
        config.hidden_size //= 2
        config.intermediate_size //= 2
        config.num_attention_heads //= 2
        config.num_key_value_heads //= 2
        config.num_hidden_layers //= 2
        config.max_position_embeddings //= 2
        config.rms_norm_eps /= 2

        if use_mac:
            mac_config = {
                "num_persistent": 16,
                "memory_size": 1024,
                "alpha": 0.1
            }
            mac_module = MACModule(dim=config.hidden_size, **mac_config)
            config.mac_module_config = mac_config
            config.architectures = ["MACTransformer"]
            config.model_type = "llama_mac"
            self.student = MACTransformer(config, mac_module)
        else:
            # Ensure no MAC module is initialized
            config.architectures = ["LlamaForCausalLM"]
            config.model_type = "llama"
            self.student = LlamaForCausalLM(config)

        if self.is_main_process:
            checkpoint_dir = self.exp_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            config_path = checkpoint_dir / 'initial_config.json'
            with open(config_path, 'w') as f:
                json.dump(config.to_dict(), f, indent=2)
            print(f"Exported initial config to {config_path}")
        
        # Move student model to device
        self.student = self.student.to(self.device)
        
        # Wrap student with DDP if in distributed mode
        if distributed:
            self.student = DDP(
                self.student,
                device_ids=[int(os.environ["LOCAL_RANK"])],
                output_device=int(os.environ["LOCAL_RANK"])
            )
        
        # Initialize optimizer and scaler AFTER moving model to device
        self.optimizer = torch.optim.AdamW(self.student.parameters(), lr=1e-4)
        self.scaler = torch.amp.GradScaler()

        # Load checkpoint if provided
        self.start_epoch = 0
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)
        
        # Initialize dataset and dataloader
        self.dataset = load_dataset("Open-Orca/SlimOrca", split="train", cache_dir="./data")
        
        # Adjust batch size based on number of GPUs
        self.batch_size = batch_size // self.world_size if self.world_size > 0 else batch_size
        
        if distributed:
            self.sampler = DistributedSampler(self.dataset)
        else:
            self.sampler = None
            
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=self.sampler,
            shuffle=(self.sampler is None),
            num_workers=4,
            pin_memory=True,
            # Custom collate function that handles different sample types:
            # - If the sample is a list of dicts (each with a "value" key), join the "value" entries.
            # - If the sample is a list of strings, join them directly.
            # - If the sample is a dict with a "text" key, use that.
            # - Otherwise convert the sample to string.
            collate_fn=lambda batch: {"text": [
                "\n".join(turn["value"] for turn in sample)
                    if isinstance(sample, list) and len(sample) > 0 and isinstance(sample[0], dict)
                else "\n".join(sample)
                    if isinstance(sample, list)
                else sample["text"]
                    if isinstance(sample, dict) and "text" in sample
                else str(sample)
                for sample in batch
            ]}
        )

    def initialize_student(self):
        """Initialize student model with same vocab size as teacher"""
        # Get the teacher's vocab size
        vocab_size = self.teacher.config.vocab_size
        
        config = copy.deepcopy(self.teacher.config)
        config.hidden_size = config.hidden_size // 2
        config.intermediate_size = config.intermediate_size // 2
        config.num_attention_heads = config.num_attention_heads // 2
        config.num_key_value_heads = config.num_key_value_heads // 2
        config.num_hidden_layers = config.num_hidden_layers // 2
        config.max_position_embeddings = config.max_position_embeddings // 2
        config.rms_norm_eps = config.rms_norm_eps / 2

        # Create the MAC module
        mac_module = MACModule(
            dim=config.hidden_size,
            num_persistent=16,
            memory_size=1024,
            alpha=0.1
        )
        
        # Initialize student model
        student = MACTransformer(config=config, mac_module=mac_module)
        
        # Enable gradient checkpointing for memory efficiency
        student.gradient_checkpointing_enable()
        
        # Move to appropriate device
        student.to(self.device)
        
        return student

    def prepare_batch(self, batch: dict[str, list[str]]):
        """Tokenize and prepare batch for training"""
        encoded: BatchEncoding = self.tokenizer(
            batch["text"],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {k: v.to(self.device) for k, v in encoded.items()}

    def compute_loss(self, teacher_logits, student_logits, input_ids):
        vocab_size = student_logits.shape[-1]

        # Shift logits and labels for causal LM
        teacher_logits = teacher_logits[:, :-1, :].contiguous()  # (B, T-1, V)
        student_logits = student_logits[:, :-1, :].contiguous()  # (B, T-1, V)
        labels = input_ids[:, 1:].contiguous()  # (B, T-1)

        # Create padding mask explicitly
        padding_mask = labels != self.tokenizer.pad_token_id  # (B, T-1)

        # Apply mask to logits and labels
        teacher_logits_filtered = teacher_logits[padding_mask]  # (N, V)
        student_logits_filtered = student_logits[padding_mask]  # (N, V)
        labels_filtered = labels[padding_mask]  # (N,)

        # Compute distillation loss (KL divergence)
        teacher_probs = F.softmax(teacher_logits_filtered / self.temperature, dim=-1)
        distil_loss = F.kl_div(
            F.log_softmax(student_logits_filtered / self.temperature, dim=-1),
            teacher_probs,
            reduction="batchmean"
        ) * (self.temperature ** 2)

        # Compute task loss (cross-entropy)
        task_loss = F.cross_entropy(
            student_logits_filtered.view(-1, vocab_size),
            labels_filtered.view(-1)
        )

        # Combine losses
        loss = self.alpha * distil_loss + (1 - self.alpha) * task_loss

        return loss, distil_loss, task_loss

    def train_epoch(self, epoch):
        self.student.train()
        self.teacher.eval()

        if self.sampler:
            self.sampler.set_epoch(epoch)

        total_loss = 0
        num_batches = 0

        with tqdm(self.dataloader, desc=f"Epoch {epoch}", disable=not self.is_main_process) as pbar:
            for batch_idx, batch in enumerate(pbar):
                self.optimizer.zero_grad()
                inputs = self.prepare_batch(batch)

                with torch.no_grad(), torch.amp.autocast(device_type='cuda'):
                    teacher_outputs = self.teacher(**inputs)
                    teacher_logits = teacher_outputs.logits

                with torch.amp.autocast("cuda"):
                    if self.use_mac:
                        student_outputs = self.student(tokens=inputs["input_ids"], use_mac=True)
                    else:
                        student_outputs = self.student(**inputs).logits  # <-- standard forward pass without MAC

                    loss, distil_loss, task_loss = self.compute_loss(
                        teacher_logits,
                        student_outputs,
                        inputs["input_ids"]
                    )

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                total_loss += loss.item()
                num_batches += 1

                if self.is_main_process:
                    pbar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss

    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint safely"""
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")

        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model state
        state_dict = checkpoint['student_state_dict']
        if self.distributed:
            self.student.module.load_state_dict(state_dict)
        else:
            self.student.load_state_dict(state_dict)
        
        # Ensure model is on correct device after loading
        if self.distributed:
            self.student.module = self.student.module.to(self.device)
        else:
            self.student = self.student.to(self.device)

        # Load optimizer state if available
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # Move optimizer state to correct device
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
        else:
            print("Warning: No optimizer state found in checkpoint")

        # Load scaler state if available
        if 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        else:
            print("Warning: No scaler state found in checkpoint, using default initialization")

        self.start_epoch = checkpoint.get('epoch', 0) + 1

        if self.is_main_process:
            wandb.log({
                "checkpoint/resume_epoch": self.start_epoch,
                "checkpoint/previous_loss": checkpoint.get('loss', float('nan'))
            })

    def log_gradient_stats(self, model, step):
        """Log gradient statistics to wandb"""
        if not self.is_main_process:
            return
        
        grad_norm_dict = {}
        max_grad_dict = {}
        has_nan_dict = {}
        
        # Check gradients for each parameter group
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad
                
                # Calculate gradient norm
                grad_norm = torch.norm(grad).item()
                grad_norm_dict[f"grad_norm/{name}"] = grad_norm
                
                # Calculate max gradient
                max_grad = torch.max(torch.abs(grad)).item()
                max_grad_dict[f"grad_max/{name}"] = max_grad
                
                # Check for NaNs
                has_nan = torch.isnan(grad).any().item()
                has_nan_dict[f"grad_nan/{name}"] = has_nan
        
        # Log aggregated statistics
        wandb.log({
            "grad/mean_norm": np.mean(list(grad_norm_dict.values())),
            "grad/max_norm": np.max(list(grad_norm_dict.values())),
            "grad/max_value": np.max(list(max_grad_dict.values())),
            "grad/has_nan": any(has_nan_dict.values()),
            **grad_norm_dict,
            **max_grad_dict,
            **has_nan_dict
        }, step=step)

    def add_activation_monitoring(self):
        """Add hooks to monitor activations"""
        self.activation_hooks = []
        self.activation_stats = {}
        
        def hook_fn(name):
            def fn(module, input, output):
                if not self.is_main_process:
                    return
                    
                # For list/tuple outputs, check the first item
                if isinstance(output, (list, tuple)):
                    output = output[0]
                    
                # Skip non-tensor outputs
                if not isinstance(output, torch.Tensor):
                    return
                    
                # Check for NaNs
                has_nan = torch.isnan(output).any().item()
                
                if has_nan or np.random.random() < 0.01:  # Log all NaNs + random sampling
                    self.activation_stats[f"act_nan/{name}"] = has_nan
                    
                    if not has_nan:
                        # Only log these if not NaN to avoid errors
                        self.activation_stats[f"act_mean/{name}"] = output.mean().item()
                        self.activation_stats[f"act_std/{name}"] = output.std().item()
                        self.activation_stats[f"act_max/{name}"] = output.abs().max().item()
            
            return fn
        
        # Register hooks for modules
        model = self.student.module if self.distributed else self.student
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.LayerNorm)) or "RMSNorm" in module.__class__.__name__:
                hook = module.register_forward_hook(hook_fn(name))
                self.activation_hooks.append(hook)
        
        return self.activation_hooks

    def log_activation_stats(self):
        """Log activation statistics to wandb"""
        if not self.is_main_process or not self.activation_stats:
            return
        
        wandb.log(self.activation_stats)
        self.activation_stats = {}  # Clear stats after logging

    def save_checkpoint(self, epoch: int, loss: float, batch_idx: int = None):
        """Save checkpoint with consistent naming"""
        if not self.is_main_process:
            return
            
        checkpoint_name = f'checkpoint_epoch_{epoch}'
        if batch_idx is not None:
            checkpoint_name += f'_batch_{batch_idx}'
        checkpoint_name += '.pt'
        
        checkpoint_path = self.exp_dir / "checkpoints" / checkpoint_name
        
        state_dict = self.student.module.state_dict() if self.distributed else self.student.state_dict()
        torch.save({
            'epoch': epoch,
            'batch_idx': batch_idx,
            'student_state_dict': state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'loss': loss,
            'config': {
                'use_mac': self.use_mac,
                # Add other relevant config items
            }
        }, checkpoint_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-model-id", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct",
                        help="Path to the teacher model")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for training")
    parser.add_argument("--max-length", type=int, default=256,
                        help="Maximum sequence length for input")
    parser.add_argument("--temperature", type=float, default=2.0,
                        help="Temperature for distillation")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Weighting factor for distillation loss")
    parser.add_argument("--use-mac", action="store_true", default=False,
                        help="Use MAC module in student model")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest experiment instead of creating new one")
    parser.add_argument("--num-epochs", type=int, default=5,
                        help="Number of epochs to train")

    args = parser.parse_args()

    # Setup distributed training
    distributed = setup_distributed()

    # Setup experiment directory and save config
    exp_dir = setup_experiment(args)

    # Find latest checkpoint
    latest_checkpoint = None
    checkpoints_dir = exp_dir / "checkpoints"
    if checkpoints_dir.exists():
        checkpoints = sorted(checkpoints_dir.glob('checkpoint_epoch_*.pt'), key=os.path.getmtime)
        if checkpoints:
            latest_checkpoint = checkpoints[-1]
            print(f"Found checkpoint: {latest_checkpoint}")

    trainer = DistillationTrainer(
        teacher_model_id=args.teacher_model_id,
        exp_dir=exp_dir,
        checkpoint_path=str(latest_checkpoint) if latest_checkpoint else None,
        batch_size=args.batch_size,
        max_length=args.max_length,
        temperature=args.temperature,
        alpha=args.alpha,
        distributed=distributed,
        use_mac=args.use_mac
    )

    for epoch in range(trainer.start_epoch, args.num_epochs):
        loss = trainer.train_epoch(epoch)

        # Save checkpoint at end of epoch
        if trainer.is_main_process:
            trainer.save_checkpoint(epoch, loss)

    # Cleanup
    if trainer.is_main_process and trainer.writer is not None:
        trainer.writer.close()
        wandb.finish()

    cleanup()

if __name__ == "__main__":
    main()