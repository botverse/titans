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

class DistillationTrainer:
    def __init__(
        self,
        teacher_model_id: str,
        log_dir: str = "runs/distillation",
        checkpoint_path: str = None,
        batch_size: int = 8,
        max_length: int = 256,
        temperature: float = 2.0,
        alpha: float = 0.5,
        distributed: bool = False
    ):
        self.distributed = distributed
        self.world_size = get_world_size()
        
        if distributed:
            self.device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
            self.is_main_process = int(os.environ['LOCAL_RANK']) == 0
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.is_main_process = True
        
        # Initialize tensorboard writer only on main process
        if self.is_main_process:
            self.writer = SummaryWriter(log_dir)
        else:
            self.writer = None
            
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
            torch_dtype=torch.float16,  # Use half precision for teacher
            device_map=device_map,
            token=os.getenv("HF_TOKEN")
        )
        self.teacher.gradient_checkpointing_enable()
        self.teacher.eval()
        
        # Initialize student model
        self.student = self.initialize_student()
        
        # Wrap student with DDP if in distributed mode
        if distributed:
            self.student = DDP(
                self.student,
                device_ids=[int(os.environ["LOCAL_RANK"])],
                output_device=int(os.environ["LOCAL_RANK"])
            )
        
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
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(self.student.parameters(), lr=1e-4)
        self.scaler = torch.amp.GradScaler()

        if self.is_main_process:
            wandb.init(
                project="titans-distillation",
                config={
                    "teacher_model_id": teacher_model_id,
                    "batch_size": batch_size,
                    "max_length": max_length,
                    "temperature": temperature,
                    "alpha": alpha,
                    "learning_rate": 1e-4,
                    "distributed": distributed,
                }
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
        
        # Convert to half precision
        student.half()
        
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
        """Compute the distillation and task losses"""
        print(f"[DEBUG] compute_loss - teacher_logits: {teacher_logits.shape}, any NaN: {torch.isnan(teacher_logits).any()}")
        print(f"[DEBUG] compute_loss - student_logits: {student_logits.shape}, any NaN: {torch.isnan(student_logits).any()}")
        
        # Temperature scaling for distillation
        teacher_logits_scaled = teacher_logits / self.temperature
        student_logits_scaled = student_logits / self.temperature
        
        print(f"[DEBUG] compute_loss - after scaling - teacher: any NaN: {torch.isnan(teacher_logits_scaled).any()}")
        print(f"[DEBUG] compute_loss - after scaling - student: any NaN: {torch.isnan(student_logits_scaled).any()}")
        
        # Create targets for next token prediction
        shift_logits = student_logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        
        print(f"[DEBUG] compute_loss - shift_logits: {shift_logits.shape}, any NaN: {torch.isnan(shift_logits).any()}")
        print(f"[DEBUG] compute_loss - shift_labels: {shift_labels.shape}, any NaN: {torch.isnan(shift_labels).any()}")
        
        # Cross-entropy loss for task objective
        task_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        
        print(f"[DEBUG] compute_loss - task_loss: {task_loss.item()}, is NaN: {torch.isnan(task_loss).any()}")
        
        # KL divergence for distillation
        teacher_probs = F.softmax(teacher_logits_scaled[..., :-1, :], dim=-1)
        print(f"[DEBUG] compute_loss - teacher_probs: {teacher_probs.shape}, any NaN: {torch.isnan(teacher_probs).any()}")
        
        student_log_softmax = F.log_softmax(student_logits_scaled[..., :-1, :], dim=-1)
        print(f"[DEBUG] compute_loss - student_log_softmax: {student_log_softmax.shape}, any NaN: {torch.isnan(student_log_softmax).any()}")
        
        distil_loss = F.kl_div(
            student_log_softmax,
            teacher_probs,
            reduction="batchmean",
        )
        
        print(f"[DEBUG] compute_loss - distil_loss: {distil_loss.item() if not torch.isnan(distil_loss) else 'NaN'}")
        
        # Combined loss
        loss = self.alpha * distil_loss + (1 - self.alpha) * task_loss
        
        print(f"[DEBUG] compute_loss - final loss: {loss.item() if not torch.isnan(loss) else 'NaN'}")
        
        return loss, distil_loss, task_loss

    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.student.train()
        self.teacher.eval()
        
        if self.sampler:
            self.sampler.set_epoch(epoch)
        
        total_loss = 0
        total_distil_loss = 0
        total_task_loss = 0
        num_batches = 0
        nan_count = 0
        
        # Before the batch loop, initialize previous memory state
        if self.distributed:
            mac_module = self.student.module.mac_module
        else:
            mac_module = self.student.mac_module
        # Check initial memory state
        p_mem = mac_module.persistent_memory.cpu().detach().numpy()
        l_mem = mac_module.long_term_memory.cpu().detach().numpy()
        
        # Log detailed memory statistics
        wandb.log({
            "initial/persistent_memory_min": float(np.min(p_mem)),
            "initial/persistent_memory_max": float(np.max(p_mem)),
            "initial/persistent_memory_mean": float(np.mean(p_mem)),
            "initial/persistent_memory_std": float(np.std(p_mem)),
            "initial/long_term_memory_min": float(np.min(l_mem)),
            "initial/long_term_memory_max": float(np.max(l_mem)),
            "initial/long_term_memory_mean": float(np.mean(l_mem)),
            "initial/long_term_memory_std": float(np.std(l_mem)),
        })
        

        previous_memory_state = mac_module.long_term_memory.clone().detach()
        
        # Add activation monitoring
        activation_hooks = self.add_activation_monitoring()
        try:
            with tqdm(self.dataloader, desc=f"Epoch {epoch}", disable=not self.is_main_process) as pbar:
                for batch_idx, batch in enumerate(pbar):
                    self.optimizer.zero_grad()
                    
                    inputs = self.prepare_batch(batch)
                    
                    # Teacher forward pass (no gradients needed)
                    with torch.no_grad(), torch.amp.autocast(device_type='cuda'):
                        teacher_outputs = self.teacher(**inputs)
                        teacher_logits = teacher_outputs.logits
                    
                    # Student forward pass and backward pass (must be in same AMP context)
                    with torch.amp.autocast("cuda"):
                        student_outputs = self.student(
                            tokens=inputs["input_ids"],
                            start_pos=0,
                            use_mac=True
                        )

                        loss, distil_loss, task_loss = self.compute_loss(
                            teacher_logits,
                            student_outputs,
                            inputs["input_ids"]
                        )
                    
                    # Scale the loss and backpropagate with AMP
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    
                    # Log NaN detection (but don't change training behavior)
                    if torch.isnan(loss):
                        nan_count += 1
                        if self.is_main_process:
                            print(f"WARNING: NaN detected in loss at batch {batch_idx}")
                            wandb.log({"nan_detected": batch_idx})
                    
                    total_loss += loss.item() if not torch.isnan(loss) else 0
                    total_distil_loss += distil_loss.item() if not torch.isnan(distil_loss) else 0
                    total_task_loss += task_loss.item() if not torch.isnan(task_loss) else 0
                    num_batches += 1
                    
                    if self.is_main_process:
                        # Log standard metrics (safely handling NaNs)
                        metrics = {
                            "batch_loss": loss.item() if not torch.isnan(loss) else -1,
                            "batch_distil_loss": distil_loss.item() if not torch.isnan(distil_loss) else -1,
                            "batch_task_loss": task_loss.item() if not torch.isnan(task_loss) else -1,
                        }
                        wandb.log(metrics)
                        
                        pbar.set_postfix({
                            'loss': loss.item() if not torch.isnan(loss) else float('nan'),
                            'distil_loss': distil_loss.item() if not torch.isnan(distil_loss) else float('nan'),
                            'task_loss': task_loss.item() if not torch.isnan(task_loss) else float('nan')
                        })

                        # Monitor memory stats - skip histograms if there are NaNs
                        p_mem = mac_module.persistent_memory.cpu().detach().numpy() 
                        l_mem = mac_module.long_term_memory.cpu().detach().numpy()
                        
                        # Detect NaNs in memory
                        has_nan_persistent = np.isnan(p_mem).any()
                        has_nan_longterm = np.isnan(l_mem).any()
                        
                        if has_nan_persistent or has_nan_longterm:
                            wandb.log({
                                "MAC/has_nan_persistent": has_nan_persistent,
                                "MAC/has_nan_longterm": has_nan_longterm
                            })
                            
                        # Log basic memory statistics
                        memory_metrics = {}
                        
                        # Memory update magnitude
                        if not has_nan_longterm and not np.isnan(previous_memory_state.cpu().numpy()).any():
                            memory_update_magnitude = torch.norm(mac_module.long_term_memory - previous_memory_state).item()
                            memory_metrics["MAC/memory_update_magnitude"] = memory_update_magnitude
                        
                        # Basic stats instead of histograms
                        if not has_nan_persistent:
                            memory_metrics.update({
                                "MAC/persistent_memory_min": float(np.min(p_mem)),
                                "MAC/persistent_memory_max": float(np.max(p_mem)),
                                "MAC/persistent_memory_mean": float(np.mean(p_mem)),
                                "MAC/persistent_memory_std": float(np.std(p_mem))
                            })
                        
                        if not has_nan_longterm:
                            memory_metrics.update({
                                "MAC/long_term_memory_min": float(np.min(l_mem)),
                                "MAC/long_term_memory_max": float(np.max(l_mem)),
                                "MAC/long_term_memory_mean": float(np.mean(l_mem)),
                                "MAC/long_term_memory_std": float(np.std(l_mem))
                            })
                        
                        wandb.log(memory_metrics)
                        
                        # Only update previous memory state if it doesn't have NaNs
                        if not has_nan_longterm:
                            previous_memory_state = mac_module.long_term_memory.clone().detach()

                        # Log activation stats
                        self.log_activation_stats()
                
            if self.is_main_process:
                # Compute safe averages
                avg_loss = total_loss / max(num_batches, 1)  # Avoid division by zero
                avg_distil_loss = total_distil_loss / max(num_batches, 1)
                avg_task_loss = total_task_loss / max(num_batches, 1)
                
                # Log epoch stats
                wandb.log({
                    "epoch": epoch,
                    "avg_loss": avg_loss,
                    "avg_distil_loss": avg_distil_loss,
                    "avg_task_loss": avg_task_loss,
                    "learning_rate": self.optimizer.param_groups[0]['lr'],
                    "nan_count": nan_count
                })
            
            return total_loss / max(num_batches, 1)  # Avoid division by zero
        finally:
            # Remove hooks after epoch
            for hook in activation_hooks:
                hook.remove()

    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint and resume training state"""
        if self.is_main_process:
            print(f"Loading checkpoint from {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model state
        if self.distributed:
            self.student.module.load_state_dict(checkpoint['student_state_dict'])
        else:
            self.student.load_state_dict(checkpoint['student_state_dict'])

        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Set starting epoch
        self.start_epoch = checkpoint['epoch'] + 1

        if self.is_main_process:
            print(f"Resuming from epoch {self.start_epoch}")
            print(f"Previous loss: {checkpoint['loss']}")

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

def main():
    try:
        # Initialize distributed training if applicable
        distributed = setup_distributed()

        # Create checkpoint and log directories
        log_dir = Path("runs/distillation")
        checkpoint_dir = Path("checkpoints")

        if (distributed and int(os.environ['LOCAL_RANK']) == 0) or (not distributed):
            log_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Check for latest checkpoint
        latest_checkpoint = None
        if checkpoint_dir.exists():
            checkpoints = list(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=lambda x: int(x.stem.split('_')[-1]))

        trainer = DistillationTrainer(
            teacher_model_id="meta-llama/Meta-Llama-3-8B-Instruct",
            log_dir=str(log_dir),
            checkpoint_path=str(latest_checkpoint) if latest_checkpoint else None,
            batch_size=2,
            max_length=512,
            temperature=2.0,
            alpha=0.5,
            distributed=distributed
        )

        num_epochs = 1  # <-- set epochs to 1 for quick testing
        for epoch in range(trainer.start_epoch, num_epochs):
            loss = trainer.train_epoch(epoch)

            if trainer.is_main_process:
                print(f"Epoch {epoch} average loss: {loss}")

                # Save checkpoint after the epoch
                state_dict = trainer.student.module.state_dict() if distributed else trainer.student.state_dict()
                checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
                torch.save({
                    'epoch': epoch,
                    'student_state_dict': state_dict,
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'loss': loss,
                }, checkpoint_path)

        # Close tensorboard writer
        if trainer.is_main_process and trainer.writer is not None:
            trainer.writer.close()
    finally:
        if trainer is not None and trainer.is_main_process:
            wandb.finish()
        cleanup()

if __name__ == "__main__":
    main()