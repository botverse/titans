import dotenv
dotenv.load_dotenv()

import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM
from models.llama_titans import MACModule, MACTransformer, ModelArgs
from datasets import load_dataset
from tqdm import tqdm
import socket
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import fairscale.nn.model_parallel.initialize as fs_init

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
        self.tokenizer = AutoTokenizer.from_pretrained(teacher_model_id, token=os.getenv("HF_TOKEN"))
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize teacher model
        if distributed:
            device_map = {"": int(os.environ["LOCAL_RANK"])}
        else:
            device_map = "auto"
            
        self.teacher = AutoModelForCausalLM.from_pretrained(
            teacher_model_id,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            token=os.getenv("HF_TOKEN")
        )
        self.teacher.gradient_checkpointing_enable()
        self.teacher.eval()
        
        # Initialize student model
        self.student = MACTransformer(
            params=ModelArgs(
                dim=self.teacher.config.hidden_size,
                n_layers=self.teacher.config.num_hidden_layers,
                n_kv_heads=self.teacher.config.num_key_value_heads,
                vocab_size=self.teacher.config.vocab_size,
                max_seq_len=self.max_length,
            ),
            mac_module=MACModule(
                dim=self.teacher.config.hidden_size,
                num_persistent=16,
                memory_size=1024,
                alpha=0.1
            )
        ).to(self.device)
        
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

    def prepare_batch(self, batch):
        """Tokenize and prepare batch for training"""
        encoded = self.tokenizer(
            batch["text"],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {k: v.to(self.device) for k, v in encoded.items()}

    def compute_loss(self, teacher_logits, student_logits, labels):
        """Compute combined distillation and task loss"""
        # Distillation loss (KL divergence)
        distil_loss = (
            F.kl_div(
                F.log_softmax(student_logits / self.temperature, dim=-1),
                F.softmax(teacher_logits / self.temperature, dim=-1),
                reduction="batchmean",
            )
            * (self.temperature ** 2)
        )
        
        # Task loss (cross entropy)
        task_loss = F.cross_entropy(student_logits.view(-1, student_logits.size(-1)), labels.view(-1))
        
        # Combined loss
        loss = (self.alpha * distil_loss) + ((1 - self.alpha) * task_loss)
        return loss, distil_loss, task_loss

    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.student.train()
        if self.sampler:
            self.sampler.set_epoch(epoch)
        
        total_loss = 0
        total_distil_loss = 0
        total_task_loss = 0
        num_batches = 0
        
        with tqdm(self.dataloader, desc=f"Epoch {epoch}", disable=not self.is_main_process) as pbar:
            for batch in pbar:
                self.optimizer.zero_grad()
                
                # Prepare input batch
                inputs = self.prepare_batch(batch)
                
                # Get teacher predictions
                with torch.amp.autocast("cuda"):
                    teacher_outputs = self.teacher(**inputs)
                    teacher_logits = teacher_outputs.logits
                
                # Get student predictions
                with torch.amp.autocast("cuda"):
                    student_outputs = self.student(
                        tokens=inputs["input_ids"],
                        start_pos=0,
                        use_mac=True
                    )
                
                # Compute loss
                loss, distil_loss, task_loss = self.compute_loss(
                    teacher_logits,
                    student_outputs,
                    inputs["input_ids"]
                )
                
                # Scale loss and backward pass with AMP
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                
                # Update running averages
                total_loss += loss.item()
                total_distil_loss += distil_loss.item()
                total_task_loss += task_loss.item()
                num_batches += 1
                
                if self.is_main_process:
                    pbar.set_postfix({
                        'loss': loss.item(),
                        'distil_loss': distil_loss.item(),
                        'task_loss': task_loss.item()
                    })
        
        # Log epoch metrics to tensorboard
        if self.is_main_process and self.writer is not None:
            avg_loss = total_loss / num_batches
            avg_distil_loss = total_distil_loss / num_batches
            avg_task_loss = total_task_loss / num_batches
            
            self.writer.add_scalar('Loss/train', avg_loss, epoch)
            self.writer.add_scalar('Loss/distillation', avg_distil_loss, epoch)
            self.writer.add_scalar('Loss/task', avg_task_loss, epoch)
            
            # Add learning rate
            self.writer.add_scalar('Learning_rate', 
                                 self.optimizer.param_groups[0]['lr'], 
                                 epoch)
        
        return total_loss / num_batches

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
            batch_size=8,
            max_length=256,
            temperature=2.0,
            alpha=0.5,
            distributed=distributed
        )
        
        num_epochs = 10
        for epoch in range(trainer.start_epoch, num_epochs):
            loss = trainer.train_epoch(epoch)
            
            if trainer.is_main_process:
                print(f"Epoch {epoch} average loss: {loss}")
                
                # Save checkpoint
                if (epoch + 1) % 5 == 0:
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
        cleanup()

if __name__ == "__main__":
    main()