import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataloader import create_gkd_dataloaders
from trainer import GKDTrainer
import wandb
from accelerate import init_empty_weights
import bitsandbytes as bnb

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def setup_distributed():
    """Initialize distributed training"""
    local_rank = int(os.environ["LOCAL_RANK"])
    # Set device before initializing process group
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method='env://')
    world_size = dist.get_world_size()
    return local_rank, world_size

def load_tokenizer(model_name: str):
    """Load and configure tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def load_8bit_model(model_name: str, device_map: str):
    """Load model in 8-bit precision"""
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        load_in_8bit=True,
        torch_dtype=torch.float16,
    )

def load_distributed_model(model_name: str, local_rank: int):
    """Load model for distributed training"""
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": f'cuda:{local_rank}'}
    )

def setup_teacher_device_map():
    """Define the teacher model device map"""
    teacher_device_map = {
        'model.embed_tokens': 0, 
        'model.layers.0': 0, 'model.layers.1': 0, 'model.layers.2': 0, 'model.layers.3': 0,
        'model.layers.4': 0, 'model.layers.5': 0, 'model.layers.6': 0, 'model.layers.7': 0,
        'model.layers.8': 0, 'model.layers.9': 0, 'model.layers.10': 0, 'model.layers.11': 0,
        'model.layers.12': 0, 'model.layers.13': 0, 'model.layers.14': 0, 'model.layers.15': 0,
        'model.layers.16': 0, 'model.layers.17': 0, 'model.layers.18': 0, 'model.layers.19': 0,
        'model.layers.20': 0, 'model.layers.21': 0, 'model.layers.22': 0, 'model.layers.23': 0,
        'model.layers.24': 0, 'model.layers.25': 0, 'model.layers.26': 0, 'model.layers.27': 0,
        'model.layers.28': 0, 'model.layers.29': 0, 'model.layers.30': 0, 'model.layers.31': 0,
        'model.layers.32': 0, 'model.layers.33': 0, 'model.layers.34': 0, 'model.layers.35': 0,
        'model.layers.36': 0, 'model.layers.37': 0, 'model.layers.38': 0, 'model.layers.39': 0,
        'model.layers.40': 0, 'model.layers.41': 0, 'model.layers.42': 0, 
        'model.layers.43.self_attn': 0,
        'model.layers.43.input_layernorm': 1, 'model.layers.43.post_attention_layernorm': 1,
        'model.layers.43.mlp': 1,
        'model.layers.44': 1, 'model.layers.45': 1, 'model.layers.46': 1, 'model.layers.47': 1,
        'model.layers.48': 1, 'model.layers.49': 1, 'model.layers.50': 1, 'model.layers.51': 1,
        'model.layers.52': 1, 'model.layers.53': 1, 'model.layers.54': 1, 'model.layers.55': 1,
        'model.layers.56': 1, 'model.layers.57': 1, 'model.layers.58': 1, 'model.layers.59': 1,
        'model.layers.60': 1, 'model.layers.61': 1, 'model.layers.62': 1, 'model.layers.63': 1,
        'model.layers.64': 1, 'model.layers.65': 1, 'model.layers.66': 1, 'model.layers.67': 1,
        'model.layers.68': 1, 'model.layers.69': 1, 'model.layers.70': 1, 'model.layers.71': 1,
        'model.layers.72': 1, 'model.layers.73': 1, 'model.layers.74': 1, 'model.layers.75': 1,
        'model.layers.76': 1, 'model.layers.77': 1, 'model.layers.78': 1, 'model.layers.79': 1,
        'model.norm': 1, 'model.rotary_emb': 1, 'lm_head': 1
    }
    return teacher_device_map

def setup_teacher_pipeline(model_name: str, local_rank: int):
    """Setup teacher model in pipeline parallel across GPUs 0 and 1"""
    # Create device maps for each GPU that include all layers but map unused ones to CPU
    if local_rank == 0:
        # GPU 0 device map: First half on GPU, rest on CPU
        device_map = {
            'model.embed_tokens': 0,
            'model.rotary_emb': 0,
            **{f'model.layers.{i}': 0 for i in range(44)},  # First 44 layers on GPU 0
            **{f'model.layers.{i}': 'cpu' for i in range(44, 80)},  # Rest on CPU
            'model.norm': 'cpu',
            'lm_head': 'cpu'
        }
    else:  # local_rank == 1
        # GPU 1 device map: Second half on GPU, rest on CPU
        device_map = {
            'model.embed_tokens': 'cpu',
            'model.rotary_emb': 1,
            **{f'model.layers.{i}': 'cpu' for i in range(44)},  # First half on CPU
            **{f'model.layers.{i}': 1 for i in range(44, 80)},  # Second half on GPU 1
            'model.norm': 1,
            'lm_head': 1
        }

    # Load model with appropriate device map
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=True,  # Enable CPU offload for 8-bit quantization
        torch_dtype=torch.bfloat16,
    )

    # Create a process group for the teacher pipeline
    teacher_group = dist.new_group([0, 1])
    
    return model, teacher_group

def main():
    # Initialize distributed setup
    local_rank, world_size = setup_distributed()
    is_main = local_rank == 0
    is_teacher = local_rank <= 1  # GPUs 0 and 1 are teacher
    
    # Model configuration
    teacher_model_name = "Qwen/Qwen2-72B-Instruct"
    student_model_name = "Qwen/Qwen2-7B-Instruct"
    student_group = dist.new_group(ranks=list(range(2, world_size)))
    
    # Training configuration
    batch_size = 1
    gradient_accumulation_steps = 16
    num_epochs = 3
    max_samples = 1000  # Limit dataset size for testing
    
    if is_main:
        print(f"Starting distributed training with {world_size} GPUs")
        
    # Load tokenizer
    tokenizer = load_tokenizer(student_model_name)
    # Synchronize before dataloader creation with device specified
    dist.barrier(device_ids=[local_rank])
    # Create dataloaders
    dataloaders = create_gkd_dataloaders(
        tokenizer=tokenizer,
        dataset_name="BAAI/Infinity-Instruct",
        max_samples=max_samples,
        batch_size=batch_size,
        num_workers=0,
        rank=local_rank,
        world_size=world_size,
        max_length=2048
    )
    train_loader = dataloaders["train"]

    dist.barrier(device_ids=[local_rank])
    
    # Load models based on rank
    teacher_model = None
    student_model = None
    if is_teacher:
        # GPU 0: Only load teacher model
        print(f"Rank 0: Loading teacher model {teacher_model_name} in 8-bit")
        teacher_model, teacher_group = setup_teacher_pipeline(teacher_model_name, local_rank)
        teacher_model.eval()
        student_model = None
    else:
        # GPUs 1-7: Only load student model
        print(f"Rank {local_rank}: Loading student model {student_model_name}")
        teacher_model = None
        student_model = load_distributed_model(student_model_name, local_rank)
        # Wrap student model in DDP
        student_model = DDP(
            student_model,
            device_ids=[local_rank],
            output_device=local_rank,
            process_group=student_group,
            find_unused_parameters=False,
        )

    dist.barrier()
    # Sample evaluation prompts
    eval_prompts = [
        "Write a story about a magical forest",
        "Explain how a car engine works",
        "What are the benefits of exercise?",
    ] if is_main else None
    
    # Initialize trainer
    trainer = GKDTrainer(
        student_model=student_model,
        teacher_model=teacher_model,
        tokenizer=tokenizer,
        optimizer_kwargs={
            "lr": 5e-5,
            "weight_decay": 0.01,
            "betas": (0.9, 0.999)
        },
        local_rank=local_rank,
        rank=local_rank,
        eval_prompts=eval_prompts,
        gradient_accumulation_steps=gradient_accumulation_steps,
        beta=0.5,
        temperature=1.0,
        lambda_gkd=0.5,
    )
    
    # Training loop
    for epoch in range(num_epochs):
        if is_main:
            print(f"\nStarting epoch {epoch + 1}/{num_epochs}")
            
        stats = trainer.train_epoch(train_loader, epoch)
        
        # Make sure all GPUs are synced before proceeding
        dist.barrier()
        
        if is_main:
            print(f"\nEpoch {epoch + 1} stats:")
            for k, v in stats.items():
                print(f"{k}: {v:.4f}")
        
        # Gather stats from GPU 1 (first student GPU) to GPU 0
        if local_rank == 1:
            checkpoint_path = f"checkpoint_epoch_{epoch + 1}.pt"
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': student_model.module.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'stats': stats,
            }
            torch.save(checkpoint, checkpoint_path)
            
            if wandb.run is not None:
                wandb.save(checkpoint_path)
        
        # Make sure checkpoint is saved before proceeding
        dist.barrier()
    
    # Cleanup
    dist.destroy_process_group()
    if is_main and wandb.run is not None:
        wandb.finish()

if __name__ == "__main__":
    # Prevent multiple wandb runs when using torchrun
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        os.environ["WANDB_DISABLED"] = "false"
    else:
        os.environ["WANDB_DISABLED"] = "true"
        
    main()