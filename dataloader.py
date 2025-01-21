from transformers import PreTrainedTokenizer
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
import torch
from typing import Dict, Optional
from torch.utils.data.distributed import DistributedSampler
from trl.trainer.utils import DataCollatorForChatML


def map_role(from_value):
    """Map conversation roles to standardized format"""
    if from_value.lower() == 'human':
        return 'user'
    elif from_value.lower() in ['gpt', 'assistant']:
        return 'assistant'
    return 'unknown'

class GKDDataset(Dataset):
    def __init__(
        self,
        dataset_name: str = "BAAI/Infinity-Instruct",
        split: str = "train",
        tokenizer: PreTrainedTokenizer = None,
        max_samples: Optional[int] = None,
        max_length: Optional[int] = 2048,
    ):
        # Load dataset from HuggingFace
        self.dataset = load_dataset(dataset_name, "3M", split=split, streaming=True)
        if max_samples:
            self.dataset = self.dataset.take(max_samples)
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Process conversations into ChatML format
        self.processed_data = []
        for item in self.dataset:
            self.processed_data.append({
                "messages": [
                    {"role": map_role(turn["from"]), "content": turn["value"]}
                    for turn in item["conversations"]
                ]
            })
            if len(self.processed_data) % 100 == 0:
                print(f"Processed {len(self.processed_data)} conversations")

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        return self.processed_data[idx]

def create_gkd_dataloaders(
    tokenizer: PreTrainedTokenizer,
    dataset_name: str = "BAAI/Infinity-Instruct",
    max_samples: Optional[int] = None,
    batch_size: int = 1,
    num_workers: int = 0,
    rank: int = 0,
    world_size: int = 1,
    max_length: int = 2048,
) -> Dict[str, DataLoader]:
    """Create distributed dataloaders for GKD training"""
    
    print("Creating a dataloader on rank: ", rank, world_size)
    # Create datasets
    train_dataset = GKDDataset(
        dataset_name=dataset_name,
        split="train",
        tokenizer=tokenizer,
        max_samples=max_samples,
        max_length=max_length,
    )
    
    # Create data collator
    data_collator = DataCollatorForChatML(
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    # Create distributed samplers
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=data_collator
    )
    
    return {
        "train": train_loader,
        "train_sampler": train_sampler,
        "max_length": max_length
    }