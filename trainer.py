from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
from typing import Dict, List, Any, Optional
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import wandb
import random
import torch.distributed as dist

class GKDTrainer:
    def __init__(
        self,
        student_model: torch.nn.Module,
        teacher_model: torch.nn.Module,
        tokenizer,
        optimizer_kwargs: Dict[str, Any] = None,
        local_rank: int = 0,
        rank: int = 0,
        eval_prompts: List[str] = None,
        scaler: Optional[GradScaler] = None,
        gradient_accumulation_steps: int = 1,
        beta: float = 0.5,  # GKD interpolation coefficient
        temperature: float = 1.0,  # Temperature for softmax
        lambda_gkd: float = 0.5,  # Probability of using on-policy samples
    ):
        self.student = student_model
        self.teacher = teacher_model
        self.device = torch.device(f'cuda:{local_rank}')
        self.local_rank = local_rank
        self.rank = rank
        self.is_main = (rank == 0 or rank == 1)
        self.tokenizer = tokenizer
        self.eval_prompts = eval_prompts
        self.scaler = scaler or GradScaler()
        self.use_amp = True
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # GKD specific parameters
        self.beta = beta
        self.temperature = temperature
        self.lambda_gkd = lambda_gkd
        
        # Only initialize optimizer on non-zero ranks (where student model exists)
        if not self.is_main:
            optimizer_kwargs = optimizer_kwargs or {
                "lr": 5e-5,
                "weight_decay": 0.01,
                "betas": (0.9, 0.999)
            }
            
            self.optimizer = AdamW(student_model.parameters(), **optimizer_kwargs)
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer, T_0=20, T_mult=2
            )
        else:
            self.optimizer = None
            self.scheduler = None

        # Put teacher in eval mode if it exists
        if self.teacher is not None:
            self.teacher.eval()
        
        if self.is_main:
            wandb.init(
                project="gkd-training",
                config={
                    "beta": beta,
                    "temperature": temperature,
                    "lambda_gkd": lambda_gkd,
                    "optimizer": optimizer_kwargs,
                }
            )

    def generalized_jsd_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        reduction: str = "batchmean"
    ) -> torch.Tensor:
        """
        Compute Generalized Jensen-Shannon Divergence loss for knowledge distillation
        """
        # Apply temperature scaling
        student_logits = student_logits / self.temperature
        teacher_logits = teacher_logits / self.temperature
        
        # Compute log probabilities
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
        
        # Compute mixture distribution
        beta = torch.tensor(self.beta, dtype=student_log_probs.dtype, device=student_log_probs.device)
        mixture_log_probs = torch.logsumexp(
            torch.stack([
                student_log_probs + torch.log(beta),
                teacher_log_probs + torch.log(1 - beta)
            ]),
            dim=0,
        )
        
        # Compute KL divergences
        kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs, reduction='none', log_target=True)
        kl_student = F.kl_div(mixture_log_probs, student_log_probs, reduction='none', log_target=True)
        
        # Compute JSD
        jsd = beta * kl_teacher + (1 - beta) * kl_student
        
        # Apply masking if labels provided
        if labels is not None:
            mask = labels != -100
            jsd = jsd[mask]
        
        # Apply reduction
        if reduction == "batchmean":
            return jsd.sum() / (mask.sum() if labels is not None else jsd.numel())
        elif reduction == "sum":
            return jsd.sum()
        else:
            return jsd.mean()

    def generate_on_policy_samples(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Generate on-policy samples using the current student model"""
        with torch.no_grad():
            model = self.student.module if hasattr(self.student, 'module') else self.student

            generated = model.generate(
                input_ids=inputs["prompt_ids"],
                attention_mask=inputs.get("prompt_attention_mask"),
                max_length=inputs["input_ids"].size(1),
                do_sample=True,
                temperature=self.temperature,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            
            # Create attention mask for generated tokens
            attention_mask = (generated != self.tokenizer.pad_token_id).long()
            
            return {
                "input_ids": generated,
                "attention_mask": attention_mask,
                "labels": generated.clone(),
            }

    def compute_loss(
        self,
        student_outputs,
        teacher_outputs,
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute GKD loss between student and teacher outputs"""
        losses = {}
        
        # Get logits and ensure same sequence length
        student_logits = student_outputs.logits[:, :-1, :]  # Remove last position
        teacher_logits = teacher_outputs.logits[:, :-1, :]  # Remove last position
        shifted_labels = labels[:, 1:]  # Shift labels right
        
        # Find minimum sequence length
        min_seq_len = min(student_logits.size(1), teacher_logits.size(1))
        
        # Truncate to same length
        student_logits = student_logits[:, :min_seq_len, :]
        teacher_logits = teacher_logits[:, :min_seq_len, :]
        shifted_labels = shifted_labels[:, :min_seq_len]
        
        # Compute GKD loss
        gkd_loss = self.generalized_jsd_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            labels=shifted_labels,
        )
        
        losses["gkd_loss"] = gkd_loss
        losses["total_loss"] = gkd_loss
        
        return losses

    def teacher_forward_first_half(self, batch):
        """Execute first half of teacher model on rank 0"""
        assert self.local_rank == 0, "First half forward should only run on rank 0"
        
        with torch.no_grad():
            # Get inputs
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            
            # Convert 2D attention mask to 4D
            bsz, seq_len = input_ids.shape

            dtype = self.teacher.model.embed_tokens.weight.dtype
            causal_mask = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=self.device, dtype=dtype), 
                diagonal=1
            )
            # Expand attention_mask to 4D: [batch_size, 1, seq_len, seq_len]
            attention_mask = attention_mask.to(dtype=dtype)
            attention_mask = attention_mask.view(bsz, 1, 1, seq_len)
            attention_mask = attention_mask.expand(-1, 1, seq_len, -1)
            attention_mask = attention_mask + causal_mask
            
            # Initial embeddings
            hidden_states = self.teacher.model.embed_tokens(input_ids)
            
            # Get position ids
            position_ids = torch.arange(seq_len, dtype=torch.long, device=self.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
            
            # Process first half of layers (0-43)
            for i in range(44):
                layer = self.teacher.model.layers[i]
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )
                hidden_states = layer_outputs[0]
            
            return hidden_states, attention_mask, position_ids

    def teacher_forward_second_half(self, hidden_states, attention_mask, position_ids):
        """Execute second half of teacher model on rank 1"""
        assert self.local_rank == 1, "Second half forward should only run on rank 1"
        
        with torch.no_grad():
            # Process second half of layers (44-79)
            for i in range(44, 80):
                layer = self.teacher.model.layers[i]
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )
                hidden_states = layer_outputs[0]
            
            # Final norm and head
            hidden_states = self.teacher.model.norm(hidden_states)
            logits = self.teacher.lm_head(hidden_states)
            
            return logits

    def train_step(self, batch, use_on_policy: bool) -> Dict[str, torch.Tensor]:
        if self.is_main:
            print(f"[Rank {self.rank}] Starting train_step")
        
        # Student forward pass (unchanged)
        if self.student is not None:
            student_outputs = self.student(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
        else:
            student_outputs = type('', (), {})()
            shape = (batch["input_ids"].size(0), batch["input_ids"].size(1), 152064)
            student_outputs.logits = torch.zeros(size=shape, device=self.device)

        dist.barrier()

        # Teacher forward pass using split methods
        if self.local_rank <= 1:
            if self.local_rank == 0:
                # First half on rank 0
                hidden_states, attention_mask, position_ids = self.teacher_forward_first_half(batch)
                
                # Convert to float16 for transfer
                hidden_states = hidden_states.to(torch.float16)

                print("SHAPES: ", hidden_states.shape, position_ids.shape, attention_mask.shape)
                
                # Send to rank 1
                dist.send(hidden_states, dst=1)
                dist.send(attention_mask, dst=1)
                dist.send(position_ids, dst=1)
                
                # Create dummy logits tensor
                teacher_outputs = type('', (), {})()
                teacher_outputs.logits = torch.zeros(
                    (batch["input_ids"].size(0), batch["input_ids"].size(1), 152064),
                    dtype=torch.float16,
                    device=self.device
                )
                
            else:  # local_rank == 1
                # Receive from rank 0
                hidden_states = torch.empty(
                    (batch["input_ids"].size(0), batch["input_ids"].size(1), 4096),
                    dtype=torch.float16,
                    device=self.device
                )
                attention_mask = torch.empty_like(batch["attention_mask"], device=self.device)
                position_ids = torch.empty_like(batch["input_ids"], device=self.device)
                
                dist.recv(hidden_states, src=0)
                dist.recv(attention_mask, src=0)
                dist.recv(position_ids, src=0)
                
                # Second half on rank 1
                logits = self.teacher_forward_second_half(hidden_states, attention_mask, position_ids)
                
                teacher_outputs = type('', (), {})()
                teacher_outputs.logits = logits.to(dtype=torch.float16)
                
                # Broadcast final logits
                shape_tensor = torch.tensor(teacher_outputs.logits.shape, dtype=torch.long, device=self.device)
                dist.broadcast(shape_tensor, src=1)
                dist.broadcast(teacher_outputs.logits, src=1)
        else:
            # Other ranks receive the logits
            shape_tensor = torch.zeros(3, dtype=torch.long, device=self.device)
            dist.broadcast(shape_tensor, src=1)
            
            teacher_outputs = type('', (), {})()
            teacher_outputs.logits = torch.empty(
                size=tuple(shape_tensor.tolist()),
                dtype=torch.float16,
                device=self.device
            )
            dist.broadcast(teacher_outputs.logits, src=1)

        dist.barrier()

        # Rest of train_step remains the same
        loss_dict = self.compute_loss(
            student_outputs=student_outputs,
            teacher_outputs=teacher_outputs,
            labels=batch["labels"],
        )
        loss_dict["total_loss"] /= self.gradient_accumulation_steps
        
        return loss_dict

    def train_epoch(self, dataloader, epoch: int) -> Dict[str, float]:
        print(f"[Rank {self.rank}] Starting epoch {epoch}")
        
        if self.student is not None:
            self.student.train()
            print(f"[Rank {self.rank}] Set student model to train mode")
            
        stats = {"total_loss": 0.0, "gkd_loss": 0.0}
        num_batches = len(dataloader)

        for batch_idx, batch in enumerate(dataloader):
            if batch_idx % 10 == 0:  # Print every 10 batches
                print(f"[Rank {self.rank}] Processing batch {batch_idx}/{num_batches}")
                
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Decide whether to use on-policy samples
            use_on_policy = (random.random() < self.lambda_gkd) if self.student is not None else False
            
            loss_dict = self.train_step(batch, use_on_policy)

            # Only do backward/update if we have a student
            if self.student is not None:
                print(f"[Rank {self.rank}] Starting backward pass")
                self.scaler.scale(loss_dict["total_loss"]).backward()

                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    print(f"[Rank {self.rank}] Optimizer step")
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)

            # Update stats
            for k, v in loss_dict.items():
                stats[k] += v.item()

        print(f"[Rank {self.rank}] Completed epoch {epoch}")
        return stats