"""
Fine-tuning utilities for recovering performance after pruning.

This module provides training loops and utilities for fine-tuning pruned models.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Callable
from pathlib import Path
import json
from tqdm import tqdm


class PrunedModelTrainer:
    """
    Trainer for fine-tuning pruned models.
    
    Implements training loops with various optimizers and learning rate schedules
    to recover performance after pruning.
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize trainer.
        
        Args:
            model: Pruned model to fine-tune
            tokenizer: Tokenizer for the model
            device: Device to train on
        """
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.training_history: List[Dict] = []
        
    def prepare_dataset(
        self,
        texts: List[str],
        max_length: int = 512,
        batch_size: int = 4
    ) -> DataLoader:
        """
        Prepare dataset from list of texts.
        
        Args:
            texts: List of training texts
            max_length: Maximum sequence length
            batch_size: Batch size for training
        
        Returns:
            DataLoader for training
        """
        class TextDataset(Dataset):
            def __init__(self, texts, tokenizer, max_length):
                self.texts = texts
                self.tokenizer = tokenizer
                self.max_length = max_length
            
            def __len__(self):
                return len(self.texts)
            
            def __getitem__(self, idx):
                text = self.texts[idx]
                # For causal LM, input and target are the same
                encoded = self.tokenizer(
                    text,
                    max_length=self.max_length,
                    truncation=True,
                    padding='max_length',
                    return_tensors='pt'
                )
                return {
                    'input_ids': encoded['input_ids'].squeeze(),
                    'attention_mask': encoded['attention_mask'].squeeze()
                }
        
        dataset = TextDataset(texts, self.tokenizer, max_length)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: Optional[Callable] = None,
        max_grad_norm: float = 1.0
    ) -> float:
        """
        Train for one epoch.
        
        Args:
            dataloader: DataLoader for training data
            optimizer: Optimizer
            loss_fn: Loss function (default: cross-entropy for language modeling)
            max_grad_norm: Maximum gradient norm for clipping
        
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        
        for batch in tqdm(dataloader, desc="Training"):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # Forward pass
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            
            # Shift for language modeling (predict next token)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            
            # Compute loss
            loss = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def evaluate(
        self,
        dataloader: DataLoader,
        loss_fn: Optional[Callable] = None
    ) -> Dict[str, float]:
        """
        Evaluate model on validation data.
        
        Args:
            dataloader: DataLoader for validation data
            loss_fn: Loss function
        
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                
                loss = loss_fn(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return {
            'loss': avg_loss,
            'perplexity': torch.exp(torch.tensor(avg_loss)).item()
        }
    
    def fine_tune(
        self,
        train_texts: List[str],
        val_texts: Optional[List[str]] = None,
        num_epochs: int = 3,
        learning_rate: float = 2e-5,
        batch_size: int = 4,
        max_length: int = 512,
        warmup_steps: int = 100,
        save_dir: Optional[str] = None,
        save_every: int = 1
    ) -> Dict[str, List[float]]:
        """
        Fine-tune the pruned model.
        
        Args:
            train_texts: List of training texts
            val_texts: Optional list of validation texts
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            batch_size: Batch size
            max_length: Maximum sequence length
            warmup_steps: Number of warmup steps for learning rate
            save_dir: Directory to save checkpoints
            save_every: Save checkpoint every N epochs
        
        Returns:
            Dictionary with training history
        """
        # Prepare data
        train_loader = self.prepare_dataset(train_texts, max_length, batch_size)
        val_loader = None
        if val_texts:
            val_loader = self.prepare_dataset(val_texts, max_length, batch_size)
        
        # Setup optimizer with learning rate schedule
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # Learning rate scheduler with warmup
        total_steps = len(train_loader) * num_epochs
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps
        )
        
        # Training loop
        train_losses = []
        val_losses = []
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_loss = self.train_epoch(train_loader, optimizer)
            train_losses.append(train_loss)
            print(f"Train Loss: {train_loss:.4f}")
            
            # Validate
            if val_loader:
                val_metrics = self.evaluate(val_loader)
                val_losses.append(val_metrics['loss'])
                print(f"Val Loss: {val_metrics['loss']:.4f}")
                print(f"Val Perplexity: {val_metrics['perplexity']:.2f}")
            
            # Update learning rate
            if epoch < warmup_steps // len(train_loader):
                scheduler.step()
            
            # Save checkpoint
            if save_dir and (epoch + 1) % save_every == 0:
                checkpoint_path = Path(save_dir) / f"checkpoint_epoch_{epoch + 1}.pt"
                self.save_checkpoint(checkpoint_path, epoch, train_loss)
        
        self.training_history = {
            'train_loss': train_losses,
            'val_loss': val_losses if val_losses else None
        }
        
        return self.training_history
    
    def save_checkpoint(
        self,
        filepath: str,
        epoch: int,
        loss: float
    ):
        """Save training checkpoint."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'loss': loss,
            'training_history': self.training_history
        }, filepath)
        print(f"Saved checkpoint to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load training checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.training_history = checkpoint.get('training_history', [])
        print(f"Loaded checkpoint from {filepath}")


def prepare_training_data_from_jsonl(
    jsonl_file: str,
    text_field: str = "response"
) -> List[str]:
    """
    Prepare training texts from JSONL file.
    
    Args:
        jsonl_file: Path to JSONL file
        text_field: Field name containing text to train on
    
    Returns:
        List of training texts
    """
    texts = []
    with open(jsonl_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            if text_field in data:
                texts.append(data[text_field])
    return texts

