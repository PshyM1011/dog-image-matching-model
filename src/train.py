"""
Training script for dual-view dog image matching model.
"""
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import json

from src.model import DualViewFusionModel, CombinedLoss, HardTripletLoss
from src.utils import DualViewDataset, create_dataloaders
from src.preprocessing import get_train_transforms, get_val_transforms


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    id_to_label: dict
) -> dict:
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    num_batches = 0
    skipped_batches = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch_idx, batch in enumerate(pbar):
        frontal = batch['frontal'].to(device)
        lateral = batch['lateral'].to(device)
        dog_ids = batch['dog_id']
        
        # Forward pass
        embeddings = model(frontal, lateral)
        
        # Create labels using GLOBAL mapping (FIX: consistent labels)
        labels = []
        for dog_id in dog_ids:
            if dog_id in id_to_label:
                labels.append(id_to_label[dog_id])
            else:
                # If dog_id not in training set, skip this batch
                skipped_batches += 1
                break
        else:
            # All dog_ids found, create tensor
            labels = torch.tensor(labels, device=device, dtype=torch.long)
            
            # Check if batch has at least 2 different classes (needed for triplet loss)
            unique_labels = torch.unique(labels)
            if len(unique_labels) < 2:
                # Skip batches with only one class (can't compute triplet loss)
                skipped_batches += 1
                continue
            
            # Compute loss
            if isinstance(criterion, CombinedLoss):
                loss, loss_dict = criterion(embeddings, labels)
            elif isinstance(criterion, HardTripletLoss):
                loss = criterion(embeddings, labels)
                loss_dict = {'total': loss, 'triplet': loss}
            else:
                loss = torch.tensor(0.0, device=device, requires_grad=True)
                loss_dict = {'total': loss}
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    if skipped_batches > 0:
        print(f'  Skipped {skipped_batches} batches (single class or unknown dog_id)')
    
    avg_loss = running_loss / num_batches if num_batches > 0 else 0.0
    return {'loss': avg_loss}


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    id_to_label: dict
) -> dict:
    """Validate model."""
    model.eval()
    running_loss = 0.0
    num_batches = 0
    skipped_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validating'):
            frontal = batch['frontal'].to(device)
            lateral = batch['lateral'].to(device)
            dog_ids = batch['dog_id']
            
            embeddings = model(frontal, lateral)
            
            # Create labels using GLOBAL mapping (FIX: consistent labels)
            labels = []
            for dog_id in dog_ids:
                if dog_id in id_to_label:
                    labels.append(id_to_label[dog_id])
                else:
                    # If dog_id not in training set, skip this batch
                    skipped_batches += 1
                    break
            else:
                # All dog_ids found, create tensor
                labels = torch.tensor(labels, device=device, dtype=torch.long)
                
                # Check if batch has at least 2 different classes
                unique_labels = torch.unique(labels)
                if len(unique_labels) < 2:
                    skipped_batches += 1
                    continue
                
                if isinstance(criterion, CombinedLoss):
                    loss, loss_dict = criterion(embeddings, labels)
                elif isinstance(criterion, HardTripletLoss):
                    loss = criterion(embeddings, labels)
                else:
                    loss = torch.tensor(0.0, device=device)
                
                running_loss += loss.item()
                num_batches += 1
    
    if skipped_batches > 0:
        print(f'  Skipped {skipped_batches} validation batches (single class or unknown dog_id)')
    
    avg_loss = running_loss / num_batches if num_batches > 0 else 0.0
    return {'loss': avg_loss}


def main():
    parser = argparse.ArgumentParser(description='Train dual-view dog image matching model')
    parser.add_argument('--data_dir', type=str, default='data', help='Root data directory')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--embedding_dim', type=int, default=512, help='Embedding dimension')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--use_combined_loss', action='store_true', help='Use combined triplet + arcface loss')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Device
    device = torch.device(args.device)
    print(f'Using device: {device}')
    
    # Create dataloaders
    print('Loading datasets...')
    train_loader, val_loader = create_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_dual_view=True
    )
    
    print(f'Train samples: {len(train_loader.dataset)}')
    print(f'Val samples: {len(val_loader.dataset)}')
    
    # Create global label mapping (FIX: consistent labels across all batches)
    print('Creating label mapping...')
    all_train_dog_ids = set()
    for sample in train_loader.dataset.samples:
        all_train_dog_ids.add(sample['dog_id'])
    
    all_train_dog_ids = sorted(list(all_train_dog_ids))
    global_id_to_label = {dog_id: idx for idx, dog_id in enumerate(all_train_dog_ids)}
    num_classes = len(all_train_dog_ids)
    
    print(f'Found {num_classes} unique dogs in training set')
    print(f'Label mapping: {dict(list(global_id_to_label.items())[:5])}...')  # Show first 5
    
    # Create model
    print('Creating model...')
    model = DualViewFusionModel(embedding_dim=args.embedding_dim).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f'Model parameters: {num_params:,}')
    
    # Create loss function
    if args.use_combined_loss:
        # Use actual number of classes from dataset
        criterion = CombinedLoss(
            embedding_dim=args.embedding_dim,
            num_classes=num_classes
        ).to(device)
    else:
        criterion = HardTripletLoss(margin=1.0).to(device)
    
    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        print(f'Resuming from {args.resume}...')
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    
    # Training loop
    print('Starting training...')
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(start_epoch, args.epochs):
        print(f'\nEpoch {epoch+1}/{args.epochs}')
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch, global_id_to_label)
        history['train_loss'].append(train_metrics['loss'])
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device, global_id_to_label)
        history['val_loss'].append(val_metrics['loss'])
        
        # Update learning rate
        scheduler.step()
        
        # Save checkpoint
        is_best = val_metrics['loss'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['loss']
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_metrics['loss'],
            'val_loss': val_metrics['loss'],
            'best_val_loss': best_val_loss,
            'args': vars(args)
        }
        
        # Save latest
        torch.save(checkpoint, os.path.join(args.save_dir, 'latest.pth'))
        
        # Save best
        if is_best:
            torch.save(checkpoint, os.path.join(args.save_dir, 'best.pth'))
            print(f'New best model saved! Val loss: {val_metrics["loss"]:.4f}')
        
        # Save history
        with open(os.path.join(args.save_dir, 'history.json'), 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f'Train loss: {train_metrics["loss"]:.4f}, Val loss: {val_metrics["loss"]:.4f}')
    
    print('\nTraining complete!')
    print(f'Best validation loss: {best_val_loss:.4f}')


if __name__ == '__main__':
    main()

