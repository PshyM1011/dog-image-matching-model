"""
Utility script to update gallery embeddings from a trained checkpoint.
Use this when you have a new checkpoint but don't want to retrain.

Usage:
    python src/update_gallery_embeddings.py --checkpoint checkpoints/best.pth --data_dir data
"""
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import torch
from torch.utils.data import DataLoader

from src.model import DualViewFusionModel
from src.utils import DualViewDataset
from src.utils.evaluation import compute_embeddings
from src.preprocessing import get_test_transforms


def main():
    parser = argparse.ArgumentParser(
        description='Update gallery embeddings from a trained checkpoint'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to trained model checkpoint (e.g., checkpoints/best.pth)'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data',
        help='Root data directory containing train folder (default: data)'
    )
    parser.add_argument(
        '--gallery_embeddings',
        type=str,
        default='gallery_embeddings.pt',
        help='Path to save gallery embeddings (default: gallery_embeddings.pt)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for computing embeddings (default: 32)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use: cpu or cuda (default: auto-detect)'
    )
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device)
    print(f'Using device: {device}')
    print()
    
    # Load checkpoint
    print(f'Loading checkpoint from {args.checkpoint}...')
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Get embedding dimension from checkpoint
    if 'embedding_dim' in checkpoint:
        embedding_dim = checkpoint['embedding_dim']
    elif 'args' in checkpoint and isinstance(checkpoint['args'], dict):
        embedding_dim = checkpoint['args'].get('embedding_dim', 512)
    else:
        embedding_dim = 512  # Default
    
    print(f'Embedding dimension: {embedding_dim}')
    
    # Create model
    print('Creating model...')
    model = DualViewFusionModel(embedding_dim=embedding_dim).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print('✅ Model loaded successfully!')
    print()
    
    # Create dataset and dataloader
    train_dir = os.path.join(args.data_dir, 'train')
    
    if not os.path.exists(train_dir):
        raise ValueError(f"Train directory not found: {train_dir}")
    
    print(f'Loading gallery dataset from {train_dir}...')
    gallery_dataset = DualViewDataset(train_dir, transform=get_test_transforms())
    gallery_loader = DataLoader(
        gallery_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0  # Set to 0 to avoid multiprocessing issues
    )
    
    print(f'Found {len(gallery_dataset)} gallery samples')
    print()
    
    # Compute embeddings
    print('Computing gallery embeddings...')
    gallery_embeddings, gallery_ids = compute_embeddings(
        model, gallery_loader, device, return_paths=False, dataset_name="gallery"
    )
    
    # Save embeddings
    print(f'\nSaving gallery embeddings to {args.gallery_embeddings}...')
    torch.save({
        'embeddings': gallery_embeddings,
        'ids': gallery_ids,
        'checkpoint_path': args.checkpoint,  # Store checkpoint path for reference
        'checkpoint_mtime': os.path.getmtime(args.checkpoint)  # Store checkpoint modification time
    }, args.gallery_embeddings)
    
    print(f'✅ Saved {len(gallery_ids)} gallery embeddings')
    print(f'   File: {args.gallery_embeddings}')
    print()
    print('🎉 Gallery embeddings updated successfully!')
    print('   You can now use match_dog.py with the updated embeddings.')


if __name__ == '__main__':
    main()

