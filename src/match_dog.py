"""
Dog Image Matching Script
==========================

This script allows you to match user-provided dog images (frontal and lateral views)
against a gallery of known dogs and get the top 5 most similar matches with percentages.

HOW TO USE:
-----------
1. Prepare your dog images:
   - You need TWO images of the dog:
     * One FRONTAL view (front-facing photo)
     * One LATERAL view (side-facing photo)
   - Images can be in JPG, PNG, or JPEG format
   - Place images in any folder (e.g., 'test_images/')

2. Run the script:
   python src/match_dog.py \
       --checkpoint checkpoints/best.pth \
       --frontal path/to/your/dog_front.jpg \
       --lateral path/to/your/dog_side.jpg \
       --data_dir data

3. The script will:
   - Load the trained model
   - Compute gallery embeddings from the train set (if not already saved)
   - Process your input images
   - Find top 5 matching dogs
   - Display results with matching percentages

OPTIONAL ARGUMENTS:
-------------------
--gallery_embeddings: Path to pre-computed gallery embeddings (saves time)
--top_k: Number of matches to return (default: 5)
--device: 'cpu' or 'cuda' (default: auto-detect)
"""
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import torch
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader

from src.model import DualViewFusionModel
from src.utils import DualViewDataset
from src.utils.evaluation import compute_embeddings, cosine_similarity_search
from src.preprocessing import get_test_transforms


def load_model(checkpoint_path: str, device: torch.device):
    """Load trained model from checkpoint."""
    print(f'Loading model from {checkpoint_path}...')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get embedding dimension from checkpoint
    model_args = checkpoint.get('args', {})
    embedding_dim = model_args.get('embedding_dim', 512)
    
    # Create and load model
    model = DualViewFusionModel(embedding_dim=embedding_dim).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f'Model loaded successfully! (embedding_dim={embedding_dim})')
    return model


def load_or_compute_gallery_embeddings(
    model: torch.nn.Module,
    data_dir: str,
    device: torch.device,
    gallery_embeddings_path: str = None,
    batch_size: int = 32,
    checkpoint_path: str = None,
    force_recompute: bool = False
):
    """
    Load pre-computed gallery embeddings or compute them from train set.
    
    Automatically recomputes if checkpoint is newer than saved embeddings.
    
    Args:
        model: Trained model
        data_dir: Root data directory
        device: Device to run on
        gallery_embeddings_path: Path to save/load embeddings
        batch_size: Batch size for computing embeddings
        checkpoint_path: Path to checkpoint file (to check if embeddings are outdated)
        force_recompute: Force recomputation even if embeddings exist
        
    Returns:
        (gallery_embeddings, gallery_ids)
    """
    # Default path for gallery embeddings
    if gallery_embeddings_path is None:
        gallery_embeddings_path = 'gallery_embeddings.pt'
    
    # Check if we should recompute
    should_recompute = force_recompute
    
    if not should_recompute and os.path.exists(gallery_embeddings_path) and checkpoint_path:
        # Check if checkpoint is newer than gallery embeddings
        try:
            checkpoint_mtime = os.path.getmtime(checkpoint_path)
            gallery_mtime = os.path.getmtime(gallery_embeddings_path)
            
            if checkpoint_mtime > gallery_mtime:
                from datetime import datetime
                checkpoint_time = datetime.fromtimestamp(checkpoint_mtime).strftime('%Y-%m-%d %H:%M:%S')
                gallery_time = datetime.fromtimestamp(gallery_mtime).strftime('%Y-%m-%d %H:%M:%S')
                
                print(f'⚠️  Checkpoint is newer than gallery embeddings!')
                print(f'   Checkpoint modified: {checkpoint_time}')
                print(f'   Gallery embeddings modified: {gallery_time}')
                print(f'   Recomputing gallery embeddings to match current model...')
                should_recompute = True
        except Exception as e:
            print(f'⚠️  Could not compare file timestamps: {e}')
            print(f'   Recomputing gallery embeddings to be safe...')
            should_recompute = True
    
    # Try to load pre-computed embeddings (if not forcing recompute)
    if not should_recompute and os.path.exists(gallery_embeddings_path):
        print(f'Loading pre-computed gallery embeddings from {gallery_embeddings_path}...')
        try:
            data = torch.load(gallery_embeddings_path, map_location=device)
            gallery_embeddings = data['embeddings']
            gallery_ids = data['ids']
            
            # Check if saved embeddings match the current checkpoint
            if checkpoint_path and 'checkpoint_path' in data:
                saved_checkpoint = data.get('checkpoint_path', '')
                if saved_checkpoint != checkpoint_path:
                    print(f'⚠️  Saved embeddings were computed with different checkpoint!')
                    print(f'   Saved checkpoint: {saved_checkpoint}')
                    print(f'   Current checkpoint: {checkpoint_path}')
                    print(f'   Recomputing gallery embeddings for current checkpoint...')
                    should_recompute = True
                elif 'checkpoint_mtime' in data and os.path.exists(checkpoint_path):
                    saved_mtime = data.get('checkpoint_mtime', 0)
                    current_mtime = os.path.getmtime(checkpoint_path)
                    if current_mtime > saved_mtime:
                        print(f'⚠️  Current checkpoint is newer than when embeddings were computed!')
                        print(f'   Recomputing gallery embeddings...')
                        should_recompute = True
            
            if not should_recompute:
                print(f'✅ Loaded {len(gallery_ids)} gallery embeddings')
                return gallery_embeddings, gallery_ids
        except Exception as e:
            print(f'⚠️  Error loading gallery embeddings: {e}')
            print(f'   Recomputing gallery embeddings...')
            should_recompute = True
    
    # Compute embeddings from train set
    print('Computing gallery embeddings from train set...')
    train_dir = os.path.join(data_dir, 'train')
    
    if not os.path.exists(train_dir):
        raise ValueError(f"Train directory not found: {train_dir}")
    
    # Create dataset and dataloader
    gallery_dataset = DualViewDataset(train_dir, transform=get_test_transforms())
    gallery_loader = DataLoader(
        gallery_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0  # Set to 0 to avoid multiprocessing issues
    )
    
    print(f'Computing embeddings for {len(gallery_dataset)} gallery samples...')
    gallery_embeddings, gallery_ids = compute_embeddings(
        model, gallery_loader, device, return_paths=False, dataset_name="gallery"
    )
    
    # Save embeddings for future use
    print(f'Saving gallery embeddings to {gallery_embeddings_path}...')
    save_data = {
        'embeddings': gallery_embeddings,
        'ids': gallery_ids
    }
    
    # Store checkpoint metadata if available
    if checkpoint_path and os.path.exists(checkpoint_path):
        save_data['checkpoint_path'] = checkpoint_path
        save_data['checkpoint_mtime'] = os.path.getmtime(checkpoint_path)
    
    torch.save(save_data, gallery_embeddings_path)
    
    print(f'Saved {len(gallery_ids)} gallery embeddings')
    return gallery_embeddings, gallery_ids


def preprocess_user_images(frontal_path: str, lateral_path: str, device: torch.device):
    """
    Load and preprocess user-provided dog images.
    
    Args:
        frontal_path: Path to frontal view image
        lateral_path: Path to lateral view image
        device: Device to run on
        
    Returns:
        (frontal_tensor, lateral_tensor)
    """
    transform = get_test_transforms()
    
    # Load and preprocess frontal image
    print(f'Loading frontal image: {frontal_path}')
    if not os.path.exists(frontal_path):
        raise FileNotFoundError(f"Frontal image not found: {frontal_path}")
    
    frontal_img = Image.open(frontal_path).convert('RGB')
    frontal_tensor = transform(frontal_img).unsqueeze(0).to(device)  # Add batch dimension
    
    # Load and preprocess lateral image
    print(f'Loading lateral image: {lateral_path}')
    if not os.path.exists(lateral_path):
        raise FileNotFoundError(f"Lateral image not found: {lateral_path}")
    
    lateral_img = Image.open(lateral_path).convert('RGB')
    lateral_tensor = transform(lateral_img).unsqueeze(0).to(device)  # Add batch dimension
    
    return frontal_tensor, lateral_tensor


def find_matching_dogs(
    model: torch.nn.Module,
    frontal_tensor: torch.Tensor,
    lateral_tensor: torch.Tensor,
    gallery_embeddings: torch.Tensor,
    gallery_ids: list,
    device: torch.device,
    top_k: int = 5
):
    """
    Find top-k matching dogs for user-provided images.
    
    Args:
        model: Trained model
        frontal_tensor: Preprocessed frontal image tensor
        lateral_tensor: Preprocessed lateral image tensor
        gallery_embeddings: Gallery embeddings
        gallery_ids: Gallery dog IDs
        device: Device
        top_k: Number of top matches to return
        
    Returns:
        List of (dog_id, similarity_score, percentage) tuples
    """
    print('Computing embedding for input images...')
    
    # Compute query embedding
    with torch.no_grad():
        query_embedding = model(frontal_tensor, lateral_tensor)
    
    print('Searching for matching dogs in gallery...')
    
    # Search in gallery using cosine similarity
    similarities, indices = cosine_similarity_search(
        query_embedding.squeeze(0),
        gallery_embeddings,
        top_k=top_k
    )
    
    # Format results with percentages
    results = []
    for sim, idx in zip(similarities.cpu().numpy(), indices.cpu().numpy()):
        dog_id = gallery_ids[idx]
        similarity_score = float(sim)
        # Cosine similarity ranges from -1 to 1, convert to percentage (0-100%)
        # Normalize: (sim + 1) / 2 * 100 gives 0-100% range
        percentage = ((similarity_score + 1) / 2) * 100
        results.append((dog_id, similarity_score, percentage))
    
    return results


def print_results(results: list):
    """Print matching results in a user-friendly format."""
    print('\n' + '='*70)
    print('DOG MATCHING RESULTS')
    print('='*70)
    print(f'\nTop {len(results)} Most Similar Dogs Found:\n')
    
    for i, (dog_id, similarity, percentage) in enumerate(results, 1):
        print(f'{i}. Dog ID: {dog_id}')
        print(f'   Similarity Score: {similarity:.4f}')
        print(f'   Match Percentage: {percentage:.2f}%')
        print()
    
    print('='*70)
    print('\nNote: Match percentage is calculated from cosine similarity.')
    print('Higher percentage indicates higher similarity to the input dog.\n')


def main():
    parser = argparse.ArgumentParser(
        description='Match user-provided dog images with gallery database',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage:
  python src/match_dog.py --checkpoint checkpoints/best.pth \\
      --frontal my_dog_front.jpg --lateral my_dog_side.jpg --data_dir data
  
  # With pre-computed gallery embeddings (faster):
  python src/match_dog.py --checkpoint checkpoints/best.pth \\
      --frontal my_dog_front.jpg --lateral my_dog_side.jpg \\
      --data_dir data --gallery_embeddings gallery_embeddings.pt
  
  # Get top 10 matches:
  python src/match_dog.py --checkpoint checkpoints/best.pth \\
      --frontal my_dog_front.jpg --lateral my_dog_side.jpg \\
      --data_dir data --top_k 10
        """
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to trained model checkpoint (e.g., checkpoints/best.pth)'
    )
    parser.add_argument(
        '--frontal',
        type=str,
        required=True,
        help='Path to frontal view image of the dog (front-facing photo)'
    )
    parser.add_argument(
        '--lateral',
        type=str,
        required=True,
        help='Path to lateral view image of the dog (side-facing photo)'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data',
        help='Root data directory containing train/val/test folders (default: data)'
    )
    parser.add_argument(
        '--gallery_embeddings',
        type=str,
        default=None,
        help='Path to pre-computed gallery embeddings file (saves time if provided)'
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=5,
        help='Number of top matching dogs to return (default: 5)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use: cpu or cuda (default: auto-detect)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for computing gallery embeddings (default: 32)'
    )
    parser.add_argument(
        '--recompute_gallery',
        action='store_true',
        help='Force recomputation of gallery embeddings (ignores saved file)'
    )
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device)
    print(f'Using device: {device}')
    print()
    
    # Load model
    model = load_model(args.checkpoint, device)
    print()
    
    # Load or compute gallery embeddings
    gallery_embeddings, gallery_ids = load_or_compute_gallery_embeddings(
        model,
        args.data_dir,
        device,
        args.gallery_embeddings,
        args.batch_size,
        checkpoint_path=args.checkpoint,  # Pass checkpoint path to check timestamps
        force_recompute=args.recompute_gallery  # Allow forcing recomputation
    )
    print()
    
    # Preprocess user images
    frontal_tensor, lateral_tensor = preprocess_user_images(
        args.frontal,
        args.lateral,
        device
    )
    print()
    
    # Find matching dogs
    results = find_matching_dogs(
        model,
        frontal_tensor,
        lateral_tensor,
        gallery_embeddings,
        gallery_ids,
        device,
        top_k=args.top_k
    )
    
    # Print results
    print_results(results)
    
    return results


if __name__ == '__main__':
    main()

