"""
Inference script for matching a found dog with database.
"""
import os
import argparse
import torch
from PIL import Image
import numpy as np
from pathlib import Path

from src.model import DualViewFusionModel
from src.preprocessing import get_test_transforms
from src.utils.evaluation import cosine_similarity_search, faiss_search
from src.detector import DogDetector


def load_model(checkpoint_path: str, device: torch.device, embedding_dim: int = 512):
    """Load trained model."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = DualViewFusionModel(embedding_dim=embedding_dim).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def preprocess_image(image_path: str, transform, detector: DogDetector = None):
    """Load and preprocess image."""
    # Load image
    img = Image.open(image_path).convert('RGB')
    
    # Detect and crop dog if detector provided
    if detector:
        cropped = detector.crop_dog(image_path)
        if cropped:
            img = cropped
    
    # Apply transforms
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
    return img_tensor


def match_dog(
    frontal_path: str,
    lateral_path: str,
    model: torch.nn.Module,
    gallery_embeddings: torch.Tensor,
    gallery_ids: list,
    device: torch.device,
    top_k: int = 10,
    detector: DogDetector = None
):
    """
    Match a found dog with gallery database.
    
    Args:
        frontal_path: Path to frontal view image
        lateral_path: Path to lateral view image
        model: Trained model
        gallery_embeddings: Pre-computed gallery embeddings
        gallery_ids: Gallery dog IDs
        device: Device
        top_k: Number of top matches to return
        detector: Optional dog detector for cropping
        
    Returns:
        List of (dog_id, similarity_score) tuples
    """
    transform = get_test_transforms()
    
    # Preprocess images
    frontal_tensor = preprocess_image(frontal_path, transform, detector).to(device)
    lateral_tensor = preprocess_image(lateral_path, transform, detector).to(device)
    
    # Get query embedding
    with torch.no_grad():
        query_embedding = model(frontal_tensor, lateral_tensor)
    
    # Search in gallery
    similarities, indices = cosine_similarity_search(
        query_embedding.squeeze(0),
        gallery_embeddings,
        top_k=top_k
    )
    
    # Format results
    results = []
    for sim, idx in zip(similarities.cpu().numpy(), indices.cpu().numpy()):
        dog_id = gallery_ids[idx]
        results.append((dog_id, float(sim)))
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Match a found dog with database')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--frontal', type=str, required=True, help='Path to frontal view image')
    parser.add_argument('--lateral', type=str, required=True, help='Path to lateral view image')
    parser.add_argument('--gallery_embeddings', type=str, help='Path to saved gallery embeddings (.pt file)')
    parser.add_argument('--gallery_dir', type=str, help='Directory with gallery images (will compute embeddings)')
    parser.add_argument('--top_k', type=int, default=10, help='Number of top matches')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
    parser.add_argument('--use_detector', action='store_true', help='Use dog detector for cropping')
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    # Load model
    print('Loading model...')
    checkpoint = torch.load(args.checkpoint, map_location=device)
    embedding_dim = checkpoint.get('args', {}).get('embedding_dim', 512)
    model = load_model(args.checkpoint, device, embedding_dim)
    
    # Load or compute gallery embeddings
    if args.gallery_embeddings and os.path.exists(args.gallery_embeddings):
        print(f'Loading gallery embeddings from {args.gallery_embeddings}...')
        data = torch.load(args.gallery_embeddings, map_location=device)
        gallery_embeddings = data['embeddings']
        gallery_ids = data['ids']
    elif args.gallery_dir:
        print('Computing gallery embeddings...')
        # This would require loading gallery dataset and computing embeddings
        # For now, placeholder
        raise NotImplementedError("Gallery embedding computation from directory not yet implemented")
    else:
        raise ValueError("Must provide either --gallery_embeddings or --gallery_dir")
    
    # Optional detector
    detector = None
    if args.use_detector:
        print('Initializing dog detector...')
        detector = DogDetector()
    
    # Match
    print('Matching dog...')
    results = match_dog(
        args.frontal,
        args.lateral,
        model,
        gallery_embeddings,
        gallery_ids,
        device,
        top_k=args.top_k,
        detector=detector
    )
    
    # Print results
    print('\n' + '='*50)
    print('MATCHING RESULTS')
    print('='*50)
    for i, (dog_id, similarity) in enumerate(results, 1):
        print(f'{i}. Dog ID: {dog_id}, Similarity: {similarity:.4f}')
    print('='*50)
    
    return results


if __name__ == '__main__':
    main()

