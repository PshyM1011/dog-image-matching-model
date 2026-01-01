"""
Example usage script for dual-view dog image matching model.
This demonstrates the basic workflow for training, evaluation, and inference.
"""
import torch
from torch.utils.data import DataLoader
from src.model import DualViewFusionModel
from src.preprocessing import get_test_transforms
from src.utils import DualViewDataset
from src.utils.evaluation import compute_embeddings, cosine_similarity_search


def example_load_model(checkpoint_path='checkpoints/best.pth'):
    """Example: Load a trained model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get embedding dimension
    embedding_dim = checkpoint.get('args', {}).get('embedding_dim', 512)
    
    # Create and load model
    model = DualViewFusionModel(embedding_dim=embedding_dim).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print('Model loaded successfully!')
    return model, device


def example_compute_gallery_embeddings(model, device, gallery_dir='data/test'):
    """Example: Compute embeddings for gallery images."""
    # Load gallery dataset
    gallery_dataset = DualViewDataset(gallery_dir, transform=get_test_transforms())
    gallery_loader = DataLoader(gallery_dataset, batch_size=32, shuffle=False)
    
    # Compute embeddings
    embeddings, dog_ids = compute_embeddings(model, gallery_loader, device)
    
    print(f'Gallery embeddings shape: {embeddings.shape}')
    print(f'Number of gallery images: {len(dog_ids)}')
    
    # Save for later use
    torch.save({'embeddings': embeddings, 'ids': dog_ids}, 'gallery_embeddings.pt')
    print('Gallery embeddings saved to gallery_embeddings.pt')
    
    return embeddings, dog_ids


def example_match_dog(model, device, frontal_path, lateral_path, gallery_embeddings, gallery_ids, top_k=10):
    """Example: Match a found dog with gallery."""
    from PIL import Image
    
    transform = get_test_transforms()
    
    # Load and preprocess images
    frontal_img = Image.open(frontal_path).convert('RGB')
    lateral_img = Image.open(lateral_path).convert('RGB')
    
    frontal_tensor = transform(frontal_img).unsqueeze(0).to(device)
    lateral_tensor = transform(lateral_img).unsqueeze(0).to(device)
    
    # Compute query embedding
    with torch.no_grad():
        query_embedding = model(frontal_tensor, lateral_tensor)
    
    # Search for matches
    similarities, indices = cosine_similarity_search(
        query_embedding.squeeze(0),
        gallery_embeddings,
        top_k=top_k
    )
    
    # Display results
    print('\nTop {} Matches:'.format(top_k))
    print('=' * 50)
    for i, (sim, idx) in enumerate(zip(similarities.cpu().numpy(), indices.cpu().numpy()), 1):
        dog_id = gallery_ids[idx]
        print(f'{i}. Dog ID: {dog_id}, Similarity: {sim:.4f}')
    
    return similarities, indices


if __name__ == '__main__':
    print("=" * 60)
    print("Dual-View Dog Image Matching - Example Usage")
    print("=" * 60)
    
    # Example 1: Load model
    print("\n1. Loading model...")
    model, device = example_load_model()
    
    # Example 2: Compute gallery embeddings
    print("\n2. Computing gallery embeddings...")
    gallery_embeddings, gallery_ids = example_compute_gallery_embeddings(model, device)
    
    # Example 3: Match a dog (update paths with your images)
    print("\n3. Matching dog...")
    print("(Update frontal_path and lateral_path with your image paths)")
    # Uncomment and update paths:
    # frontal_path = 'path/to/frontal_image.jpg'
    # lateral_path = 'path/to/lateral_image.jpg'
    # example_match_dog(model, device, frontal_path, lateral_path, gallery_embeddings, gallery_ids)

