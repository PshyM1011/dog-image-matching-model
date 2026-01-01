"""
Evaluation script for dual-view dog image matching model.
"""
import os
import argparse
import torch
from torch.utils.data import DataLoader

from src.model import DualViewFusionModel
from src.utils import DualViewDataset, create_dataloaders
from src.utils.evaluation import evaluate_model
from src.preprocessing import get_test_transforms


def main():
    parser = argparse.ArgumentParser(description='Evaluate dual-view dog image matching model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='data', help='Root data directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--top_k', type=int, default=10, help='Top-k for retrieval')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
    parser.add_argument('--use_faiss', action='store_true', help='Use FAISS for fast search')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device(args.device)
    print(f'Using device: {device}')
    
    # Load checkpoint
    print(f'Loading checkpoint from {args.checkpoint}...')
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Get model args
    model_args = checkpoint.get('args', {})
    embedding_dim = model_args.get('embedding_dim', 512)
    
    # Create model
    model = DualViewFusionModel(embedding_dim=embedding_dim).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print('Model loaded successfully!')
    
    # Create dataloaders
    print('Loading datasets...')
    test_dir = os.path.join(args.data_dir, 'test')
    
    # For evaluation, we need query and gallery sets
    # For simplicity, using test set as both (in practice, split appropriately)
    query_dataset = DualViewDataset(test_dir, transform=get_test_transforms())
    gallery_dataset = DualViewDataset(test_dir, transform=get_test_transforms())
    
    query_loader = DataLoader(
        query_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    gallery_loader = DataLoader(
        gallery_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    print(f'Query samples: {len(query_dataset)}')
    print(f'Gallery samples: {len(gallery_dataset)}')
    
    # Evaluate
    print('Evaluating model...')
    results = evaluate_model(
        model,
        query_loader,
        gallery_loader,
        device,
        top_k=args.top_k,
        use_faiss=args.use_faiss
    )
    
    # Print results
    print('\n' + '='*50)
    print('EVALUATION RESULTS')
    print('='*50)
    print(f'Accuracy@1:  {results["accuracies"].get(1, 0):.4f}')
    print(f'Accuracy@5:  {results["accuracies"].get(5, 0):.4f}')
    print(f'Accuracy@10: {results["accuracies"].get(10, 0):.4f}')
    print('='*50)
    
    # Save results
    import json
    output_file = 'evaluation_results.json'
    with open(output_file, 'w') as f:
        json.dump({
            'accuracies': results['accuracies'],
            'num_queries': len(results['query_ids']),
            'num_gallery': len(results['gallery_ids'])
        }, f, indent=2)
    
    print(f'\nResults saved to {output_file}')


if __name__ == '__main__':
    main()

