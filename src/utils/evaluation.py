"""
Evaluation utilities for dog image matching.
Includes similarity search, re-ranking, and accuracy metrics.
"""
import torch
import numpy as np
from typing import List, Tuple, Dict
from sklearn.metrics.pairwise import cosine_similarity
import faiss


def compute_embeddings(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device
) -> Tuple[torch.Tensor, List[str]]:
    """
    Compute embeddings for all images in dataloader.
    
    Args:
        model: Trained model
        dataloader: DataLoader
        device: Device to run on
        
    Returns:
        (embeddings, dog_ids)
    """
    model.eval()
    embeddings = []
    dog_ids = []
    
    with torch.no_grad():
        for batch in dataloader:
            if 'frontal' in batch and 'lateral' in batch:
                # Dual-view model
                frontal = batch['frontal'].to(device)
                lateral = batch['lateral'].to(device)
                emb = model(frontal, lateral)
            else:
                # Single-view model
                images = batch['image'].to(device)
                emb = model(images)
            
            embeddings.append(emb.cpu())
            dog_ids.extend(batch['dog_id'])
    
    embeddings = torch.cat(embeddings, dim=0)
    return embeddings, dog_ids


def cosine_similarity_search(
    query_embedding: torch.Tensor,
    gallery_embeddings: torch.Tensor,
    top_k: int = 10
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Find top-k most similar embeddings using cosine similarity.
    
    Args:
        query_embedding: Query embedding [embedding_dim]
        gallery_embeddings: Gallery embeddings [N, embedding_dim]
        top_k: Number of top results to return
        
    Returns:
        (similarities, indices)
    """
    # Normalize
    query_norm = query_embedding / query_embedding.norm(p=2)
    gallery_norm = gallery_embeddings / gallery_embeddings.norm(p=2, dim=1, keepdim=True)
    
    # Compute cosine similarity
    similarities = torch.mm(query_norm.unsqueeze(0), gallery_norm.t()).squeeze(0)
    
    # Get top-k
    top_similarities, top_indices = torch.topk(similarities, k=min(top_k, len(similarities)))
    
    return top_similarities, top_indices


def faiss_search(
    query_embeddings: np.ndarray,
    gallery_embeddings: np.ndarray,
    top_k: int = 10,
    use_gpu: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fast similarity search using FAISS.
    
    Args:
        query_embeddings: Query embeddings [N_query, embedding_dim]
        gallery_embeddings: Gallery embeddings [N_gallery, embedding_dim]
        top_k: Number of top results
        use_gpu: Use GPU if available
        
    Returns:
        (distances, indices)
    """
    dimension = gallery_embeddings.shape[1]
    
    # Normalize for cosine similarity
    faiss.normalize_L2(gallery_embeddings)
    faiss.normalize_L2(query_embeddings)
    
    # Create index
    index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
    
    if use_gpu and faiss.get_num_gpus() > 0:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    
    # Add gallery embeddings
    index.add(gallery_embeddings.astype('float32'))
    
    # Search
    distances, indices = index.search(query_embeddings.astype('float32'), top_k)
    
    return distances, indices


def compute_accuracy_at_k(
    query_ids: List[str],
    gallery_ids: List[str],
    top_indices: np.ndarray,
    k_values: List[int] = [1, 5, 10]
) -> Dict[int, float]:
    """
    Compute accuracy@k metrics.
    
    Args:
        query_ids: Query dog IDs
        gallery_ids: Gallery dog IDs
        top_indices: Top-k indices for each query [N_query, k]
        k_values: List of k values to compute
        
    Returns:
        Dictionary of {k: accuracy}
    """
    accuracies = {}
    
    for k in k_values:
        if k > top_indices.shape[1]:
            continue
        
        correct = 0
        for i, query_id in enumerate(query_ids):
            # Check if correct dog is in top-k
            top_k_ids = [gallery_ids[idx] for idx in top_indices[i, :k]]
            if query_id in top_k_ids:
                correct += 1
        
        accuracies[k] = correct / len(query_ids)
    
    return accuracies


def re_rank_results(
    query_embedding: torch.Tensor,
    candidate_embeddings: torch.Tensor,
    initial_scores: torch.Tensor,
    method: str = 'reciprocal'
) -> torch.Tensor:
    """
    Re-rank search results using reciprocal rank fusion or other methods.
    
    Args:
        query_embedding: Query embedding
        candidate_embeddings: Candidate embeddings
        initial_scores: Initial similarity scores
        method: Re-ranking method ('reciprocal', 'rerank')
        
    Returns:
        Re-ranked scores
    """
    if method == 'reciprocal':
        # Reciprocal rank fusion
        ranks = torch.argsort(initial_scores, descending=True)
        reciprocal_ranks = 1.0 / (torch.arange(len(ranks), dtype=torch.float32) + 1)
        reranked_scores = torch.zeros_like(initial_scores)
        reranked_scores[ranks] = reciprocal_ranks
        return reranked_scores
    
    elif method == 'rerank':
        # Re-compute with more detailed comparison
        # Could use additional features or different distance metric
        return initial_scores  # Placeholder
    
    else:
        return initial_scores


def evaluate_model(
    model: torch.nn.Module,
    query_loader: torch.utils.data.DataLoader,
    gallery_loader: torch.utils.data.DataLoader,
    device: torch.device,
    top_k: int = 10,
    use_faiss: bool = True
) -> Dict:
    """
    Evaluate model on query and gallery sets.
    
    Args:
        model: Trained model
        query_loader: Query dataloader
        gallery_loader: Gallery dataloader
        device: Device
        top_k: Top-k for retrieval
        use_faiss: Use FAISS for fast search
        
    Returns:
        Evaluation metrics dictionary
    """
    # Compute embeddings
    query_embeddings, query_ids = compute_embeddings(model, query_loader, device)
    gallery_embeddings, gallery_ids = compute_embeddings(model, gallery_loader, device)
    
    # Convert to numpy for FAISS
    query_emb_np = query_embeddings.numpy()
    gallery_emb_np = gallery_embeddings.numpy()
    
    # Search
    if use_faiss:
        distances, indices = faiss_search(query_emb_np, gallery_emb_np, top_k=top_k)
        # Convert distances to similarities (FAISS returns inner product for normalized vectors)
        similarities = distances
    else:
        # Use cosine similarity
        similarities_list = []
        indices_list = []
        for i in range(len(query_embeddings)):
            sim, idx = cosine_similarity_search(query_embeddings[i], gallery_embeddings, top_k)
            similarities_list.append(sim.numpy())
            indices_list.append(idx.numpy())
        similarities = np.array(similarities_list)
        indices = np.array(indices_list)
    
    # Compute accuracy metrics
    accuracies = compute_accuracy_at_k(query_ids, gallery_ids, indices, k_values=[1, 5, 10])
    
    return {
        'accuracies': accuracies,
        'similarities': similarities,
        'indices': indices,
        'query_ids': query_ids,
        'gallery_ids': gallery_ids
    }

