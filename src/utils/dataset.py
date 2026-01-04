"""
Dataset loader for dual-view dog images.
Supports both single-view and dual-view (frontal + lateral) images.
"""
import os
import sys
from pathlib import Path
from typing import Optional, Tuple, List
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

# Add project root to Python path if not already added
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.preprocessing import get_train_transforms, get_val_transforms, get_test_transforms


class DogDataset(Dataset):
    """
    Dataset for dog images organized by dog ID.
    
    Expected structure:
    data/train/
        dog1/
            dog1_front_1.jpg
            dog1_front_2.jpg
            dog1_side_1.jpg
            dog1_side_2.jpg
        dog2/
            dog2_front.jpg
            dog2_side.jpg
    """
    
    def __init__(
        self,
        data_dir: str,
        view_type: str = 'both',  # 'frontal', 'lateral', 'both'
        transform=None,
        return_view_type: bool = False
    ):
        """
        Initialize dataset.
        
        Args:
            data_dir: Root directory containing dog folders
            view_type: Which view(s) to load ('frontal', 'lateral', 'both')
            transform: Image transforms
            return_view_type: Whether to return view type label
        """
        self.data_dir = Path(data_dir)
        self.view_type = view_type
        self.transform = transform
        self.return_view_type = return_view_type
        
        # Load all image paths
        self.samples = self._load_samples()
    
    def _load_samples(self) -> List[dict]:
        """Load all image paths and labels."""
        samples = []
        
        if not self.data_dir.exists():
            print(f"Warning: Data directory {self.data_dir} does not exist")
            return samples
        
        # Iterate through dog folders
        for dog_folder in sorted(self.data_dir.iterdir()):
            if not dog_folder.is_dir():
                continue
            
            dog_id = dog_folder.name
            
            # Find all images in this folder
            images = list(dog_folder.glob('*.jpg')) + list(dog_folder.glob('*.png'))
            
            for img_path in images:
                # Determine view type from filename
                filename_lower = img_path.name.lower()
                
                if 'front' in filename_lower or 'frontal' in filename_lower:
                    img_view = 'frontal'
                elif 'side' in filename_lower or 'lateral' in filename_lower:
                    img_view = 'lateral'
                else:
                    # Default: try to infer from position in list
                    img_view = 'frontal' if len(samples) % 2 == 0 else 'lateral'
                
                # Filter by view type
                if self.view_type == 'both' or self.view_type == img_view:
                    samples.append({
                        'path': str(img_path),
                        'dog_id': dog_id,
                        'view': img_view
                    })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> dict:
        """Get a sample."""
        sample = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(sample['path']).convert('RGB')
        except Exception as e:
            print(f"Error loading image {sample['path']}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), color='black')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        result = {
            'image': image,
            'dog_id': sample['dog_id'],
            'path': sample['path']
        }
        
        if self.return_view_type:
            result['view'] = sample['view']
        
        return result


class DualViewDataset(Dataset):
    """
    Dataset that pairs frontal and lateral views of the same dog.
    """
    
    def __init__(
        self,
        data_dir: str,
        transform=None,
        allow_single_view: bool = True
    ):
        """
        Initialize dual-view dataset.
        
        Args:
            data_dir: Root directory containing dog folders
            transform: Image transforms
            allow_single_view: If True, use same image for both views if only one available
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.allow_single_view = allow_single_view
        
        # Load paired samples
        self.samples = self._load_paired_samples()
    
    def _load_paired_samples(self) -> List[dict]:
        """Load paired frontal and lateral images."""
        samples = []
        
        if not self.data_dir.exists():
            print(f"Warning: Data directory {self.data_dir} does not exist")
            return samples
        
        # Iterate through dog folders
        for dog_folder in sorted(self.data_dir.iterdir()):
            if not dog_folder.is_dir():
                continue
            
            dog_id = dog_folder.name
            
            # Separate frontal and lateral images
            frontal_images = []
            lateral_images = []
            
            images = list(dog_folder.glob('*.jpg')) + list(dog_folder.glob('*.png'))
            
            for img_path in images:
                filename_lower = img_path.name.lower()
                
                if 'front' in filename_lower or 'frontal' in filename_lower:
                    frontal_images.append(str(img_path))
                elif 'side' in filename_lower or 'lateral' in filename_lower:
                    lateral_images.append(str(img_path))
            
            # Create pairs
            if frontal_images and lateral_images:
                # Create ALL possible pairs (not just first of each)
                # This allows multiple samples per dog for better evaluation
                for front_path in frontal_images:
                    for side_path in lateral_images:
                        samples.append({
                            'frontal_path': front_path,
                            'lateral_path': side_path,
                            'dog_id': dog_id
                        })
            elif self.allow_single_view:
                # Use same image for both views if only one type available
                if frontal_images:
                    path = frontal_images[0]
                    samples.append({
                        'frontal_path': path,
                        'lateral_path': path,
                        'dog_id': dog_id
                    })
                elif lateral_images:
                    path = lateral_images[0]
                    samples.append({
                        'frontal_path': path,
                        'lateral_path': path,
                        'dog_id': dog_id
                    })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> dict:
        """Get a paired sample."""
        sample = self.samples[idx]
        
        # Load frontal image
        try:
            frontal_img = Image.open(sample['frontal_path']).convert('RGB')
        except Exception as e:
            print(f"Error loading frontal image {sample['frontal_path']}: {e}")
            frontal_img = Image.new('RGB', (224, 224), color='black')
        
        # Load lateral image
        try:
            lateral_img = Image.open(sample['lateral_path']).convert('RGB')
        except Exception as e:
            print(f"Error loading lateral image {sample['lateral_path']}: {e}")
            lateral_img = Image.new('RGB', (224, 224), color='black')
        
        # Apply transforms
        if self.transform:
            frontal_img = self.transform(frontal_img)
            lateral_img = self.transform(lateral_img)
        
        return {
            'frontal': frontal_img,
            'lateral': lateral_img,
            'dog_id': sample['dog_id'],
            'frontal_path': sample['frontal_path'],
            'lateral_path': sample['lateral_path']
        }


class TripletDataset(Dataset):
    """
    Dataset that generates triplets (anchor, positive, negative) for metric learning.
    """
    
    def __init__(
        self,
        data_dir: str,
        transform=None,
        view_type: str = 'both'
    ):
        """
        Initialize triplet dataset.
        
        Args:
            data_dir: Root directory containing dog folders
            transform: Image transforms
            view_type: 'frontal', 'lateral', or 'both'
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.view_type = view_type
        
        # Load all images grouped by dog_id
        self.dog_images = self._load_dog_images()
        self.dog_ids = list(self.dog_images.keys())
    
    def _load_dog_images(self) -> dict:
        """Load images grouped by dog ID."""
        dog_images = {}
        
        if not self.data_dir.exists():
            return dog_images
        
        for dog_folder in sorted(self.data_dir.iterdir()):
            if not dog_folder.is_dir():
                continue
            
            dog_id = dog_folder.name
            images = []
            
            for img_path in dog_folder.glob('*.jpg'):
                images.append(str(img_path))
            for img_path in dog_folder.glob('*.png'):
                images.append(str(img_path))
            
            if images:
                dog_images[dog_id] = images
        
        return dog_images
    
    def __len__(self) -> int:
        return len(self.dog_ids)
    
    def __getitem__(self, idx: int) -> dict:
        """Get a triplet."""
        # Anchor and positive: same dog
        anchor_dog_id = self.dog_ids[idx]
        anchor_images = self.dog_images[anchor_dog_id]
        
        # Select anchor and positive
        anchor_path = np.random.choice(anchor_images)
        positive_path = np.random.choice(anchor_images)
        
        # Negative: different dog
        negative_dog_id = anchor_dog_id
        while negative_dog_id == anchor_dog_id:
            negative_dog_id = np.random.choice(self.dog_ids)
        negative_path = np.random.choice(self.dog_images[negative_dog_id])
        
        # Load images
        anchor_img = Image.open(anchor_path).convert('RGB')
        positive_img = Image.open(positive_path).convert('RGB')
        negative_img = Image.open(negative_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)
        
        return {
            'anchor': anchor_img,
            'positive': positive_img,
            'negative': negative_img,
            'anchor_id': anchor_dog_id,
            'negative_id': negative_dog_id
        }


def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    use_dual_view: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.
    
    Args:
        data_dir: Root data directory
        batch_size: Batch size
        num_workers: Number of worker processes
        use_dual_view: Use dual-view dataset
        
    Returns:
        (train_loader, val_loader)
    """
    from src.preprocessing import get_train_transforms, get_val_transforms
    
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    
    if use_dual_view:
        train_dataset = DualViewDataset(train_dir, transform=get_train_transforms())
        val_dataset = DualViewDataset(val_dir, transform=get_val_transforms())
    else:
        train_dataset = DogDataset(train_dir, transform=get_train_transforms())
        val_dataset = DogDataset(val_dir, transform=get_val_transforms())
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

