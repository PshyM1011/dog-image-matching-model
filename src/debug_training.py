"""
Debug script to check training setup and diagnose high loss issues.
"""
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import DualViewDataset
from src.preprocessing import get_train_transforms, get_val_transforms

def check_dataset():
    """Check dataset structure and label consistency."""
    print("=" * 60)
    print("Dataset Diagnostic Check")
    print("=" * 60)
    
    # Check train set
    train_dir = Path('data/train')
    val_dir = Path('data/val')
    
    if not train_dir.exists():
        print(f"ERROR: {train_dir} does not exist!")
        return
    
    if not val_dir.exists():
        print(f"ERROR: {val_dir} does not exist!")
        return
    
    # Get dog IDs
    train_dogs = set([d.name for d in train_dir.iterdir() if d.is_dir()])
    val_dogs = set([d.name for d in val_dir.iterdir() if d.is_dir()])
    
    print(f"\n1. Dataset Structure:")
    print(f"   Train dogs: {len(train_dogs)}")
    print(f"   Val dogs: {len(val_dogs)}")
    
    # Check if all val dogs are in train
    val_not_in_train = val_dogs - train_dogs
    if val_not_in_train:
        print(f"\n   ⚠️  WARNING: {len(val_not_in_train)} validation dogs NOT in training set!")
        print(f"   These dogs: {list(val_not_in_train)[:5]}...")
        print(f"   This will cause label mismatch → HIGH LOSS!")
    else:
        print(f"\n   ✅ All validation dogs are in training set")
    
    # Check dataset samples
    print(f"\n2. Dataset Samples:")
    train_dataset = DualViewDataset(str(train_dir), transform=get_train_transforms())
    val_dataset = DualViewDataset(str(val_dir), transform=get_val_transforms())
    
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Val samples: {len(val_dataset)}")
    
    # Check label distribution
    print(f"\n3. Label Distribution:")
    train_dog_ids = [sample['dog_id'] for sample in train_dataset.samples]
    val_dog_ids = [sample['dog_id'] for sample in val_dataset.samples]
    
    from collections import Counter
    train_counts = Counter(train_dog_ids)
    val_counts = Counter(val_dog_ids)
    
    print(f"   Train: {len(train_counts)} unique dogs")
    print(f"   Val: {len(val_counts)} unique dogs")
    
    # Check if any val dog has very few samples
    print(f"\n4. Sample Counts per Dog:")
    print(f"   Train - Min: {min(train_counts.values())}, Max: {max(train_counts.values())}, Avg: {sum(train_counts.values())/len(train_counts):.1f}")
    print(f"   Val - Min: {min(val_counts.values())}, Max: {max(val_counts.values())}, Avg: {sum(val_counts.values())/len(val_counts):.1f}")
    
    # Check for potential issues
    print(f"\n5. Potential Issues:")
    issues = []
    
    if val_not_in_train:
        issues.append(f"   ❌ {len(val_not_in_train)} validation dogs not in training set")
    
    if len(train_dataset) < 50:
        issues.append(f"   ⚠️  Very few training samples ({len(train_dataset)})")
    
    if len(val_dataset) < 10:
        issues.append(f"   ⚠️  Very few validation samples ({len(val_dataset)})")
    
    if min(train_counts.values()) < 2:
        issues.append(f"   ⚠️  Some dogs have <2 training images")
    
    if not issues:
        print(f"   ✅ No obvious issues found")
    else:
        for issue in issues:
            print(issue)
    
    print("\n" + "=" * 60)
    
    # Create label mapping preview
    print("\n6. Label Mapping Preview:")
    all_train_dog_ids = sorted(list(train_dogs))
    id_to_label = {dog_id: idx for idx, dog_id in enumerate(all_train_dog_ids)}
    
    print(f"   Total classes: {len(id_to_label)}")
    print(f"   First 5 mappings:")
    for i, (dog_id, label) in enumerate(list(id_to_label.items())[:5]):
        print(f"     {dog_id} → {label}")
    
    # Check if val labels will be valid
    print(f"\n7. Validation Label Check:")
    invalid_labels = []
    for dog_id in val_dogs:
        if dog_id not in id_to_label:
            invalid_labels.append(dog_id)
    
    if invalid_labels:
        print(f"   ❌ {len(invalid_labels)} validation dogs will have INVALID labels!")
        print(f"   This will cause VERY HIGH LOSS!")
        print(f"   Invalid dogs: {invalid_labels[:5]}")
    else:
        print(f"   ✅ All validation dogs have valid labels")
    
    print("\n" + "=" * 60)
    print("Diagnostic Complete!")
    print("=" * 60)

if __name__ == '__main__':
    check_dataset()

