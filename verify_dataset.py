"""
Simple script to verify dataset structure.
"""
import os
from pathlib import Path

def verify_dataset(data_dir='data'):
    """Verify dataset structure."""
    data_path = Path(data_dir)
    train_dir = data_path / "train"
    val_dir = data_path / "val"
    test_dir = data_path / "test"
    
    print("=" * 60)
    print("Dataset Verification")
    print("=" * 60)
    print()
    
    # Check if folders exist
    if not train_dir.exists():
        print("[X] ERROR: train/ folder not found!")
        return False
    if not val_dir.exists():
        print("[X] ERROR: val/ folder not found!")
        return False
    if not test_dir.exists():
        print("[X] ERROR: test/ folder not found!")
        return False
    
    print("[OK] All folders (train/val/test) exist")
    print()
    
    # Get dog IDs from each folder
    train_dogs = set([d.name for d in train_dir.iterdir() if d.is_dir()])
    val_dogs = set([d.name for d in val_dir.iterdir() if d.is_dir()])
    test_dogs = set([d.name for d in test_dir.iterdir() if d.is_dir()])
    
    print(f"Found {len(train_dogs)} dogs in train/")
    print(f"Found {len(val_dogs)} dogs in val/")
    print(f"Found {len(test_dogs)} dogs in test/")
    print()
    
    # Check if same dogs in all folders
    all_dogs = train_dogs | val_dogs | test_dogs
    missing_in_train = all_dogs - train_dogs
    missing_in_val = all_dogs - val_dogs
    missing_in_test = all_dogs - test_dogs
    
    issues = []
    
    if missing_in_train:
        issues.append(f"❌ Dogs in val/test but not in train: {missing_in_train}")
    if missing_in_val:
        issues.append(f"❌ Dogs in train/test but not in val: {missing_in_val}")
    if missing_in_test:
        issues.append(f"❌ Dogs in train/val but not in test: {missing_in_test}")
    
    if not issues:
        print("[OK] Same dog IDs in all folders")
    else:
        for issue in issues:
            print(issue.replace("❌", "[X]"))
    
    # Count images
    print()
    print("Checking images...")
    train_count = 0
    val_count = 0
    test_count = 0
    
    for dog_id in train_dogs:
        train_imgs = list((train_dir / dog_id).glob("*.jpg")) + list((train_dir / dog_id).glob("*.jpeg")) + list((train_dir / dog_id).glob("*.JPEG")) + list((train_dir / dog_id).glob("*.png"))
        train_count += len(train_imgs)
    
    for dog_id in val_dogs:
        val_imgs = list((val_dir / dog_id).glob("*.jpg")) + list((val_dir / dog_id).glob("*.jpeg")) + list((val_dir / dog_id).glob("*.JPEG")) + list((val_dir / dog_id).glob("*.png"))
        val_count += len(val_imgs)
    
    for dog_id in test_dogs:
        test_imgs = list((test_dir / dog_id).glob("*.jpg")) + list((test_dir / dog_id).glob("*.jpeg")) + list((test_dir / dog_id).glob("*.JPEG")) + list((test_dir / dog_id).glob("*.png"))
        test_count += len(test_imgs)
    
    print(f"  Train images: {train_count}")
    print(f"  Val images: {val_count}")
    print(f"  Test images: {test_count}")
    print()
    
    # Check for frontal/lateral
    print("Checking for frontal/lateral views...")
    has_frontal = 0
    has_lateral = 0
    
    for dog_id in list(train_dogs)[:10]:  # Check first 10 dogs
        train_imgs = list((train_dir / dog_id).glob("*"))
        for img in train_imgs:
            name_lower = img.name.lower()
            if 'front' in name_lower or 'frontal' in name_lower:
                has_frontal += 1
            if 'side' in name_lower or 'lateral' in name_lower:
                has_lateral += 1
    
    if has_frontal > 0 and has_lateral > 0:
        print(f"[OK] Found frontal and lateral views (checked {min(10, len(train_dogs))} dogs)")
    else:
        print("[!] Warning: May be missing frontal/lateral view indicators")
    
    print()
    print("=" * 60)
    
    if len(issues) == 0:
        print("[OK] Dataset structure looks good!")
        return True
    else:
        print("[!] Dataset has some issues (see above)")
        return False

if __name__ == '__main__':
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else 'data'
    verify_dataset(data_dir)

