"""
Diagnostic script to understand the discrepancy between image counts and sample counts.
This will show why verify_dataset.py reports more images than train.py/evaluate.py.
"""
import os
from pathlib import Path
from collections import defaultdict

def diagnose_dataset(data_dir='data'):
    """Diagnose why sample counts differ from image counts."""
    data_path = Path(data_dir)
    
    for split in ['val', 'test']:
        split_dir = data_path / split
        
        if not split_dir.exists():
            print(f"[SKIP] {split}/ folder not found")
            continue
        
        print("=" * 70)
        print(f"DIAGNOSIS: {split.upper()} DATASET")
        print("=" * 70)
        
        # Count all images
        total_images = 0
        images_by_dog = defaultdict(list)
        frontal_by_dog = defaultdict(list)
        lateral_by_dog = defaultdict(list)
        unclassified_by_dog = defaultdict(list)
        
        for dog_folder in sorted(split_dir.iterdir()):
            if not dog_folder.is_dir():
                continue
            
            dog_id = dog_folder.name
            # Collect all images, using set to avoid duplicates (Windows glob is case-insensitive)
            images = set()
            images.update(dog_folder.glob('*.jpg'))
            images.update(dog_folder.glob('*.jpeg'))
            images.update(dog_folder.glob('*.JPEG'))
            images.update(dog_folder.glob('*.png'))
            images.update(dog_folder.glob('*.PNG'))
            images = sorted(list(images))  # Convert to sorted list for consistent ordering
            
            for img_path in images:
                total_images += 1
                images_by_dog[dog_id].append(img_path)
                
                filename_lower = img_path.name.lower()
                if 'front' in filename_lower or 'frontal' in filename_lower:
                    frontal_by_dog[dog_id].append(img_path)
                elif 'side' in filename_lower or 'lateral' in filename_lower:
                    lateral_by_dog[dog_id].append(img_path)
                else:
                    unclassified_by_dog[dog_id].append(img_path)
        
        print(f"\n[TOTAL IMAGE COUNT] (what verify_dataset.py reports): {total_images}")
        
        # Calculate how many samples DualViewDataset would create
        total_samples = 0
        dogs_with_both_views = 0
        dogs_with_only_frontal = 0
        dogs_with_only_lateral = 0
        dogs_with_unclassified = 0
        dogs_with_no_valid_images = 0
        
        for dog_id in sorted(images_by_dog.keys()):
            frontal_count = len(frontal_by_dog[dog_id])
            lateral_count = len(lateral_by_dog[dog_id])
            unclassified_count = len(unclassified_by_dog[dog_id])
            
            if frontal_count > 0 and lateral_count > 0:
                # Create ALL possible pairs
                samples_for_dog = frontal_count * lateral_count
                total_samples += samples_for_dog
                dogs_with_both_views += 1
                print(f"  [OK] {dog_id}: {frontal_count} frontal + {lateral_count} lateral = {samples_for_dog} samples")
            elif frontal_count > 0:
                # Only frontal (allow_single_view=True uses first image for both)
                total_samples += 1
                dogs_with_only_frontal += 1
                print(f"  [WARN] {dog_id}: {frontal_count} frontal only = 1 sample (using same image for both views)")
            elif lateral_count > 0:
                # Only lateral (allow_single_view=True uses first image for both)
                total_samples += 1
                dogs_with_only_lateral += 1
                print(f"  [WARN] {dog_id}: {lateral_count} lateral only = 1 sample (using same image for both views)")
            elif unclassified_count > 0:
                # Images that don't match naming pattern
                dogs_with_unclassified += 1
                print(f"  [SKIP] {dog_id}: {unclassified_count} unclassified images (no 'front'/'side' in filename) = 0 samples")
                for img in unclassified_by_dog[dog_id]:
                    print(f"      - {img.name}")
            else:
                dogs_with_no_valid_images += 1
        
        print(f"\n[TOTAL SAMPLE COUNT] (what DualViewDataset creates): {total_samples}")
        print(f"\n[BREAKDOWN]:")
        print(f"  - Dogs with both views: {dogs_with_both_views}")
        print(f"  - Dogs with only frontal: {dogs_with_only_frontal}")
        print(f"  - Dogs with only lateral: {dogs_with_only_lateral}")
        print(f"  - Dogs with unclassified images: {dogs_with_unclassified}")
        print(f"  - Dogs with no valid images: {dogs_with_no_valid_images}")
        
        print(f"\n[KEY INSIGHT]:")
        print(f"  - verify_dataset.py counts: {total_images} images")
        print(f"  - DualViewDataset creates: {total_samples} samples")
        print(f"  - Difference: {total_images - total_samples} images are not used")
        
        if unclassified_by_dog:
            print(f"\n[WARNING]: {sum(len(imgs) for imgs in unclassified_by_dog.values())} images")
            print(f"   don't have 'front'/'frontal' or 'side'/'lateral' in their filename!")
            print(f"   These images are IGNORED by DualViewDataset.")
        
        print("\n" + "=" * 70 + "\n")


if __name__ == '__main__':
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else 'data'
    diagnose_dataset(data_dir)

