# Summary of Changes for Pre-Augmented Images

## Files Created

1. **`src/utils/generate_augmented_dataset.py`**
   - Script to generate and save augmented images to disk
   - Detects dataset changes using hash-based approach
   - Automatically regenerates if source dataset changed

## Files Modified

1. **`src/train.py`**
   - Added `--use_pre_augmented` flag to enable pre-augmented images mode
   - Added `--augmented_dir` argument to specify augmented images directory
   - Added `--num_augmentations` argument to control number of augmented versions
   - Added `--force_regenerate_aug` flag to force regeneration
   - Automatically generates augmented images if needed when `--use_pre_augmented` is used
   - Uses pre-augmented images during training (only normalization, no augmentation)

2. **`src/utils/dataset.py`**
   - Modified `DualViewDataset` to support pre-augmented images
   - Added `use_augmented` and `augmented_dir` parameters
   - Randomly selects one augmented version per image during training
   - Falls back to original images if augmented versions not found

## How to Use

### Quick Start

```bash
# Train with pre-augmented images (auto-generates if needed)
python src/train.py --data_dir data --use_pre_augmented
```

### Manual Generation

```bash
# Generate augmented images manually
python src/utils/generate_augmented_dataset.py \
    --source_dir data/train \
    --output_dir data/train_augmented \
    --num_augmentations 5

# Then train
python src/train.py --data_dir data --use_pre_augmented
```

## Key Features

1. **Automatic Change Detection**: Detects if source dataset changed and regenerates automatically
2. **Hash-Based Detection**: Uses MD5 hash of file names, sizes, and modification times
3. **Fallback Support**: Falls back to original images if augmented versions not found
4. **Random Selection**: Randomly selects one of N augmented versions during training
5. **Disk Space Efficient**: Only generates augmented versions for training images

## Benefits

- **Faster Training**: No augmentation computation during training
- **Reproducibility**: Can use same augmented images across runs
- **Automatic**: Detects dataset changes and regenerates if needed
- **Flexible**: Can still use on-the-fly augmentation if desired

## Notes

- Augmented images are saved as regular image files (JPEG/PNG)
- Only normalization is applied during training (augmentation already done)
- Validation images are never augmented (as before)
- System automatically handles missing augmented images (falls back to originals)




