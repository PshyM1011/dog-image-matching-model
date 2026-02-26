# Pre-Augmented Images Guide

## Overview

This guide explains how to use pre-augmented images to speed up training. Instead of applying augmentation on-the-fly during training, you can pre-generate augmented images and save them to disk.

## Benefits

1. **Faster Training**: No augmentation computation during training
2. **Reproducibility**: Same augmented images each time (if you want)
3. **Disk Space Trade-off**: Uses more disk space but saves computation time
4. **Automatic Detection**: Automatically detects if source dataset changed and regenerates if needed

## How It Works

1. **Generate Augmented Images**: Creates multiple augmented versions of each training image
2. **Save to Disk**: Saves augmented images to `data/train_augmented/` (or custom directory)
3. **Use During Training**: Training script loads pre-augmented images instead of applying augmentation on-the-fly
4. **Auto-Regenerate**: Automatically regenerates if source dataset changes

## Usage

> **Note for Windows Users**: In PowerShell, use backticks (`` ` ``) for line continuation instead of backslashes (`\`). Alternatively, you can write commands on a single line.

### Option 1: Automatic (Recommended)

Simply add the `--use_pre_augmented` flag when training:

**Windows PowerShell:**
```powershell
python src/train.py --data_dir data --use_pre_augmented
```

**Linux/Mac (bash):**
```bash
python src/train.py --data_dir data --use_pre_augmented
```

This will:
- Automatically generate augmented images if they don't exist
- Check if source dataset changed and regenerate if needed
- Use pre-augmented images during training

### Option 2: Manual Generation

Generate augmented images manually first:

**Windows PowerShell:**
```powershell
python src/utils/generate_augmented_dataset.py `
    --source_dir data/train `
    --output_dir data/train_augmented `
    --num_augmentations 5
```

**Windows PowerShell (single line):**
```powershell
python src/utils/generate_augmented_dataset.py --source_dir data/train --output_dir data/train_augmented --num_augmentations 5
```

**Linux/Mac (bash):**
```bash
python src/utils/generate_augmented_dataset.py \
    --source_dir data/train \
    --output_dir data/train_augmented \
    --num_augmentations 5
```

Then train with pre-augmented images:

**Windows PowerShell:**
```powershell
python src/train.py --data_dir data --use_pre_augmented
```

**Linux/Mac (bash):**
```bash
python src/train.py --data_dir data --use_pre_augmented
```

### Option 3: Force Regeneration

Force regeneration even if dataset hasn't changed:

**Windows PowerShell:**
```powershell
python src/train.py --data_dir data --use_pre_augmented --force_regenerate_aug
```

**Linux/Mac (bash):**
```bash
python src/train.py --data_dir data --use_pre_augmented --force_regenerate_aug
```

## Command-Line Arguments

### For `train.py`:

- `--use_pre_augmented`: Enable pre-augmented images mode
- `--augmented_dir`: Directory containing pre-augmented images (default: `data/train_augmented`)
- `--num_augmentations`: Number of augmented versions per image (default: 5)
- `--force_regenerate_aug`: Force regeneration even if dataset unchanged

### For `generate_augmented_dataset.py`:

- `--source_dir`: Source directory with original images (default: `data/train`)
- `--output_dir`: Output directory for augmented images (default: `data/train_augmented`)
- `--num_augmentations`: Number of augmented versions per image (default: 5)
- `--force`: Force regeneration even if dataset unchanged

## How Dataset Change Detection Works

The system uses a hash-based approach to detect dataset changes:

1. **Computes Hash**: Creates MD5 hash from:
   - File names
   - File sizes
   - File modification times

2. **Compares Hash**: Compares current hash with saved hash

3. **Regenerates if Changed**: If hash differs, regenerates augmented images

4. **Saves Hash**: Saves hash to `.dataset_hash.json` in output directory

## Directory Structure

After generating augmented images, your directory structure will look like:

```
data/
  train/                    # Original images
    dog1/
      dog1_front_1.jpg
      dog1_side_1.jpg
    ...
  train_augmented/          # Augmented images
    dog1/
      dog1_front_1_aug0.jpg  # Augmented version 0
      dog1_front_1_aug1.jpg  # Augmented version 1
      dog1_front_1_aug2.jpg  # Augmented version 2
      dog1_front_1_aug3.jpg  # Augmented version 3
      dog1_front_1_aug4.jpg  # Augmented version 4
      dog1_side_1_aug0.jpg
      dog1_side_1_aug1.jpg
      ...
    ...
  .dataset_hash.json        # Hash file for change detection
```

## Augmentation Applied

The following augmentations are applied when generating images:

1. **Resize**: 256×256
2. **Random Crop**: 224×224 (random position)
3. **Random Horizontal Flip**: 50% chance
4. **Color Jitter**: Brightness, contrast, saturation, hue variations
5. **Random Rotation**: ±10 degrees

**Note**: Normalization is NOT applied when saving (it's applied during training).

## During Training

When using pre-augmented images:

- **Training**: Randomly selects one of the N augmented versions for each image
- **Transforms**: Only normalization is applied (no augmentation)
- **Validation**: Uses original images (no augmentation, as before)

## Example Workflow

**Windows PowerShell:**

```powershell
# 1. Generate augmented images (5 versions per image)
python src/utils/generate_augmented_dataset.py `
    --source_dir data/train `
    --output_dir data/train_augmented `
    --num_augmentations 5

# Or as a single line:
python src/utils/generate_augmented_dataset.py --source_dir data/train --output_dir data/train_augmented --num_augmentations 5

# 2. Train using pre-augmented images
python src/train.py `
    --data_dir data `
    --batch_size 16 `
    --epochs 50 `
    --use_pre_augmented `
    --augmented_dir data/train_augmented

# Or as a single line:
python src/train.py --data_dir data --batch_size 16 --epochs 50 --use_pre_augmented --augmented_dir data/train_augmented

# 3. If you add new images to data/train/, the system will automatically
#    detect the change and regenerate augmented images on next run
```

**Linux/Mac (bash):**

```bash
# 1. Generate augmented images (5 versions per image)
python src/utils/generate_augmented_dataset.py \
    --source_dir data/train \
    --output_dir data/train_augmented \
    --num_augmentations 5

# 2. Train using pre-augmented images
python src/train.py \
    --data_dir data \
    --batch_size 16 \
    --epochs 50 \
    --use_pre_augmented \
    --augmented_dir data/train_augmented

# 3. If you add new images to data/train/, the system will automatically
#    detect the change and regenerate augmented images on next run
```

## Performance Considerations

### Disk Space

- **Original images**: ~1-2 MB per image
- **Augmented images**: ~1-2 MB per augmented version
- **Total**: If you have 1000 images and generate 5 augmented versions:
  - Original: ~1-2 GB
  - Augmented: ~5-10 GB
  - **Total**: ~6-12 GB

### Training Speed

- **On-the-fly augmentation**: Slower (augmentation computed each epoch)
- **Pre-augmented**: Faster (just load images, no augmentation computation)

### Recommendation

- Use pre-augmented images if:
  - You have sufficient disk space
  - You want faster training
  - You're doing many training runs

- Use on-the-fly augmentation if:
  - Disk space is limited
  - You want different augmentations each epoch (more randomness)

## Troubleshooting

### Issue: "generate_augmented_images not available"

**Solution**: Make sure `src/utils/generate_augmented_dataset.py` exists and is accessible.

### Issue: Augmented images not being used

**Solution**: Check that:
1. `--use_pre_augmented` flag is set
2. Augmented images exist in the specified directory
3. Directory structure matches expected format

### Issue: Want to regenerate but hash matches

**Solution**: Use `--force_regenerate_aug` flag to force regeneration.

## Notes

- Augmented images are only generated for **training** images
- **Validation** and **test** images are never augmented (as before)
- The system automatically detects dataset changes and regenerates if needed
- You can manually delete `data/train_augmented/` to force regeneration


