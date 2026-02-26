# Why Image Counts Differ from Sample Counts

## The Problem

You noticed a discrepancy:
- `verify_dataset.py` reports **94 val images** and **95 test images**
- `train.py` reports **52 val samples**
- `evaluate.py` reports **53 query samples**

## The Root Cause

The difference comes from how `DualViewDataset` creates samples versus how `verify_dataset.py` counts images.

### How `verify_dataset.py` Counts Images

`verify_dataset.py` simply counts **ALL image files** it finds:
```python
# It counts every .jpg, .jpeg, .png file
for dog_id in val_dogs:
    val_imgs = list((val_dir / dog_id).glob("*.jpg")) + ...
    val_count += len(val_imgs)  # Counts EVERY image
```

**Result:** 94 images total in val folder

### How `DualViewDataset` Creates Samples

`DualViewDataset` creates samples by **pairing frontal and lateral images**:

1. **For each dog**, it separates images into:
   - `frontal_images`: Images with "front" or "frontal" in filename
   - `lateral_images`: Images with "side" or "lateral" in filename

2. **If a dog has BOTH frontal AND lateral images:**
   - Creates **ALL possible pairs**: `frontal_count × lateral_count`
   - Example: 2 frontal + 3 lateral = **6 samples**

3. **If a dog has ONLY frontal OR ONLY lateral:**
   - With `allow_single_view=True` (default): Creates **1 sample** using the same image for both views
   - With `allow_single_view=False`: Creates **0 samples** (skips the dog)

4. **If images don't match naming pattern:**
   - Images without "front"/"frontal" or "side"/"lateral" in filename are **IGNORED**
   - These images don't contribute to any samples

## Example Scenario

Let's say you have 3 dogs in the val folder:

```
val/
  dog1/
    dog1_front_1.jpg    ← frontal
    dog1_side_1.jpg     ← lateral
  dog2/
    dog2_front_1.jpg    ← frontal
    dog2_front_2.jpg    ← frontal (no lateral!)
  dog3/
    dog3_side_1.jpg     ← lateral
    dog3_side_2.jpg     ← lateral (no frontal!)
```

**`verify_dataset.py` counts:** 6 images total

**`DualViewDataset` creates:**
- dog1: 1 frontal × 1 lateral = **1 sample**
- dog2: 2 frontal × 0 lateral = **1 sample** (uses first frontal for both views)
- dog3: 0 frontal × 2 lateral = **1 sample** (uses first lateral for both views)

**Total samples:** 3 samples (not 6!)

## Why This Happens

The model requires **both frontal and lateral views** to create a dual-view embedding. When:
- Dogs have imbalanced views (e.g., 2 frontal but only 1 lateral)
- Dogs are missing one view entirely
- Images don't follow the naming convention

The sample count will be **less than** the total image count.

## Your Specific Case

Based on the diagnostic output:

### Validation Set (94 images → 52 samples)
- **39 dogs** have both views → creates pairs
- **7 dogs** have only frontal → creates 1 sample each (using same image for both)
- **4 dogs** have only lateral → creates 1 sample each (using same image for both)
- Some dogs have multiple images of one view, creating multiple pairs

### Test Set (95 images → 53 samples)
- Similar pattern: some dogs have both views, some have only one view
- The pairing logic creates fewer samples than total images

## Solutions

### Option 1: Accept the Current Behavior (Recommended)
This is **normal and expected**. The model needs paired views, so:
- Dogs with both views create multiple samples (good for training)
- Dogs with only one view still contribute (1 sample per dog)
- This is the intended behavior for a dual-view model

### Option 2: Ensure All Dogs Have Both Views
If you want to maximize sample count:
1. Check which dogs are missing views
2. Add missing frontal/lateral images for those dogs
3. Re-run the diagnostic to verify

### Option 3: Use Single-View Dataset
If you want to use ALL images (not just pairs), use `DogDataset` instead of `DualViewDataset`:
```python
from src.utils import DogDataset
dataset = DogDataset(val_dir, view_type='both')
```

## How to Diagnose Your Dataset

Run the diagnostic script:
```bash
python diagnose_dataset.py data
```

This will show:
- Which dogs have both views
- Which dogs are missing views
- How many samples each dog contributes
- Why the sample count differs from image count

## Summary

**The discrepancy is NOT a bug** - it's the expected behavior of `DualViewDataset`:
- It pairs frontal and lateral images
- Dogs with imbalanced views create fewer samples
- This is necessary for the dual-view fusion model architecture

The important thing is that your model is training and evaluating correctly, which it is!




