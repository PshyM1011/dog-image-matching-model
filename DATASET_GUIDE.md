# Dataset Organization Guide

## ✅ Your Understanding is CORRECT!

Yes, you have the right idea! Here's the complete structure:

## 📁 Dataset Structure

### Key Principle
- **Same dog IDs** across train/val/test folders
- **Different images** in each folder (no overlap)
- Each folder should have **frontal AND lateral** views for each dog

### Example Structure

```
data/
├── train/
│   ├── dog1/
│   │   ├── dog1_front_1.jpg    ← Training images
│   │   ├── dog1_front_2.jpg
│   │   ├── dog1_side_1.jpg
│   │   └── dog1_side_2.jpg
│   ├── dog2/
│   │   ├── dog2_front_1.jpg
│   │   └── dog2_side_1.jpg
│   └── ...
│
├── val/
│   ├── dog1/
│   │   ├── dog1_front_3.jpg    ← DIFFERENT images of same dog1
│   │   └── dog1_side_3.jpg
│   ├── dog2/
│   │   ├── dog2_front_2.jpg    ← DIFFERENT images of same dog2
│   │   └── dog2_side_2.jpg
│   └── ...
│
└── test/
    ├── dog1/
    │   ├── dog1_front_4.jpg    ← DIFFERENT images of same dog1
    │   └── dog1_side_4.jpg
    ├── dog2/
    │   ├── dog2_front_3.jpg    ← DIFFERENT images of same dog2
    │   └── dog2_side_3.jpg
    └── ...
```

## 📋 What Goes in Each Folder

### 1. **Train Folder** ✅
- **Purpose**: Training the model
- **Content**: 
  - At least 2 images per dog (1 frontal + 1 lateral minimum)
  - More images = better training (recommended: 4-10 images per dog)
  - Should have the **most images** (typically 60-70% of all images)

**Example:**
```
train/dog1/
  - dog1_front_1.jpg
  - dog1_front_2.jpg
  - dog1_side_1.jpg
  - dog1_side_2.jpg
```

### 2. **Val Folder** ✅
- **Purpose**: Validation during training (monitoring performance, early stopping)
- **Content**:
  - **Different images** of the **same dogs** from train folder
  - At least 1 frontal + 1 lateral per dog
  - Used to check if model is learning correctly
  - Typically 15-20% of all images

**Example:**
```
val/dog1/
  - dog1_front_3.jpg    ← Different from train images
  - dog1_side_3.jpg     ← Different from train images
```

**Why Val Folder?**
- Prevents overfitting (model learning train images too well)
- Helps choose best model during training
- Monitors if training is going well

### 3. **Test Folder** ✅
- **Purpose**: Final evaluation (unbiased performance measurement)
- **Content**:
  - **Different images** of the **same dogs** from train/val
  - At least 1 frontal + 1 lateral per dog
  - Used ONLY for final evaluation (never during training)
  - Typically 15-20% of all images

**Example:**
```
test/dog1/
  - dog1_front_4.jpg    ← Different from train/val images
  - dog1_side_4.jpg     ← Different from train/val images
```

## 🎯 Important Rules

### ✅ DO:
1. **Same dog IDs** in all three folders (train/val/test)
2. **Different images** in each folder (no duplicates)
3. **Both views** (frontal + lateral) in each folder
4. **More images in train** than val/test

### ❌ DON'T:
1. Put same image in multiple folders
2. Have different dogs in test (should be same dogs as train)
3. Use test images during training
4. Mix up dog IDs (dog1 in train should be same dog1 in val/test)

## 📊 Recommended Split

If you have **10 images per dog** (5 frontal + 5 lateral):

- **Train**: 6-7 images (3-4 frontal + 3-4 lateral) = 60-70%
- **Val**: 2 images (1 frontal + 1 lateral) = 20%
- **Test**: 2 images (1 frontal + 1 lateral) = 20%

## 🔧 How to Organize

### Option 1: Manual Organization
1. Collect all images for each dog
2. Split them into train/val/test (different images)
3. Place in respective folders

### Option 2: Use Helper Script
We've created `src/utils/organize_dataset.py` to help you split your dataset automatically!

## 💡 Why This Structure?

This is called a **"same-identity split"** in metric learning:

- **Train**: Model learns to recognize each dog
- **Val**: Checks if model generalizes to new images of same dogs
- **Test**: Final test on completely unseen images of same dogs

This tests if your model can:
- Recognize the same dog in different photos
- Handle variations in lighting, angle, background
- Generalize beyond training images

## ✅ Summary

| Folder | Same Dogs? | Same Images? | Purpose |
|--------|------------|--------------|---------|
| **Train** | ✅ Yes | ❌ No | Training |
| **Val** | ✅ Yes | ❌ No | Validation during training |
| **Test** | ✅ Yes | ❌ No | Final evaluation |

**Your understanding is 100% correct!** 🎉

