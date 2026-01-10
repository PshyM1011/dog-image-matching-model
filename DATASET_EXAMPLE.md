# Dataset Organization - Visual Example

## ✅ Your Understanding is 100% CORRECT!

Here's a clear visual example to confirm:

## 📸 Example: Dog1

### Scenario: You have 10 images of Dog1 (5 frontal + 5 lateral)

```
All Dog1 Images:
├── dog1_front_1.jpg
├── dog1_front_2.jpg
├── dog1_front_3.jpg
├── dog1_front_4.jpg
├── dog1_front_5.jpg
├── dog1_side_1.jpg
├── dog1_side_2.jpg
├── dog1_side_3.jpg
├── dog1_side_4.jpg
└── dog1_side_5.jpg
```

### How to Split:

```
data/
├── train/dog1/          ← Put 6-7 images here (for training)
│   ├── dog1_front_1.jpg
│   ├── dog1_front_2.jpg
│   ├── dog1_front_3.jpg
│   ├── dog1_side_1.jpg
│   ├── dog1_side_2.jpg
│   └── dog1_side_3.jpg
│
├── val/dog1/            ← Put 2 DIFFERENT images here (for validation)
│   ├── dog1_front_4.jpg  ← Different from train!
│   └── dog1_side_4.jpg   ← Different from train!
│
└── test/dog1/           ← Put 2 DIFFERENT images here (for testing)
    ├── dog1_front_5.jpg  ← Different from train/val!
    └── dog1_side_5.jpg   ← Different from train/val!
```

## 🎯 Key Points:

1. ✅ **Same dog** (dog1) in all three folders
2. ✅ **Different images** in each folder (no overlap)
3. ✅ **Both views** (frontal + lateral) in each folder
4. ✅ **Val folder** = validation during training
5. ✅ **Test folder** = final evaluation

## 📊 What Goes in Val Folder?

**Val folder contains:**
- **Different images** of the **same dogs** from train
- Used **during training** to:
  - Monitor if model is learning correctly
  - Prevent overfitting
  - Choose the best model checkpoint
  - Stop training early if needed

**Think of it like this:**
- **Train**: Teacher shows student examples → Student learns
- **Val**: Teacher gives quiz during learning → Checks if student understands
- **Test**: Final exam → Tests student's knowledge

## 🔢 Recommended Split (if you have 10 images per dog):

- **Train**: 6-7 images (60-70%) ← Most images here
- **Val**: 2 images (20%) ← Different images
- **Test**: 2 images (20%) ← Different images

## ✅ Quick Checklist:

For each dog (e.g., dog1, dog2, dog3...):

- [ ] At least 2 images in **train** folder (1 frontal + 1 lateral minimum)
- [ ] At least 1 image in **val** folder (different from train)
- [ ] At least 1 image in **test** folder (different from train/val)
- [ ] Same dog ID in all three folders
- [ ] No duplicate images across folders

## 🚀 Use Helper Script:

If you have all images in one folder, use:

```bash
python src/utils/organize_dataset.py \
    --source_dir my_dog_images \
    --output_dir data \
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --test_ratio 0.15
```

This automatically splits your images correctly!

## ❓ Common Questions:

**Q: Can I use the same image in train and val?**
A: ❌ No! Use different images. Same dog, different photos.

**Q: What if I only have 2 images per dog?**
A: Put 1 in train, 1 in val. Skip test (or use same as val for testing).

**Q: Do I need the same number of images in each folder?**
A: ❌ No! Train should have the most (60-70%), val and test can have fewer.

**Q: What if a dog is only in train folder?**
A: ⚠️ Not ideal. Try to have same dogs in all folders for proper evaluation.

---

**You got it right!** 🎉 Your understanding is perfect!

