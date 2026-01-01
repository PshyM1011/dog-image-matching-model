# Step-by-Step Training Guide for Beginners

## 🎯 Overview
This guide will walk you through training your dual-view dog image matching model from scratch. Your dataset is already organized correctly! ✅

---

## 📋 STEP 1: Open Terminal/Command Prompt

1. **Open VS Code** (or your code editor)
2. **Open Terminal**:
   - Press `Ctrl + ~` (backtick) in VS Code, OR
   - Go to: `Terminal` → `New Terminal`
3. Make sure you're in the project folder:
   ```bash
   cd C:\Users\shyni\Research_Project\dog-image-matching-model
   ```

---

## 📋 STEP 2: Create Virtual Environment

**What is a virtual environment?**
- It's like a separate box for your project's Python packages
- Prevents conflicts with other projects
- Keeps everything organized

### Commands (Copy and paste one by one):

```bash
python -m venv venv
```

**Wait for it to finish** (takes 10-30 seconds)

**What happened?** A folder called `venv` was created with Python environment.

---

## 📋 STEP 3: Activate Virtual Environment

**Why activate?**
- Tells your computer to use packages from this project only

### Command:

```bash
venv\Scripts\activate
```

**You should see `(venv)` at the start of your command line!**

Example:
```
(venv) C:\Users\shyni\Research_Project\dog-image-matching-model>
```

✅ **If you see `(venv)`, you're good!**

**If it doesn't work**, try:
```bash
.\venv\Scripts\activate
```

---

## 📋 STEP 4: Install Required Packages

**What are packages?**
- Pre-built code libraries (like PyTorch, OpenCV, etc.)
- They make coding easier

### Command:

```bash
pip install -r requirements.txt
```

**This will take 5-15 minutes** depending on your internet speed.

**What's happening?**
- Installing PyTorch (deep learning framework)
- Installing OpenCV (image processing)
- Installing other necessary libraries

**You'll see lots of text scrolling** - that's normal! ✅

**Wait until you see:**
```
Successfully installed ...
```

**If you get errors:**
- Make sure you're in the `(venv)` environment
- Make sure you're in the project folder
- Try: `pip install --upgrade pip` first, then retry

---

## 📋 STEP 5: Verify Installation

**Let's check if everything installed correctly:**

### Command:

```bash
python verify_installation.py
```

**You should see:**
```
✅ PyTorch - OK
✅ torchvision - OK
✅ opencv-python - OK
...
🎉 All checks passed! Your installation is ready.
```

✅ **If all checks pass, you're ready to train!**

---

## 📋 STEP 6: Check Your Dataset

**Your dataset is already organized!** Let's verify it:

### Command:

```bash
python src/utils/organize_dataset.py --verify data
```

**This will check:**
- ✅ Same dog IDs in train/val/test
- ✅ No duplicate images
- ✅ Proper structure

**Expected output:**
```
✅ Same dog IDs in all folders
✅ No duplicate images across folders
```

---

## 📋 STEP 7: Start Training! 🚀

**This is the main step!** Training will take time (30 minutes to several hours).

### Basic Training Command:

```bash
python src/train.py --data_dir data --batch_size 16 --epochs 50
```

**What does this do?**
- `--data_dir data`: Uses your dataset in the `data` folder
- `--batch_size 16`: Processes 16 images at a time (reduce if you get memory errors)
- `--epochs 50`: Trains for 50 complete passes through your data

**What you'll see:**
```
Loading datasets...
Train samples: XXX
Val samples: XXX
Creating model...
Model parameters: XXX,XXX
Starting training...

Epoch 1/50
Train loss: 0.XXXX, Val loss: 0.XXXX
...
```

**Training will show:**
- Progress bars
- Loss values (should decrease over time)
- Time remaining

### Recommended Training Command (Better Settings):

```bash
python src/train.py --data_dir data --batch_size 16 --epochs 50 --lr 0.0001 --embedding_dim 512 --use_combined_loss
```

**What's different?**
- `--use_combined_loss`: Uses both Triplet Loss + ArcFace (better results)
- `--lr 0.0001`: Learning rate (how fast model learns)
- `--embedding_dim 512`: Size of the feature vector

---

## 📋 STEP 8: Monitor Training

**While training, you'll see:**

1. **Epoch progress**: `Epoch 1/50`, `Epoch 2/50`, etc.
2. **Loss values**: Should decrease over time
   - Train loss: How well model fits training data
   - Val loss: How well model generalizes
3. **Progress bars**: Shows how much is done

**Good signs:**
- ✅ Loss decreasing
- ✅ Val loss similar to train loss (not much higher)
- ✅ Progress bars moving

**Warning signs:**
- ⚠️ Loss not decreasing → Model not learning
- ⚠️ Val loss much higher than train → Overfitting
- ⚠️ Out of memory error → Reduce batch_size

---

## 📋 STEP 9: Training Complete!

**When training finishes, you'll see:**
```
Training complete!
Best validation loss: 0.XXXX
```

**What was created?**
- `checkpoints/best.pth` - Best model (use this!)
- `checkpoints/latest.pth` - Latest model
- `checkpoints/history.json` - Training history

---

## 📋 STEP 10: Evaluate Your Model

**Test how well your model works:**

### Command:

```bash
python src/evaluate.py --checkpoint checkpoints/best.pth --data_dir data --top_k 10
```

**This will show:**
- Accuracy@1: Top-1 match accuracy
- Accuracy@5: Top-5 match accuracy
- Accuracy@10: Top-10 match accuracy

**Expected output:**
```
EVALUATION RESULTS
==================================================
Accuracy@1:  0.XXXX
Accuracy@5:  0.XXXX
Accuracy@10: 0.XXXX
==================================================
```

**Good results:**
- Accuracy@1 > 0.70 (70%) = Good
- Accuracy@5 > 0.85 (85%) = Very Good
- Accuracy@10 > 0.90 (90%) = Excellent

---

## 🛠️ Troubleshooting

### Problem: "Out of memory" error

**Solution:**
```bash
# Reduce batch size
python src/train.py --data_dir data --batch_size 8 --epochs 50
```

### Problem: "Module not found" error

**Solution:**
1. Make sure venv is activated: `(venv)` should be visible
2. Reinstall: `pip install -r requirements.txt`

### Problem: Training is too slow

**Solutions:**
- Use GPU if available (automatic if CUDA installed)
- Reduce batch_size
- Reduce epochs for testing

### Problem: Loss not decreasing

**Solutions:**
- Train for more epochs: `--epochs 100`
- Adjust learning rate: `--lr 0.001` (try higher)
- Check if dataset is correct

---

## 📊 Understanding Training Output

### Loss Values:
- **Lower is better**
- Should decrease over time
- Train loss < Val loss is normal (small difference is good)

### Example Good Training:
```
Epoch 1/50:  Train loss: 0.8500, Val loss: 0.9000
Epoch 10/50: Train loss: 0.4500, Val loss: 0.5000
Epoch 20/50: Train loss: 0.3000, Val loss: 0.3500
Epoch 50/50: Train loss: 0.2000, Val loss: 0.2500
```

**See how loss decreases?** ✅ That's good!

---

## 🎯 Quick Reference: All Commands

```bash
# 1. Create venv
python -m venv venv

# 2. Activate venv
venv\Scripts\activate

# 3. Install packages
pip install -r requirements.txt

# 4. Verify installation
python verify_installation.py

# 5. Verify dataset
python src/utils/organize_dataset.py --verify data

# 6. Train model
python src/train.py --data_dir data --batch_size 16 --epochs 50 --use_combined_loss

# 7. Evaluate model
python src/evaluate.py --checkpoint checkpoints/best.pth --data_dir data --top_k 10
```

---

## ✅ Success Checklist

Before training, make sure:
- [ ] Virtual environment created and activated `(venv)`
- [ ] All packages installed (verify_installation.py passed)
- [ ] Dataset verified (no errors)
- [ ] You have enough disk space (checkpoints can be large)

During training:
- [ ] Loss values are decreasing
- [ ] No error messages
- [ ] Progress bars are moving

After training:
- [ ] `checkpoints/best.pth` file exists
- [ ] Evaluation shows reasonable accuracy
- [ ] Ready to use model for matching!

---

## 🎉 You're Ready!

Follow these steps in order, and you'll have a trained model! 

**Remember:**
- Training takes time - be patient!
- Loss should decrease - that's good!
- Save your checkpoints - they're your trained model!

**Good luck!** 🐕

