# 🚀 START HERE - Complete Training Guide

## Welcome! 👋

This is your **complete step-by-step guide** to train the dual-view dog image matching model.

**Your dataset is already organized and verified! ✅**
- 30 dogs in train/val/test
- 205 training images
- 60 validation images  
- 66 test images
- All properly structured!

---

## 📚 Choose Your Guide

### For Complete Beginners:
👉 **Read:** `STEP_BY_STEP_TRAINING.md`
- Detailed explanations
- What each step does
- Troubleshooting tips

### For Quick Reference:
👉 **Read:** `TRAINING_COMMANDS.md`
- Copy-paste commands
- Quick command reference
- Common fixes

---

## ⚡ Quick Start (5 Steps)

### Step 1: Open Terminal
- Press `Ctrl + ~` in VS Code, OR
- Go to: `Terminal` → `New Terminal`

### Step 2: Navigate to Project
```bash
cd C:\Users\shyni\Research_Project\dog-image-matching-model
```

### Step 3: Create & Activate Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate
```
**You should see `(venv)` at the start!**

### Step 4: Install Packages
```bash
pip install -r requirements.txt
```
**Wait 5-15 minutes for installation**

### Step 5: Start Training!
```bash
python src/train.py --data_dir data --batch_size 16 --epochs 50 --use_combined_loss
```

**That's it!** Training will start. 🎉

---

## 🎯 Recommended Training Command

For best results, use this command:

```bash
python src/train.py --data_dir data --batch_size 16 --epochs 50 --lr 0.0001 --embedding_dim 512 --use_combined_loss
```

**What this does:**
- Uses your dataset in `data/` folder
- Processes 16 images at a time
- Trains for 50 complete passes
- Uses combined loss (better results)
- Creates checkpoints automatically

---

## 📊 What to Expect

### During Installation (Step 4):
- Lots of text scrolling
- Takes 5-15 minutes
- Wait for "Successfully installed..."

### During Training (Step 5):
- Progress bars showing training
- Loss values (should decrease)
- Time remaining
- Takes 30 minutes - 3 hours

### After Training:
- `checkpoints/best.pth` - Your trained model!
- `checkpoints/history.json` - Training history

---

## ✅ Verify Everything First

Before training, verify your setup:

```bash
# 1. Check installation
python verify_installation.py

# 2. Check dataset
python verify_dataset.py data
```

Both should show `[OK]` messages!

---

## 🛠️ If Something Goes Wrong

### "Out of memory" error?
```bash
# Use smaller batch size
python src/train.py --data_dir data --batch_size 8 --epochs 50
```

### "Module not found" error?
```bash
# Make sure venv is activated (you see (venv))
venv\Scripts\activate
pip install -r requirements.txt
```

### Want to test quickly?
```bash
# Train for only 2 epochs (5 minutes)
python src/train.py --data_dir data --batch_size 8 --epochs 2
```

---

## 📖 Full Documentation

- **STEP_BY_STEP_TRAINING.md** - Complete beginner guide
- **TRAINING_COMMANDS.md** - Quick command reference
- **DATASET_GUIDE.md** - Dataset organization explained
- **README.md** - Full system documentation

---

## 🎉 You're Ready!

Your dataset is verified and ready. Just follow the 5 steps above!

**Good luck with training!** 🐕

---

## 💡 Pro Tips

1. **Start with quick test**: Use `--epochs 2` first to verify everything works
2. **Monitor loss**: Should decrease over time (that's good!)
3. **Be patient**: Training takes time, especially first time
4. **Save checkpoints**: Best model is saved automatically in `checkpoints/`

---

## 📞 Need Help?

1. Check `STEP_BY_STEP_TRAINING.md` for detailed explanations
2. Check `TRAINING_COMMANDS.md` for quick fixes
3. Look at error messages - they usually tell you what's wrong

**You've got this!** 💪

