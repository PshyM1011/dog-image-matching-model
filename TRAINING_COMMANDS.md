# Quick Command Reference

## 🚀 Copy-Paste Commands (In Order)

### Step 1: Navigate to Project Folder
```bash
cd C:\Users\shyni\Research_Project\dog-image-matching-model
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
```

### Step 3: Activate Virtual Environment
```bash
venv\Scripts\activate
```
**Check:** You should see `(venv)` at the start of your line!

### Step 4: Install Packages
```bash
pip install -r requirements.txt
```
**Wait:** This takes 5-15 minutes!

### Step 5: Verify Installation
```bash
python verify_installation.py
```

### Step 6: Verify Dataset
```bash
python src/utils/organize_dataset.py --verify data
```

### Step 7: Train Model (Basic)
```bash
python src/train.py --data_dir data --batch_size 16 --epochs 50
```

### Step 7: Train Model (Recommended - Better Results)
```bash
python src/train.py --data_dir data --batch_size 16 --epochs 50 --lr 0.0001 --embedding_dim 512 --use_combined_loss
```

### Step 8: Evaluate Model
```bash
python src/evaluate.py --checkpoint checkpoints/best.pth --data_dir data --top_k 10
```

---

## 🔧 If You Get Errors

### Out of Memory Error?
```bash
# Use smaller batch size
python src/train.py --data_dir data --batch_size 8 --epochs 50
```

### Module Not Found?
```bash
# Make sure venv is activated (you see (venv))
venv\Scripts\activate
pip install -r requirements.txt
```

### CUDA/GPU Issues?
```bash
# Force CPU mode
python src/train.py --data_dir data --batch_size 16 --epochs 50 --device cpu
```

---

## 📊 Training Options Explained

| Option | What It Does | Recommended Value |
|--------|--------------|-------------------|
| `--data_dir` | Where your dataset is | `data` |
| `--batch_size` | Images processed at once | `16` (reduce if memory error) |
| `--epochs` | How many times to train | `50` (more = better but slower) |
| `--lr` | Learning speed | `0.0001` (default is good) |
| `--embedding_dim` | Feature vector size | `512` (default is good) |
| `--use_combined_loss` | Better loss function | Add this for better results |
| `--device` | CPU or GPU | Auto-detects (use `cpu` if GPU issues) |

---

## ⏱️ Expected Times

- **Package installation**: 5-15 minutes
- **Training (50 epochs)**: 30 minutes - 3 hours (depends on GPU/CPU)
- **Evaluation**: 1-5 minutes

---

## 💡 Pro Tips

1. **Start small**: Test with `--epochs 5` first to make sure everything works
2. **Monitor loss**: Should decrease over time
3. **Save checkpoints**: Best model is saved automatically
4. **Use GPU**: Much faster if you have NVIDIA GPU with CUDA

---

## ✅ Quick Test (5 minutes)

Want to test if everything works? Run a quick training:

```bash
python src/train.py --data_dir data --batch_size 8 --epochs 2
```

This trains for only 2 epochs - just to verify everything is set up correctly!

