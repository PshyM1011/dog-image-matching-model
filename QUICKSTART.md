# Quick Start Guide

This guide will help you get started with the dual-view dog image matching model in **5 simple steps**.

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended, but CPU works too)
- At least 8GB RAM

## Step 1: Setup Environment

### Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt.

### Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: If you have a CUDA GPU and want to use FAISS GPU acceleration:
```bash
pip install faiss-gpu
```

## Step 2: Prepare Your Dataset

### Important: Dataset Organization

**Key Principle**: Same dog IDs across train/val/test, but **different images** in each folder!

```
data/
├── train/
│   ├── dog1/
│   │   ├── dog1_front_1.jpg    ← Training images
│   │   ├── dog1_front_2.jpg
│   │   ├── dog1_side_1.jpg
│   │   └── dog1_side_2.jpg
│   └── ...
├── val/
│   ├── dog1/
│   │   ├── dog1_front_3.jpg    ← DIFFERENT images of same dog1
│   │   └── dog1_side_3.jpg
│   └── ...
└── test/
    ├── dog1/
    │   ├── dog1_front_4.jpg    ← DIFFERENT images of same dog1
    │   └── dog1_side_4.jpg
    └── ...
```

**Rules:**
- ✅ **Same dog IDs** in train/val/test (dog1 in train = same dog1 in val/test)
- ✅ **Different images** in each folder (no duplicates)
- ✅ **Both views** (frontal + lateral) in each folder
- ✅ **Val folder**: Used during training to monitor performance
- ✅ **Test folder**: Used only for final evaluation

**Naming rules:**
- Frontal images: Must contain `front` or `frontal` in filename
- Lateral images: Must contain `side` or `lateral` in filename

### Option A: Manual Organization

Organize your images manually following the structure above.

### Option B: Use Helper Script (Recommended)

If you have all images in one folder per dog, use the helper script:

```bash
python src/utils/organize_dataset.py \
    --source_dir path/to/your/images \
    --output_dir data \
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --test_ratio 0.15
```

This will automatically split your images into train/val/test.

**Verify your dataset:**
```bash
python src/utils/organize_dataset.py --verify data
```

See `DATASET_GUIDE.md` for detailed explanation!

## Step 3: Train the Model

Run the training script:

```bash
python src/train.py --data_dir data --batch_size 16 --epochs 50
```

**Key parameters:**
- `--data_dir`: Root directory containing train/val folders
- `--batch_size`: Batch size (reduce if you run out of memory)
- `--epochs`: Number of training epochs
- `--lr`: Learning rate (default: 0.0001)
- `--embedding_dim`: Embedding dimension (default: 512)
- `--use_combined_loss`: Use combined Triplet + ArcFace loss

**Example with custom settings:**
```bash
python src/train.py \
    --data_dir data \
    --batch_size 32 \
    --epochs 100 \
    --lr 0.001 \
    --embedding_dim 1024 \
    --use_combined_loss \
    --save_dir checkpoints
```

Training will:
- Save checkpoints to `checkpoints/` directory
- Save best model as `checkpoints/best.pth`
- Save training history as `checkpoints/history.json`

## Step 4: Evaluate the Model

After training, evaluate on your test set:

```bash
python src/evaluate.py \
    --checkpoint checkpoints/best.pth \
    --data_dir data \
    --top_k 10 \
    --use_faiss
```

This will output:
- Accuracy@1, Accuracy@5, Accuracy@10
- Save results to `evaluation_results.json`

## Step 5: Use the Model for Matching

### First, compute gallery embeddings:

```python
import torch
from src.model import DualViewFusionModel
from src.utils import DualViewDataset
from src.utils.evaluation import compute_embeddings
from torch.utils.data import DataLoader
from src.preprocessing import get_test_transforms

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DualViewFusionModel(embedding_dim=512).to(device)
checkpoint = torch.load('checkpoints/best.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load gallery
gallery_dataset = DualViewDataset('data/test', transform=get_test_transforms())
gallery_loader = DataLoader(gallery_dataset, batch_size=32)

# Compute embeddings
embeddings, dog_ids = compute_embeddings(model, gallery_loader, device)

# Save
torch.save({'embeddings': embeddings, 'ids': dog_ids}, 'gallery_embeddings.pt')
print('Gallery embeddings saved!')
```

### Then, match a found dog:

```bash
python src/inference.py \
    --checkpoint checkpoints/best.pth \
    --frontal path/to/found_dog_front.jpg \
    --lateral path/to/found_dog_side.jpg \
    --gallery_embeddings gallery_embeddings.pt \
    --top_k 10 \
    --use_detector
```

The `--use_detector` flag enables automatic dog detection and cropping using YOLOv8.

## Troubleshooting

### Out of Memory Errors

- Reduce `--batch_size` (try 8 or 4)
- Use smaller model (reduce `--embedding_dim` to 256)
- Use CPU instead of GPU (slower but uses less memory)

### No Dog Detected

- Check image quality and lighting
- Ensure dog is clearly visible in image
- Try adjusting `--detector_conf_threshold` in detector code

### Poor Matching Results

- Ensure you have enough training data (at least 10-20 images per dog)
- Train for more epochs
- Try different loss functions (use `--use_combined_loss`)
- Check that frontal/lateral images are correctly labeled

## Next Steps

1. **Experiment with hyperparameters**: Try different learning rates, batch sizes, embedding dimensions
2. **Data augmentation**: Modify transforms in `src/preprocessing/transform.py`
3. **Model architecture**: Experiment with different CNN/ViT combinations
4. **Integration**: Connect with nose-print module for Stage 2 verification

## Need Help?

- Check the main [README.md](README.md) for detailed documentation
- Review example notebook: `notebooks/01_example_usage.ipynb`
- Check training logs in `checkpoints/history.json`

Happy training! 🐕

