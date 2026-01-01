# Dual-View Dog Image Matching Model

A deep learning-based dog identification system that matches found dogs with registered lost dogs using **frontal** and **lateral** view images. This model forms the first stage of a hybrid identification system (Dual-view recognition → Nose print verification → Geolocation → Secure communication).

## 🌟 Overview

This system uses a **dual-view approach** combining:
- **Frontal view**: Captures facial features (muzzle, nose, eyes, forehead patterns)
- **Lateral view**: Captures body features (shape, coat color, torso patterns, tail)

The model architecture fuses:
- **EfficientNet-B0 (CNN)**: Extracts local texture features
- **Vision Transformer (ViT-B/16)**: Extracts global structure features
- **Metric Learning**: Uses Triplet Loss and ArcFace for embedding learning

## 📁 Project Structure

```
dog-image-matching-model/
├── data/
│   ├── train/          # Training images organized by dog_id
│   ├── val/            # Validation images
│   └── test/           # Test images
├── src/
│   ├── detector/       # Dog detection (YOLO)
│   ├── preprocessing/  # Image transforms and augmentation
│   ├── model/          # Model architectures and losses
│   ├── utils/          # Dataset loaders and evaluation
│   ├── train.py        # Training script
│   ├── evaluate.py     # Evaluation script
│   └── inference.py    # Inference/matching script
├── notebooks/          # Jupyter notebooks for analysis
├── checkpoints/        # Saved model checkpoints
├── requirements.txt    # Python dependencies
├── config.yaml         # Configuration file
└── README.md
```

## 🚀 Quick Start

### Step 1: Create Virtual Environment

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

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Prepare Your Dataset

Organize your images in the following structure:

```
data/train/
    dog1/
        dog1_front_1.jpg
        dog1_front_2.jpg
        dog1_side_1.jpg
        dog1_side_2.jpg
    dog2/
        dog2_front.jpg
        dog2_side.jpg
    ...
```

**Naming convention:**
- Frontal images: Include `front` or `frontal` in filename
- Lateral images: Include `side` or `lateral` in filename

### Step 4: Train the Model

```bash
python src/train.py \
    --data_dir data \
    --batch_size 16 \
    --epochs 50 \
    --lr 0.0001 \
    --embedding_dim 512 \
    --save_dir checkpoints
```

**With combined loss (Triplet + ArcFace):**
```bash
python src/train.py \
    --data_dir data \
    --batch_size 16 \
    --epochs 50 \
    --use_combined_loss
```

### Step 5: Evaluate the Model

```bash
python src/evaluate.py \
    --checkpoint checkpoints/best.pth \
    --data_dir data \
    --top_k 10 \
    --use_faiss
```

### Step 6: Match a Found Dog

```bash
python src/inference.py \
    --checkpoint checkpoints/best.pth \
    --frontal path/to/frontal_image.jpg \
    --lateral path/to/lateral_image.jpg \
    --gallery_embeddings gallery_embeddings.pt \
    --top_k 10 \
    --use_detector
```

## 🏗️ Architecture

### Dual-View Encoder

The model uses two encoders that process frontal and lateral views separately:

1. **Frontal Encoder** (EfficientNet-B0):
   - Focuses on facial features
   - Output: 256-dim embedding

2. **Lateral Encoder** (ViT-B/16):
   - Focuses on body structure
   - Output: 256-dim embedding

3. **Fusion Layer**:
   - Concatenates both embeddings (512-dim)
   - Projects to final embedding space (512-dim)
   - L2 normalization for cosine similarity

### Loss Functions

- **Triplet Loss**: Ensures same-dog embeddings are close, different dogs are far apart
- **Hard Triplet Loss**: Uses hardest positive/negative pairs in batch
- **ArcFace Loss**: Adds angular margin for better discrimination
- **Combined Loss**: Weighted combination of Triplet + ArcFace

## 📊 Evaluation Metrics

The evaluation pipeline computes:
- **Accuracy@1**: Top-1 retrieval accuracy
- **Accuracy@5**: Top-5 retrieval accuracy
- **Accuracy@10**: Top-10 retrieval accuracy

Uses **FAISS** for fast similarity search on large databases.

## 🔧 Configuration

Edit `config.yaml` to customize:
- Model parameters (embedding dimension, dropout)
- Training hyperparameters (batch size, learning rate, epochs)
- Loss function settings
- Data paths

## 🐕 Dog Detection

The system includes optional YOLOv8-based dog detection for:
- Automatic background removal
- Face/body region cropping
- Better handling of community photos

Enable with `--use_detector` flag in inference.

## 📝 Usage Examples

### Training with Custom Settings

```python
from src.train import main
import sys

sys.argv = [
    'train.py',
    '--data_dir', 'data',
    '--batch_size', '32',
    '--epochs', '100',
    '--lr', '0.001',
    '--embedding_dim', '1024',
    '--use_combined_loss'
]

main()
```

### Computing Gallery Embeddings

```python
import torch
from src.model import DualViewFusionModel
from src.utils import DualViewDataset
from src.utils.evaluation import compute_embeddings
from torch.utils.data import DataLoader

# Load model
model = DualViewFusionModel(embedding_dim=512)
checkpoint = torch.load('checkpoints/best.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load gallery dataset
gallery_dataset = DualViewDataset('data/test', transform=get_test_transforms())
gallery_loader = DataLoader(gallery_dataset, batch_size=32)

# Compute embeddings
embeddings, dog_ids = compute_embeddings(model, gallery_loader, device='cuda')

# Save
torch.save({'embeddings': embeddings, 'ids': dog_ids}, 'gallery_embeddings.pt')
```

## 🔬 Research Background

This implementation is based on:
- **Proposal IT22071248**: Dual-view dog identification system
- **ModelDevPart**: CNN + ViT fusion architecture
- **Imple_model_perplexity**: Augmentation and normalization strategies

Key research findings:
- Face + Body views improve accuracy to **86.5%**
- Adding soft biometrics reaches **~92%**
- Transformers handle messy community images with **>90% accuracy**

## 🎯 Integration with Full System

This dual-view model is **Stage 1** of the complete system:

1. **Stage 1**: Dual-view visual matching (this model)
2. **Stage 2**: Nose print verification (DNNet-style, 98.97% accuracy)
3. **Stage 3**: Adaptive geolocation alerts (~402m radius)
4. **Stage 4**: Secure in-app communication
5. **Stage 5**: Fallback reporting to shelters

## 📦 Dependencies

- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- ultralytics (YOLOv8)
- faiss-cpu (or faiss-gpu for GPU)
- scikit-learn
- opencv-python
- Pillow

See `requirements.txt` for complete list.

## 🤝 Contributing

This is a research project. For questions or issues, please refer to the project documentation.

## 📄 License

[Add your license here]

## 🙏 Acknowledgments

- EfficientNet and Vision Transformer architectures from torchvision
- YOLOv8 from Ultralytics
- FAISS for fast similarity search

---

**Note**: This system is designed for non-invasive dog identification using only photographs, making it accessible and user-friendly compared to microchip-based systems.
