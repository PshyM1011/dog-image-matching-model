# Dog Image Matching - Usage Guide

## Overview

The `match_dog.py` script allows you to match user-provided dog images against a gallery of known dogs and get the top 5 most similar matches with matching percentages.

## How to Prepare Your Dog Images

### Image Requirements:

1. **You need TWO images of the dog:**
   - **Frontal view**: A front-facing photo of the dog
   - **Lateral view**: A side-facing photo of the dog

2. **Image Format:**
   - Supported formats: JPG, JPEG, PNG
   - Images should be clear and show the dog well
   - Recommended: Good lighting, dog clearly visible

3. **Image Naming:**
   - You can name the images anything you want
   - Example: `my_dog_front.jpg` and `my_dog_side.jpg`
   - Or: `dog1.jpg` and `dog2.jpg`

### Example Image Organization:

```
test_images/
├── my_dog_front.jpg    (frontal view)
└── my_dog_side.jpg     (lateral view)
```

## How to Run the Script

### Basic Usage:

**For PowerShell (Windows):**
```powershell
python src/match_dog.py --checkpoint checkpoints/best.pth --frontal path/to/your/dog_front.jpg --lateral path/to/your/dog_side.jpg --data_dir data
```

**For Bash/Linux/Mac:**
```bash
python src/match_dog.py \
    --checkpoint checkpoints/best.pth \
    --frontal path/to/your/dog_front.jpg \
    --lateral path/to/your/dog_side.jpg \
    --data_dir data
```

### Example with Full Paths:

**For PowerShell (Windows):**
```powershell
python src/match_dog.py --checkpoint checkpoints/best.pth --frontal test_images/my_dog_front.jpg --lateral test_images/my_dog_side.jpg --data_dir data
```

**For Bash/Linux/Mac:**
```bash
python src/match_dog.py \
    --checkpoint checkpoints/best.pth \
    --frontal test_images/my_dog_front.jpg \
    --lateral test_images/my_dog_side.jpg \
    --data_dir data
```

### Using Pre-computed Gallery Embeddings (Faster):

If you've already computed gallery embeddings, you can use them to save time:

**For PowerShell (Windows):**
```powershell
python src/match_dog.py --checkpoint checkpoints/best.pth --frontal test_images/my_dog_front.jpg --lateral test_images/my_dog_side.jpg --data_dir data --gallery_embeddings gallery_embeddings.pt
```

**For Bash/Linux/Mac:**
```bash
python src/match_dog.py \
    --checkpoint checkpoints/best.pth \
    --frontal test_images/my_dog_front.jpg \
    --lateral test_images/my_dog_side.jpg \
    --data_dir data \
    --gallery_embeddings gallery_embeddings.pt
```

### Get More Matches (Top 10):

**For PowerShell (Windows):**
```powershell
python src/match_dog.py --checkpoint checkpoints/best.pth --frontal test_images/my_dog_front.jpg --lateral test_images/my_dog_side.jpg --data_dir data --top_k 10
```

**For Bash/Linux/Mac:**
```bash
python src/match_dog.py \
    --checkpoint checkpoints/best.pth \
    --frontal test_images/my_dog_front.jpg \
    --lateral test_images/my_dog_side.jpg \
    --data_dir data \
    --top_k 10
```

### PowerShell Multi-line (Alternative):

If you prefer multi-line in PowerShell, use backticks (`` ` ``) instead of backslashes:

```powershell
python src/match_dog.py `
    --checkpoint checkpoints/best.pth `
    --frontal test_images/my_dog_front.jpg `
    --lateral test_images/my_dog_side.jpg `
    --data_dir data
```

## Command Line Arguments

| Argument | Required | Description | Default |
|----------|----------|-------------|---------|
| `--checkpoint` | Yes | Path to trained model checkpoint | - |
| `--frontal` | Yes | Path to frontal view image | - |
| `--lateral` | Yes | Path to lateral view image | - |
| `--data_dir` | No | Root data directory | `data` |
| `--gallery_embeddings` | No | Path to pre-computed embeddings | Auto-compute |
| `--top_k` | No | Number of matches to return | `5` |
| `--device` | No | Device: `cpu` or `cuda` | Auto-detect |
| `--batch_size` | No | Batch size for computing embeddings | `32` |

## What the Script Does

1. **Loads the trained model** from the checkpoint
2. **Computes gallery embeddings** from the train set (first time only, then saves for future use)
3. **Preprocesses your input images** (frontal and lateral views)
4. **Computes embedding** for your input dog
5. **Searches the gallery** for similar dogs using cosine similarity
6. **Returns top 5 matches** with matching percentages

## Understanding the Results

The script will display results like this:

```
======================================================================
DOG MATCHING RESULTS
======================================================================

Top 5 Most Similar Dogs Found:

1. Dog ID: dog1
   Similarity Score: 0.9234
   Match Percentage: 96.17%

2. Dog ID: dog5
   Similarity Score: 0.8567
   Match Percentage: 92.84%

3. Dog ID: dog12
   Similarity Score: 0.7891
   Match Percentage: 89.46%

4. Dog ID: dog3
   Similarity Score: 0.7123
   Match Percentage: 85.62%

5. Dog ID: dog8
   Similarity Score: 0.6543
   Match Percentage: 82.72%

======================================================================

Note: Match percentage is calculated from cosine similarity.
Higher percentage indicates higher similarity to the input dog.
```

### Understanding the Metrics:

- **Similarity Score**: Cosine similarity between embeddings (ranges from -1 to 1)
  - Higher values = more similar
  - 1.0 = identical
  - 0.0 = orthogonal (no similarity)
  - -1.0 = opposite

- **Match Percentage**: Normalized similarity score (0-100%)
  - Calculated as: `((similarity + 1) / 2) × 100`
  - Higher percentage = better match
  - 100% = perfect match
  - 50% = neutral similarity

## First Run vs. Subsequent Runs

### First Run:
- Takes longer because it needs to compute gallery embeddings
- Creates `gallery_embeddings.pt` file for future use
- Typical time: 1-5 minutes (depending on gallery size and device)

### Subsequent Runs:
- Much faster if you use `--gallery_embeddings gallery_embeddings.pt`
- Typical time: 10-30 seconds

## Troubleshooting

### Error: "Frontal image not found"
- Check that the path to your frontal image is correct
- Use absolute path if relative path doesn't work
- Example: `C:\Users\YourName\images\dog_front.jpg` (Windows) or `/home/user/images/dog_front.jpg` (Linux/Mac)

### Error: "Lateral image not found"
- Same as above, but for lateral image

### Error: "Train directory not found"
- Make sure your `data` directory has a `train` subdirectory
- Check that `--data_dir` points to the correct location

### Low Match Percentages
- This is normal if the input dog is not in the gallery
- The script finds the most similar dogs, even if they're not exact matches
- Try using better quality images with clear frontal and lateral views

### Slow Performance
- Use `--gallery_embeddings` to avoid recomputing embeddings
- Reduce `--batch_size` if you run out of memory
- Use `--device cpu` if GPU is causing issues

## Tips for Best Results

1. **Use clear, well-lit images** with the dog clearly visible
2. **Ensure proper views**: 
   - Frontal = dog facing camera
   - Lateral = dog from the side
3. **Use images similar to training data** (similar angles, lighting)
4. **Make sure the dog is the main subject** in the image
5. **Use high-resolution images** when possible

## Example Workflow

1. **Prepare images:**
   ```bash
   # Create a folder for test images
   mkdir test_images
   
   # Copy your dog images there
   cp my_dog_front.jpg test_images/
   cp my_dog_side.jpg test_images/
   ```

2. **Run the script:**

   **For PowerShell (Windows):**
   ```powershell
   python src/match_dog.py --checkpoint checkpoints/best.pth --frontal test_images/my_dog_front.jpg --lateral test_images/my_dog_side.jpg --data_dir data
   ```

   **For Bash/Linux/Mac:**
   ```bash
   python src/match_dog.py \
       --checkpoint checkpoints/best.pth \
       --frontal test_images/my_dog_front.jpg \
       --lateral test_images/my_dog_side.jpg \
       --data_dir data
   ```

3. **View results:**
   - The script will print the top 5 matches
   - Review the match percentages
   - The highest percentage is the best match

## Need Help?

If you encounter issues:
1. Check that all file paths are correct
2. Verify that the checkpoint file exists
3. Ensure your data directory structure is correct
4. Check the error messages for specific issues

