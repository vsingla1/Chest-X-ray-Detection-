# X-ray Pneumonia Detection - Setup Guide

## Quick Start

### 1. Install Dependencies
\`\`\`bash
pip install -r requirements.txt
\`\`\`

### 2. Prepare Your Dataset

Create a folder structure like this:
\`\`\`
project_root/
├── data/
│   ├── normal/          # X-ray images of normal lungs
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── pneumonia/       # X-ray images with pneumonia
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
\`\`\`

### 3. Configure Paths

**Option A: Update config.py (Recommended)**
Edit `scripts/config.py` and update these lines:
\`\`\`python
DATASET_CONFIG = {
    "normal_dir": "D:/data/chest_xray/normal",  # Change this path
    "pneumonia_dir": "D:/data/chest_xray/pneumonia",  # Change this path
}
\`\`\`

**Option B: Use Environment Variables**
\`\`\`bash
# Windows
set NORMAL_DIR=D:\data\chest_xray\normal
set PNEUMONIA_DIR=D:\data\chest_xray\pneumonia

# Linux/Mac
export NORMAL_DIR=/home/user/data/normal
export PNEUMONIA_DIR=/home/user/data/pneumonia
\`\`\`

**Option C: Use Command-Line Arguments**
\`\`\`bash
python scripts/train_model.py --normal-dir "D:/data/normal" --pneumonia-dir "D:/data/pneumonia"
\`\`\`

### 4. Validate Configuration
\`\`\`bash
python scripts/config.py
\`\`\`

You should see:
\`\`\`
✓ Found 1000 normal images
✓ Found 1000 pneumonia images
✓ All paths validated successfully!
\`\`\`

### 5. Train the Model
\`\`\`bash
python scripts/train_model.py
\`\`\`

### 6. Evaluate the Model
\`\`\`bash
python scripts/evaluate_model.py
\`\`\`

### 7. Run the Web App
\`\`\`bash
streamlit run scripts/app.py
\`\`\`

Then open your browser to `http://localhost:8501`

## Troubleshooting

### Error: "The system cannot find the path specified"
- Check that your data paths are correct in `config.py`
- Use forward slashes `/` or double backslashes `\\` in Windows paths
- Example: `"D:/data/normal"` or `"D:\\data\\normal"`

### Error: "No images found in directory"
- Verify images are in JPG, JPEG, or PNG format
- Check that the directory path is correct
- Run `python scripts/config.py` to validate paths

### Error: "Model file not found"
- Train the model first: `python scripts/train_model.py`
- Check that the model was saved to the correct location

## Dataset Sources

- **Kaggle**: "Chest X-Ray Images (Pneumonia)" - https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
- **NIH**: Chest X-ray Dataset - https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community
- Your own labeled X-ray images

## Next Steps

1. Collect or download X-ray images
2. Organize them in the `data/normal/` and `data/pneumonia/` folders
3. Update paths in `config.py`
4. Run training and evaluation
5. Use the web app for predictions
