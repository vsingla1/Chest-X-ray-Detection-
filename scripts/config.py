"""
Configuration file for X-ray Pneumonia Detection Project
Update these paths to match your dataset location
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

DATASET_CONFIG = {
    # Training data paths
    "train_normal_dir": os.getenv("TRAIN_NORMAL_DIR", str(PROJECT_ROOT / "data" / "train" / "NORMAL")),
    "train_pneumonia_dir": os.getenv("TRAIN_PNEUMONIA_DIR", str(PROJECT_ROOT / "data" / "train" / "PNEUMONIA")),
    # Test data paths
    "test_normal_dir": os.getenv("TEST_NORMAL_DIR", str(PROJECT_ROOT / "data" / "test" / "NORMAL")),
    "test_pneumonia_dir": os.getenv("TEST_PNEUMONIA_DIR", str(PROJECT_ROOT / "data" / "test" / "PNEUMONIA")),
}

# Model paths
MODEL_CONFIG = {
    "model_path": os.getenv("MODEL_PATH", str(PROJECT_ROOT / "models" / "pneumonia_detection_model.h5")),
    "results_dir": os.getenv("RESULTS_DIR", str(PROJECT_ROOT / "results")),
}

# Training parameters
TRAINING_CONFIG = {
    "img_size": (224, 224),
    "batch_size": 32,
    "epochs": 35,
    "val_size": 0.15,  # Validation split from training data
    "learning_rate": 1e-4,
}

# Grad-CAM configuration
GRADCAM_CONFIG = {
    "last_conv_layer_name": "conv5_block3_out",
    "alpha": 0.4,
}

def ensure_directories():
    """Create necessary directories if they don't exist"""
    os.makedirs(PROJECT_ROOT / "models", exist_ok=True)
    os.makedirs(PROJECT_ROOT / "results", exist_ok=True)

def validate_dataset_paths():
    """Validate that dataset directories exist and contain images"""
    errors = []
    
    paths_to_check = [
        ("Train Normal", DATASET_CONFIG["train_normal_dir"]),
        ("Train Pneumonia", DATASET_CONFIG["train_pneumonia_dir"]),
        ("Test Normal", DATASET_CONFIG["test_normal_dir"]),
        ("Test Pneumonia", DATASET_CONFIG["test_pneumonia_dir"]),
    ]
    
    for name, path in paths_to_check:
        if not os.path.exists(path):
            errors.append(f"{name} directory not found: {path}")
        else:
            count = len([f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            if count == 0:
                errors.append(f"No images found in {name} directory: {path}")
            else:
                print(f"Found {count} {name} images")
    
    return errors

if __name__ == "__main__":
    print("Configuration Validation")
    print("=" * 50)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Train Normal: {DATASET_CONFIG['train_normal_dir']}")
    print(f"Train Pneumonia: {DATASET_CONFIG['train_pneumonia_dir']}")
    print(f"Test Normal: {DATASET_CONFIG['test_normal_dir']}")
    print(f"Test Pneumonia: {DATASET_CONFIG['test_pneumonia_dir']}")
    print(f"Model Path: {MODEL_CONFIG['model_path']}")
    print("=" * 50)
    
    ensure_directories()
    errors = validate_dataset_paths()
    
    if errors:
        print("\nConfiguration Issues:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("\nAll paths validated successfully!")
