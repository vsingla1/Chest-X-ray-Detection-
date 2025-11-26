import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import os
import sys
import argparse
import json

from config import (
    DATASET_CONFIG, MODEL_CONFIG, TRAINING_CONFIG, 
    ensure_directories, validate_dataset_paths
)
from data_processing import XRayDataProcessor

def build_model(input_shape=(224, 224, 3)):
    """Build ResNet50-based model for pneumonia detection"""
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    for layer in base_model.layers:
        layer.trainable = False
    
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    output = Dense(1, activation='sigmoid')(x)
    
    model = Model(base_model.input, output)
    
    model.compile(
        optimizer=Adam(learning_rate=TRAINING_CONFIG['learning_rate']),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(X_train, y_train, X_val, y_val, epochs=None, batch_size=None):
    """Train the pneumonia detection model"""
    if epochs is None:
        epochs = TRAINING_CONFIG['epochs']
    if batch_size is None:
        batch_size = TRAINING_CONFIG['batch_size']
    
    model = build_model()
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7
    )
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    return model, history

def plot_training_history(history):
    """Plot training and validation metrics"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Val Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Model Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train pneumonia detection model')
    parser.add_argument('--epochs', type=int, default=TRAINING_CONFIG['epochs'], help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=TRAINING_CONFIG['batch_size'], help='Batch size')
    
    args = parser.parse_args()
    
    train_normal_dir = DATASET_CONFIG['train_normal_dir']
    train_pneumonia_dir = DATASET_CONFIG['train_pneumonia_dir']
    test_normal_dir = DATASET_CONFIG['test_normal_dir']
    test_pneumonia_dir = DATASET_CONFIG['test_pneumonia_dir']
    model_path = MODEL_CONFIG['model_path']
    
    print("=" * 60)
    print("X-RAY PNEUMONIA DETECTION - MODEL TRAINING")
    print("=" * 60)
    print(f"Train Normal: {train_normal_dir}")
    print(f"Train Pneumonia: {train_pneumonia_dir}")
    print(f"Test Normal: {test_normal_dir}")
    print(f"Test Pneumonia: {test_pneumonia_dir}")
    print(f"Model will be saved to: {model_path}")
    print("=" * 60)
    
    ensure_directories()
    errors = validate_dataset_paths()
    
    if errors:
        print("\n[ERROR] Configuration errors found:")
        for error in errors:
            print(f"  - {error}")
        print("\nPlease check your data folder structure:")
        print("  data/")
        print("    train/")
        print("      NORMAL/")
        print("      PNEUMONIA/")
        print("    test/")
        print("      NORMAL/")
        print("      PNEUMONIA/")
        sys.exit(1)
    
    print("\n[OK] Configuration validated successfully!")
    print("\nLoading dataset...")
    
    processor = XRayDataProcessor(img_size=TRAINING_CONFIG['img_size'])
    
    try:
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = processor.prepare_dataset_from_folders(
            train_normal_dir=train_normal_dir,
            train_pneumonia_dir=train_pneumonia_dir,
            test_normal_dir=test_normal_dir,
            test_pneumonia_dir=test_pneumonia_dir,
            val_size=TRAINING_CONFIG['val_size']
        )
        
        print(f"\n[OK] Dataset loaded successfully!")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Validation samples: {len(X_val)}")
        print(f"  Test samples: {len(X_test)}")
        
        print("\nTraining model...")
        model, history = train_model(
            X_train, y_train, X_val, y_val,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        print(f"\nSaving model to {model_path}...")
        model.save(model_path)
        print("[OK] Model saved successfully!")
        
        # Save training history as JSON
        print("\nSaving training history...")
        results_dir = MODEL_CONFIG['results_dir']
        os.makedirs(results_dir, exist_ok=True)
        
        history_data = {
            'accuracy': [float(x) for x in history.history['accuracy']],
            'val_accuracy': [float(x) for x in history.history['val_accuracy']],
            'loss': [float(x) for x in history.history['loss']],
            'val_loss': [float(x) for x in history.history['val_loss']]
        }
        
        history_json_path = os.path.join(results_dir, 'training_history.json')
        with open(history_json_path, 'w') as f:
            json.dump(history_data, f, indent=2)
        print(f"[OK] Training history saved to {history_json_path}")
        
        # Save training plot
        print("\nGenerating training plots...")
        fig = plot_training_history(history)
        plot_path = os.path.join(results_dir, 'training_history.png')
        fig.savefig(plot_path)
        print(f"[OK] Training plot saved to {plot_path}")
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE!")
        print("=" * 60)
        print(f"Model saved to: {model_path}")
        print(f"History saved to: {history_json_path}")
        print(f"Plot saved to: {plot_path}")
        print("\nYou can now run the Streamlit app:")
        print("  streamlit run scripts/app.py")
        print("=" * 60)
        
        plt.show()
        
    except Exception as e:
        print(f"\n[ERROR] Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
