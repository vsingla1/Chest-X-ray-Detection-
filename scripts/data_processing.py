import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import cv2

class XRayDataProcessor:
    """Handle X-ray image loading, preprocessing, and augmentation"""
    
    def __init__(self, img_size=(224, 224)):
        self.img_size = img_size
    
    def load_images_from_directory(self, directory, label):
        """Load images from directory and assign labels"""
        images = []
        labels = []
        
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        image_files = [f for f in os.listdir(directory) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if len(image_files) == 0:
            raise ValueError(f"No image files found in {directory}")
        
        print(f"Loading {len(image_files)} images from {directory}...")
        
        for i, filename in enumerate(image_files):
            if (i + 1) % 500 == 0:
                print(f"  Loaded {i + 1}/{len(image_files)} images...")
            
            img_path = os.path.join(directory, filename)
            try:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, self.img_size)
                    img = np.stack([img] * 3, axis=-1)
                    images.append(img / 255.0)
                    labels.append(label)
            except Exception as e:
                print(f"Warning: Error loading {img_path}: {e}")
        
        if len(images) == 0:
            raise ValueError(f"Failed to load any images from {directory}")
        
        print(f"  Completed loading {len(images)} images")
        return np.array(images), np.array(labels)
    
    def create_data_generators(self):
        """Create ImageDataGenerator for training and validation"""
        train_gen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        val_gen = ImageDataGenerator()
        
        return train_gen, val_gen
    
    def prepare_dataset_from_folders(self, train_normal_dir, train_pneumonia_dir, 
                                      test_normal_dir, test_pneumonia_dir, val_size=0.15):
        """Prepare train, validation, and test datasets from train/test folder structure"""
        
        # Load training images
        print("\n--- Loading Training Data ---")
        train_normal_imgs, train_normal_labels = self.load_images_from_directory(train_normal_dir, 0)
        train_pneumonia_imgs, train_pneumonia_labels = self.load_images_from_directory(train_pneumonia_dir, 1)
        
        # Combine training data
        X_train_full = np.concatenate([train_normal_imgs, train_pneumonia_imgs])
        y_train_full = np.concatenate([train_normal_labels, train_pneumonia_labels])
        
        # Split training data into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=val_size, random_state=42, stratify=y_train_full
        )
        
        # Load test images
        print("\n--- Loading Test Data ---")
        test_normal_imgs, test_normal_labels = self.load_images_from_directory(test_normal_dir, 0)
        test_pneumonia_imgs, test_pneumonia_labels = self.load_images_from_directory(test_pneumonia_dir, 1)
        
        # Combine test data
        X_test = np.concatenate([test_normal_imgs, test_pneumonia_imgs])
        y_test = np.concatenate([test_normal_labels, test_pneumonia_labels])
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
