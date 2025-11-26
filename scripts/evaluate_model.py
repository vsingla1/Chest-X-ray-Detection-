import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import tensorflow as tf
import os
import sys

from config import MODEL_CONFIG, DATASET_CONFIG, TRAINING_CONFIG, validate_dataset_paths
from data_processing import XRayDataProcessor

class ModelEvaluator:
    """Evaluate model performance with comprehensive metrics"""
    
    def __init__(self, model):
        self.model = model
    
    def evaluate(self, X_test, y_test):
        """Get predictions and evaluate"""
        y_pred_proba = self.model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        return y_pred, y_pred_proba.flatten()
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Normal', 'Pneumonia'],
                    yticklabels=['Normal', 'Pneumonia'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        
        return fig
    
    def plot_roc_curve(self, y_true, y_pred_proba):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        return fig, roc_auc
    
    def print_classification_report(self, y_true, y_pred):
        """Print detailed classification metrics"""
        report = classification_report(
            y_true, y_pred,
            target_names=['Normal', 'Pneumonia']
        )
        print(report)
        return report
    
    def generate_full_report(self, X_test, y_test):
        """Generate complete evaluation report"""
        y_pred, y_pred_proba = self.evaluate(X_test, y_test)
        
        print("=" * 50)
        print("MODEL EVALUATION REPORT")
        print("=" * 50)
        
        self.print_classification_report(y_test, y_pred)
        
        fig1 = self.plot_confusion_matrix(y_test, y_pred)
        fig2, roc_auc = self.plot_roc_curve(y_test, y_pred_proba)
        
        return {
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'confusion_matrix_fig': fig1,
            'roc_curve_fig': fig2,
            'roc_auc': roc_auc
        }

if __name__ == "__main__":
    print("=" * 60)
    print("X-RAY PNEUMONIA DETECTION - MODEL EVALUATION")
    print("=" * 60)
    
    # Validate paths
    errors = validate_dataset_paths()
    
    if errors:
        print("\n[ERROR] Configuration errors found:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)
    
    model_path = MODEL_CONFIG['model_path']
    
    if not os.path.exists(model_path):
        print(f"\n[ERROR] Model not found at: {model_path}")
        print("Please train the model first by running: python train_model.py")
        sys.exit(1)
    
    print(f"\n[OK] Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    print("Loading test dataset...")
    processor = XRayDataProcessor(img_size=TRAINING_CONFIG['img_size'])
    
    try:
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = processor.prepare_dataset_from_folders(
            train_normal_dir=DATASET_CONFIG['train_normal_dir'],
            train_pneumonia_dir=DATASET_CONFIG['train_pneumonia_dir'],
            test_normal_dir=DATASET_CONFIG['test_normal_dir'],
            test_pneumonia_dir=DATASET_CONFIG['test_pneumonia_dir'],
            val_size=TRAINING_CONFIG['val_size']
        )
        
        print(f"[OK] Test dataset loaded: {len(X_test)} samples")
        
        evaluator = ModelEvaluator(model)
        results = evaluator.generate_full_report(X_test, y_test)
        
        # Save figures
        results_dir = MODEL_CONFIG['results_dir']
        os.makedirs(results_dir, exist_ok=True)
        
        cm_path = os.path.join(results_dir, 'confusion_matrix.png')
        roc_path = os.path.join(results_dir, 'roc_curve.png')
        
        results['confusion_matrix_fig'].savefig(cm_path)
        results['roc_curve_fig'].savefig(roc_path)
        
        print(f"\n[OK] Confusion matrix saved to {cm_path}")
        print(f"[OK] ROC curve saved to {roc_path}")
        print(f"[OK] ROC-AUC Score: {results['roc_auc']:.4f}")
        
        plt.show()
        
    except Exception as e:
        print(f"\n[ERROR] Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
