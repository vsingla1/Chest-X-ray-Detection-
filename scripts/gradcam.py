import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

class GradCAM:
    """Generate Grad-CAM heatmaps for model interpretability"""
    
    def __init__(self, model, last_conv_layer_name):
        self.model = model
        self.last_conv_layer_name = last_conv_layer_name
        self.grad_model = tf.keras.models.Model(
            [model.inputs],
            [model.get_layer(last_conv_layer_name).output, model.output]
        )
    
    def generate_heatmap(self, img_array):
        """Generate Grad-CAM heatmap for input image"""
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(img_array)
            class_channel = predictions[:, 0]
        
        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()
    
    def overlay_heatmap(self, img, heatmap, alpha=0.4):
        """Overlay heatmap on original image"""
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Convert grayscale image to BGR for overlay
        if len(img.shape) == 2:
            img_bgr = cv2.cvtColor(np.uint8(img * 255), cv2.COLOR_GRAY2BGR)
        else:
            img_bgr = np.uint8(img * 255)
        
        overlay = cv2.addWeighted(img_bgr, 1 - alpha, heatmap, alpha, 0)
        return overlay
    
    def visualize_prediction(self, img_array, img_original, prediction, class_names=['Normal', 'Pneumonia']):
        """Create visualization with prediction and heatmap"""
        heatmap = self.generate_heatmap(img_array)
        overlay = self.overlay_heatmap(img_original, heatmap)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(img_original, cmap='gray')
        axes[0].set_title('Original X-ray')
        axes[0].axis('off')
        
        # Heatmap
        axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap')
        axes[1].axis('off')
        
        # Overlay
        axes[2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axes[2].set_title(f'Prediction: {class_names[int(prediction > 0.5)]}\nConfidence: {prediction*100:.2f}%')
        axes[2].axis('off')
        
        plt.tight_layout()
        return fig
