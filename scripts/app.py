import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from gradcam import GradCAM
import os
import json

from config import MODEL_CONFIG, GRADCAM_CONFIG

MODEL_PATH = MODEL_CONFIG['model_path']
HISTORY_PATH = os.path.join(MODEL_CONFIG['results_dir'], "training_history.json")

IMG_SIZE = (224, 224)

st.set_page_config(page_title="Pneumonia Detector", layout="wide")

st.title("Chest X-Ray Pneumonia Detection")
st.markdown("AI-powered pneumonia detection from chest X-ray images")

st.sidebar.header("About")
st.sidebar.info(
    "This application uses a deep learning model to detect pneumonia "
    "from chest X-ray images. Upload an X-ray image to get a prediction."
)

# Load model with caching
@st.cache_resource
def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return load_model(MODEL_PATH)

def preprocess_image(uploaded_file):
    """Preprocess image exactly like training data"""
    # Open image and convert to grayscale (like training)
    img = Image.open(uploaded_file).convert('L')
    img_array = np.array(img)
    
    # Resize to model input size
    img_resized = cv2.resize(img_array, IMG_SIZE)
    
    # Convert grayscale to 3-channel (like training)
    img_rgb = np.stack([img_resized] * 3, axis=-1)
    
    # Normalize to 0-1 (like training)
    img_normalized = img_rgb / 255.0
    
    # Add batch dimension
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    return img_batch, img, img_resized

model = load_trained_model()

if model is None:
    st.error(f"Model not found at {MODEL_PATH}. Please train the model first.")
    st.info("Run: python scripts/train_model.py")
else:
    # Initialize Grad-CAM
    gradcam = GradCAM(model, GRADCAM_CONFIG['last_conv_layer_name'])
    
    tab1, tab2, tab3, tab4 = st.tabs(["Prediction", "Batch Analysis", "Grad-CAM", "Training History"])
    
    with tab1:
        st.header("Single Image Prediction")
        
        uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Uploaded Image")
                img = Image.open(uploaded_file)
                st.image(img, use_container_width=True)
            
            with col2:
                st.subheader("Prediction Result")
                
                uploaded_file.seek(0)  # Reset file pointer
                img_batch, _, _ = preprocess_image(uploaded_file)
                
                # Make prediction
                prediction = model.predict(img_batch, verbose=0)
                confidence = prediction[0][0]
                
                # Training: Normal=0, Pneumonia=1
                # Model output: probability of class 1 (Pneumonia)
                # So: high confidence = Pneumonia, low confidence = Normal
                pneumonia_prob = confidence
                normal_prob = 1 - confidence
                
                # Display results
                if pneumonia_prob > 0.5:
                    st.error(f"**Prediction: PNEUMONIA**")
                    st.metric("Confidence", f"{pneumonia_prob:.2%}")
                else:
                    st.success(f"**Prediction: NORMAL**")
                    st.metric("Confidence", f"{normal_prob:.2%}")
                
                st.info(
                    f"**Probability Breakdown:**\n\n"
                    f"- Pneumonia: {pneumonia_prob:.2%}\n"
                    f"- Normal: {normal_prob:.2%}"
                )
    
    with tab2:
        st.header("Batch Analysis")
        
        st.write("Upload multiple X-ray images for batch analysis")
        
        uploaded_files = st.file_uploader(
            "Upload multiple chest X-ray images",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="batch_uploader"
        )
        
        if uploaded_files:
            results = []
            
            for uploaded_file in uploaded_files:
                img_batch, _, _ = preprocess_image(uploaded_file)
                
                prediction = model.predict(img_batch, verbose=0)
                pneumonia_prob = prediction[0][0]
                
                if pneumonia_prob > 0.5:
                    pred_label = 'Pneumonia'
                    conf = pneumonia_prob
                else:
                    pred_label = 'Normal'
                    conf = 1 - pneumonia_prob
                
                results.append({
                    'Filename': uploaded_file.name,
                    'Prediction': pred_label,
                    'Confidence': f"{conf:.2%}"
                })
            
            st.dataframe(results, use_container_width=True)
            
            # Summary statistics
            pneumonia_count = sum(1 for r in results if r['Prediction'] == 'Pneumonia')
            normal_count = len(results) - pneumonia_count
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Images", len(results))
            col2.metric("Pneumonia Cases", pneumonia_count)
            col3.metric("Normal Cases", normal_count)
    
    with tab3:
        st.header("Grad-CAM Visualization")
        st.write("Upload an X-ray to see which regions the model focuses on")
        
        gradcam_file = st.file_uploader(
            "Upload a chest X-ray image for Grad-CAM",
            type=["jpg", "jpeg", "png"],
            key="gradcam_uploader"
        )
        
        if gradcam_file is not None:
            image_batch, original_img, image_resized = preprocess_image(gradcam_file)
            
            # Make prediction
            pred = model.predict(image_batch, verbose=0)[0][0]
            
            # Generate Grad-CAM
            heatmap = gradcam.generate_heatmap(image_batch)
            overlay = gradcam.overlay_heatmap(image_resized, heatmap, alpha=GRADCAM_CONFIG['alpha'])
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("Original X-ray")
                st.image(original_img, use_container_width=True)
            
            with col2:
                st.subheader("Grad-CAM Heatmap")
                st.image(heatmap, use_container_width=True, clamp=True)
            
            with col3:
                st.subheader("Overlay")
                st.image(
                    cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
                    use_container_width=True
                )
            
            pneumonia_prob = pred
            normal_prob = 1 - pred
            
            if pneumonia_prob > 0.5:
                st.error(f"Prediction: PNEUMONIA ({pneumonia_prob:.2%} confidence)")
            else:
                st.success(f"Prediction: NORMAL ({normal_prob:.2%} confidence)")
            
            st.info(
                "The Grad-CAM heatmap shows which regions of the X-ray the model focused on. "
                "Warmer colors (red/yellow) indicate higher importance for the prediction."
            )
    
    with tab4:
        st.header("Training History")
        
        if os.path.exists(HISTORY_PATH):
            with open(HISTORY_PATH, 'r') as f:
                history = json.load(f)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Accuracy")
                st.line_chart({
                    'Training': history['accuracy'],
                    'Validation': history['val_accuracy']
                })
            
            with col2:
                st.subheader("Loss")
                st.line_chart({
                    'Training': history['loss'],
                    'Validation': history['val_loss']
                })
            
            # Final metrics
            st.subheader("Final Metrics")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Final Train Accuracy", f"{history['accuracy'][-1]:.4f}")
            col2.metric("Final Val Accuracy", f"{history['val_accuracy'][-1]:.4f}")
            col3.metric("Final Train Loss", f"{history['loss'][-1]:.4f}")
            col4.metric("Final Val Loss", f"{history['val_loss'][-1]:.4f}")
        else:
            st.info("Training history not available. Train the model first.")
            st.write(f"Expected path: {HISTORY_PATH}")
