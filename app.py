import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import io
import os

# Set page configuration
st.set_page_config(
    page_title="Rice Quality Detection",
    page_icon="🍚",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
.main {
    padding: 2rem;
}
.stApp {
    background-color: #f5f5f5;
}
.upload-box {
    border: 2px dashed #aaa;
    border-radius: 10px;
    padding: 2rem;
    text-align: center;
    margin-bottom: 2rem;
    background-color: white;
}
.result-box {
    background-color: white;
    padding: 1.5rem;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    margin-top: 2rem;
}
.header {
    text-align: center;
    margin-bottom: 2rem;
}
</style>
""", unsafe_allow_html=True)

# App title and description
st.markdown("<h1 class='header'>Rice Quality Detection</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; margin-bottom: 2rem;'>Upload an image of rice to detect its quality.</p>", unsafe_allow_html=True)

# Function to load the model
@st.cache_resource
def load_model():
    # Replace this with the actual path to your saved model
    model_path = "model/rice_quality_model.h5"
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        return model
    else:
        st.error(f"Model not found at {model_path}. Please ensure the model file exists.")
        return None

# Function to preprocess the image
def preprocess_image(image, target_size=(224, 224)):
    # Convert to RGB if needed
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Resize image
    image = image.resize(target_size)
    
    # Convert to numpy array and normalize
    img_array = np.array(image) / 255.0
    
    # Expand dimensions to match model input shape
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# Function to check if the image contains rice
def is_rice_image(image):
    # Convert PIL image to OpenCV format
    img = np.array(image.convert('RGB'))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply threshold to create binary image
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Count objects that could be rice grains (based on size)
    rice_like_objects = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        # Rice grains typically have an area within a certain range
        # These values may need adjustment based on your specific images
        if 10 < area < 500:  
            rice_like_objects += 1
    
    # If there are enough rice-like objects, consider it a rice image
    return rice_like_objects > 5

# Main function for prediction
def predict_rice_quality(image):
    # Load model
    model = load_model()
    if model is None:
        return None
    
    # Preprocess image
    processed_img = preprocess_image(image)
    
    # Make prediction
    prediction = model.predict(processed_img)
    
    # Get class with highest probability
    class_idx = np.argmax(prediction[0])
    
    # Map index to class name
    class_names = ["normal", "damage", "chalky", "broken", "discolored"]
    predicted_class = class_names[class_idx]
    
    # Get confidence score
    confidence = float(prediction[0][class_idx])
    
    return predicted_class, confidence

# Create a file uploader widget
st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
upload_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
st.markdown("</div>", unsafe_allow_html=True)

# Process the uploaded image
if upload_file is not None:
    # Display the uploaded image
    image = Image.open(upload_file)
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Check if it's a rice image
    if not is_rice_image(image):
        with col2:
            st.markdown("<div class='result-box'>", unsafe_allow_html=True)
            st.error("⚠️ This doesn't appear to be a rice image. Please upload an image containing rice grains.")
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        # Make prediction
        try:
            result = predict_rice_quality(image)
            
            if result:
                predicted_class, confidence = result
                
                with col2:
                    st.markdown("<div class='result-box'>", unsafe_allow_html=True)
                    st.success("✅ Rice quality detected!")
                    st.markdown(f"<h3>Result: {predicted_class.upper()}</h3>", unsafe_allow_html=True)
                    st.markdown(f"<p>Confidence: {confidence:.2%}</p>", unsafe_allow_html=True)
                    
                    # Display description based on rice quality
                    descriptions = {
                        "normal": "This rice appears to be of normal quality with no significant defects.",
                        "damage": "This rice shows signs of damage, possibly due to improper handling or storage.",
                        "chalky": "This rice has chalky areas, which may affect cooking quality and appearance.",
                        "broken": "This rice contains broken grains, which may affect the cooking texture.",
                        "discolored": "This rice shows discoloration, which may indicate aging or improper storage."
                    }
                    
                    st.markdown(f"<p><b>Description:</b> {descriptions[predicted_class]}</p>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
            else:
                with col2:
                    st.markdown("<div class='result-box'>", unsafe_allow_html=True)
                    st.error("❌ Failed to make prediction. Please try again.")
                    st.markdown("</div>", unsafe_allow_html=True)
        except Exception as e:
            with col2:
                st.markdown("<div class='result-box'>", unsafe_allow_html=True)
                st.error(f"❌ Error during prediction: {str(e)}")
                st.markdown("</div>", unsafe_allow_html=True)

# Add information about rice quality classes
with st.expander("About Rice Quality Classes"):
    st.markdown("""
    ### Rice Quality Classes
    
    - **Normal**: Rice grains with standard appearance, size, and color.
    - **Damage**: Rice grains that have been physically damaged during harvesting or processing.
    - **Chalky**: Rice grains with opaque white spots due to incomplete maturation.
    - **Broken**: Rice grains that are not whole (less than 3/4 of a whole kernel).
    - **Discolored**: Rice grains with abnormal color, often yellowish or brownish.
    """)

# Add footer
st.markdown("""
<div style='text-align: center; margin-top: 3rem; color: #888;'>
    <p>© 2023 Rice Quality Detection System</p>
</div>
""", unsafe_allow_html=True)