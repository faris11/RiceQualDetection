import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import io
import os
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms

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
    
    # (Optional) downscale huge inputs to keep thresholds stable and fast
    H, W = img.shape[:2]
    max_side = 768
    if max(H, W) > max_side:
        scale = max_side / max(H, W)
        img = cv2.resize(img, (int(W*scale), int(H*scale)), interpolation=cv2.INTER_AREA)
        H, W = img.shape[:2]

    img_area = float(H * W)

    # --- 1) Color mask: bright & low-saturation (white-ish) AND not-blue
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # white-ish: low S, high V (tuneable)
    sat_low = 0
    sat_high = 70
    val_low = 140
    val_high = 255
    white_mask = cv2.inRange(hsv, (0, sat_low, val_low), (179, sat_high, val_high))

    # blue range to exclude background (approx 90–140 hue on OpenCV's 0–179 scale)
    blue_mask = cv2.inRange(hsv, (90, 30, 40), (140, 255, 255))
    mask = cv2.bitwise_and(white_mask, cv2.bitwise_not(blue_mask))

    # --- 2) Clean up mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Fallback: if mask is too small, try Otsu on grayscale
    if cv2.countNonZero(mask) < 0.001 * img_area:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # If background is light, invert so rice becomes white
        if np.mean(gray[th == 255]) > np.mean(gray[th == 0]):
            th = cv2.bitwise_not(th)
        mask = th

    # --- 3) Contour analysis
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return False

    # Take the largest white object — should be the grain
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area <= 0:
        return False

    # Normalize by image area (scale-invariant)
    area_ratio = area / img_area  # expected ~0.002–0.2 depending on crop
    if not (0.002 <= area_ratio <= 0.25):
        return False

    # Shape: elongated, compact, convex-ish
    rect = cv2.minAreaRect(c)
    (w, h) = rect[1]
    if w == 0 or h == 0:
        return False
    aspect = max(w, h) / min(w, h)  # rice is elongated
    if not (1.8 <= aspect <= 10.0):
        return False

    # Solidity: area / convex hull area (should be high)
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0
    if solidity < 0.85:
        return False

    # Optional: ellipse fit to check eccentricity (extra safety)
    if len(c) >= 5:
        (xc, yc), (MA, ma), angle = cv2.fitEllipse(c)
        if MA > 0 and ma > 0:
            major, minor = max(MA, ma), min(MA, ma)
            ecc = np.sqrt(max(0.0, 1.0 - (minor/major)**2))  # 0=circle, ~1=line
            if ecc < 0.6:  # should be fairly elongated
                return False
    return True


# Main function for prediction
#def predict_rice_quality(image):
    # Load model
    #model = load_model()
    #if model is None:
    #    return None
    
    # Preprocess image
    #processed_img = preprocess_image(image)
    
    # Make prediction
    #prediction = model.predict(processed_img)
    
    # Get class with highest probability
    #class_idx = np.argmax(prediction[0])
    
    # Map index to class name
    #class_names = ["normal", "damage", "chalky", "broken", "discolored"]
    #predicted_class = class_names[class_idx]
    
    # Get confidence score
    #confidence = float(prediction[0][class_idx])
    
    #return predicted_class, confidence

#Read model from PyTorch
def predict_rice_quality(image):
    # Load PyTorch model
    model = torch.load("model.pth", map_location=torch.device("cpu"))
    model.eval()

    # Preprocess image (contoh, sesuaikan dengan training)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    processed_img = preprocess(image).unsqueeze(0)  # add batch dim

    with torch.no_grad():
        output = model(processed_img)
        prediction = F.softmax(output, dim=1)  # convert to probabilities
        class_idx = torch.argmax(prediction, dim=1).item()
        confidence = prediction[0][class_idx].item()

    class_names = ["normal", "damage", "chalky", "broken", "discolored"]
    predicted_class = class_names[class_idx]

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


