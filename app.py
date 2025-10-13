import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import io
import os
import torch
import torch.nn as nn
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

# --- definisi arsitektur kamu ---
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)
    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

class VGG16WithCBAM(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(VGG16WithCBAM, self).__init__()
        vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None)
        self.features = vgg16.features
        self.cbam = CBAM(in_planes=512)
        self.avgpool = vgg16.avgpool
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.cbam(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# --- ALIAS untuk memuaskan pickle ---
class EfficientNetWithCBAM(VGG16WithCBAM):
    """Dummy alias agar FULL model bisa dibuka"""
    def __init__(self, num_classes=5, pretrained=False):
        super().__init__(num_classes=num_classes, pretrained=pretrained)

try:
    from models.efficientnet_cbam import EfficientNetWithCBAM as _EfficientNetWithCBAM
    setattr(sys.modules[__name__], 'EfficientNetWithCBAM', _EfficientNetWithCBAM)
except Exception:
    pass

# Konfigurasi model
MODEL_PATH = "model/efficientnet_cbam_6.pth"   # ganti sesuai lokasi .pth Anda
NUM_CLASSES = 5
CLASS_NAMES = ["normal", "damage", "chalky", "broken", "discolored"]
    
# Function to load the model
@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found at {MODEL_PATH}")
        return None
    try:
        model = torch.load(MODEL_PATH, map_location="cpu")  # unpickle full model
        if not isinstance(model, torch.nn.Module):
            raise TypeError("Loaded object is not an nn.Module. Did you save a state_dict instead?")
        model.eval()
        return model
    except Exception as e:
        st.error(
            "Gagal memuat FULL model (.pth). "
            "Jika error bertuliskan `Can't get attribute 'EfficientNetWithCBAM'`, "
            "pastikan kelas tersebut di-import/terdefinisi di file ini. "
            f"Detail: {e}"
        )
        return None
        
_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

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
    if not (0.002 <= area_ratio <= 0.3):
        return False

    # Shape: elongated, compact, convex-ish
    rect = cv2.minAreaRect(c)
    (w, h) = rect[1]
    if w == 0 or h == 0:
        return False
    aspect = max(w, h) / min(w, h)  # rice is elongated
    if not (1.5 <= aspect <= 10.0):
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
    model = load_model()
    if model is None:
        return None

    pil_img = _to_pil_rgb(image)
    x = _preprocess(pil_img).unsqueeze(0)  # [1,C,H,W]

    with torch.no_grad():
        logits = model(x)
        if logits.ndim != 2 or logits.size(1) != NUM_CLASSES:
            raise RuntimeError(
                f"Model output shape {tuple(logits.shape)} tidak sesuai NUM_CLASSES={NUM_CLASSES}. "
                "Pastikan checkpoint & arsitektur sama."
            )
        probs = F.softmax(logits, dim=1)
        idx = int(torch.argmax(probs, dim=1).item())
        conf = float(probs[0, idx].item())

    label = CLASS_NAMES[idx] if 0 <= idx < len(CLASS_NAMES) else str(idx)
    return label, conf

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









