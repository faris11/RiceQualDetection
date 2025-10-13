import os
import io
import sys
import time
import cv2
import shutil
import pickle
import types
import tempfile
import requests
import numpy as np
from PIL import Image

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

# =========================
# === Streamlit UI setup ==
# =========================
st.set_page_config(
    page_title="Rice Quality Detection",
    page_icon="🍚",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main { padding: 2rem; }
.stApp { background-color: #f5f5f5; }
.upload-box {
  border: 2px dashed #aaa; border-radius: 10px; padding: 2rem;
  text-align: center; margin-bottom: 2rem; background-color: white;
}
.result-box {
  background-color: white; padding: 1.5rem; border-radius: 10px;
  box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-top: 2rem;
}
.header { text-align: center; margin-bottom: 2rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='header'>Rice Quality Detection</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; margin-bottom: 2rem;'>Upload an image of rice to detect its quality.</p>", unsafe_allow_html=True)


# ===================================
# === Model Architecture (VGG+CBAM) ==
# ===================================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
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
        super().__init__()
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
        super().__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)
    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

class VGG16WithCBAM(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None)
        self.features = vgg16.features   # -> [B, 512, 7, 7]
        self.cbam = CBAM(in_planes=512)
        self.avgpool = vgg16.avgpool     # AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.cbam(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# ============================
# === Global Config & Paths ==
# ============================
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(APP_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

# Ambil parameter dari Secrets
MODEL_URL = st.secrets.get("MODEL_URL", "")          # opsi A: URL langsung
GDRIVE_FILE_ID = st.secrets.get("GDRIVE_FILE_ID", "")# opsi B: File ID Google Drive
LOCAL_MODEL_NAME = st.secrets.get("LOCAL_MODEL_NAME", "efficientnet_cbam_6.pth")  # .ts (TorchScript) atau .pth (state_dict)
LOCAL_MODEL_PATH = os.path.join(MODEL_DIR, LOCAL_MODEL_NAME)

NUM_CLASSES = 5
CLASS_NAMES = ["normal", "damage", "chalky", "broken", "discolored"]

# ===========================
# === Downloading helpers ===
# ===========================
def _download_from_url(url: str, dst: str, chunk=8 * 1024 * 1024):
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dst, "wb") as f:
            for blob in r.iter_content(chunk_size=chunk):
                if blob:
                    f.write(blob)

def _download_from_gdrive(file_id: str, dst: str):
    """
    Download file besar dari Google Drive (Anyone-with-link) dengan token konfirmasi.
    """
    session = requests.Session()
    base = "https://docs.google.com/uc?export=download"
    params = {"id": file_id}

    # Step-1: request awal untuk cek token
    with session.get(base, params=params, stream=True, timeout=60) as r1:
        r1.raise_for_status()
        token = None
        for k, v in r1.cookies.items():
            if k.startswith("download_warning"):
                token = v
                break
        if token is None:
            # mungkin file kecil; simpan langsung
            with open(dst, "wb") as f:
                for chunk in r1.iter_content(8 * 1024 * 1024):
                    if chunk:
                        f.write(chunk)
            return

    # Step-2: request ulang dengan token konfirmasi
    params["confirm"] = token
    with session.get(base, params=params, stream=True, timeout=60) as r2:
        r2.raise_for_status()
        with open(dst, "wb") as f:
            for chunk in r2.iter_content(8 * 1024 * 1024):
                if chunk:
                    f.write(chunk)

def _ensure_model_local():
    """
    Pastikan model tersedia lokal & valid (>1KB). Jika belum, unduh dari URL/GDrive.
    """
    if os.path.exists(LOCAL_MODEL_PATH) and os.path.isfile(LOCAL_MODEL_PATH):
        if os.path.getsize(LOCAL_MODEL_PATH) > 1024:
            return

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = tmp.name
    try:
        if MODEL_URL:
            _download_from_url(MODEL_URL, tmp_path)
        elif GDRIVE_FILE_ID:
            _download_from_gdrive(GDRIVE_FILE_ID, tmp_path)
        else:
            raise RuntimeError("MODEL_URL atau GDRIVE_FILE_ID belum di-set di Secrets.")
        if os.path.getsize(tmp_path) <= 1024:
            raise EOFError("Download gagal/terpotong: ukuran <= 1KB")
        shutil.move(tmp_path, LOCAL_MODEL_PATH)
    finally:
        if os.path.exists(tmp_path):
            try: os.remove(tmp_path)
            except Exception: pass

#verifikasi secret
dbg = {
    "has_MODEL_URL": bool(st.secrets.get("MODEL_URL")),
    "has_GDRIVE_FILE_ID": bool(st.secrets.get("GDRIVE_FILE_ID")),
    "LOCAL_MODEL_NAME": st.secrets.get("LOCAL_MODEL_NAME", None),
}
st.caption(f"[secrets-check] {dbg}")


# ==============================
# === Model loading (robust) ===
# ==============================
@st.cache_resource(show_spinner=False)
def load_model():
    """
    Urutan:
    1) Pastikan file ada (download kalau perlu)
    2) Jika ekstensi .ts/.pt -> TorchScript via torch.jit.load
    3) Jika ekstensi .pth -> state_dict: build VGG16WithCBAM lalu load_state_dict
    4) Fallback: pakai VGG16WithCBAM pretrained ImageNet (UI tetap hidup)
    """
    try:
        _ensure_model_local()

        # TorchScript?
        if LOCAL_MODEL_NAME.lower().endswith((".ts", ".pt")):
            try:
                m = torch.jit.load(LOCAL_MODEL_PATH, map_location="cpu")
                m.eval()
                return m
            except Exception as e_ts:
                st.warning(f"TorchScript load gagal, coba state_dict. Detail: {e_ts}")

        # state_dict?
        if LOCAL_MODEL_NAME.lower().endswith(".pth"):
            sd = torch.load(LOCAL_MODEL_PATH, map_location="cpu")
            if not (isinstance(sd, dict) and any(isinstance(v, torch.Tensor) for v in sd.values())):
                raise TypeError("File .pth bukan state_dict yang berisi tensor.")
            # dukung format {'state_dict': ...}
            if "state_dict" in sd and isinstance(sd["state_dict"], dict):
                sd = sd["state_dict"]

            # strip prefix umum
            new_sd = {}
            for k, v in sd.items():
                for pref in ("module.", "model.", "net."):
                    if k.startswith(pref):
                        k = k[len(pref):]
                new_sd[k] = v

            model = VGG16WithCBAM(num_classes=NUM_CLASSES, pretrained=False)
            missing, unexpected = model.load_state_dict(new_sd, strict=False)
            if missing:  print("[load_model] missing:", missing)
            if unexpected: print("[load_model] unexpected:", unexpected)
            model.eval()
            return model

        # format tak dikenali → fallback
        raise RuntimeError("Format model tidak dikenali (gunakan .ts TorchScript atau .pth state_dict).")

    except Exception as e:
        st.warning(f"[Fallback] Memakai VGG16WithCBAM pretrained ImageNet. Detail: {e}")
        m = VGG16WithCBAM(num_classes=NUM_CLASSES, pretrained=True)
        # sesuaikan head ke 5 kelas agar aplikasimu jalan (meski inferensi belum akurat)
        with torch.no_grad():
            m.classifier[-1] = nn.Linear(4096, NUM_CLASSES)
        m.eval()
        return m


# =========================
# === Pre/Post Processing ==
# =========================
_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def is_rice_image(image: Image.Image) -> bool:
    img = np.array(image.convert('RGB'))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    H, W = img.shape[:2]
    max_side = 768
    if max(H, W) > max_side:
        scale = max_side / max(H, W)
        img = cv2.resize(img, (int(W*scale), int(H*scale)), interpolation=cv2.INTER_AREA)
        H, W = img.shape[:2]

    img_area = float(H * W)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    sat_low, sat_high = 0, 70
    val_low, val_high = 140, 255
    white_mask = cv2.inRange(hsv, (0, sat_low, val_low), (179, sat_high, val_high))

    blue_mask = cv2.inRange(hsv, (90, 30, 40), (140, 255, 255))
    mask = cv2.bitwise_and(white_mask, cv2.bitwise_not(blue_mask))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    if cv2.countNonZero(mask) < 0.001 * img_area:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.mean(gray[th == 255]) > np.mean(gray[th == 0]):
            th = cv2.bitwise_not(th)
        mask = th

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return False

    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area <= 0:
        return False

    area_ratio = area / img_area
    if not (0.002 <= area_ratio <= 0.3):
        return False

    rect = cv2.minAreaRect(c)
    (w, h) = rect[1]
    if w == 0 or h == 0:
        return False
    aspect = max(w, h) / min(w, h)
    if not (1.5 <= aspect <= 10.0):
        return False

    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0
    if solidity < 0.85:
        return False

    if len(c) >= 5:
        (xc, yc), (MA, ma), angle = cv2.fitEllipse(c)
        if MA > 0 and ma > 0:
            major, minor = max(MA, ma), min(MA, ma)
            ecc = np.sqrt(max(0.0, 1.0 - (minor/major)**2))
            if ecc < 0.6:
                return False
    return True


def predict_rice_quality(image: Image.Image):
    model = load_model()
    if model is None:
        return None

    pil_img = image.convert("RGB")
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


# ====================
# === Streamlit UI ===
# ====================
st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
upload_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
st.markdown("</div>", unsafe_allow_html=True)

if upload_file is not None:
    image = Image.open(upload_file)
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)

    if not is_rice_image(image):
        with col2:
            st.markdown("<div class='result-box'>", unsafe_allow_html=True)
            st.error("⚠️ This doesn't appear to be a rice image. Please upload an image containing rice grains.")
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        try:
            result = predict_rice_quality(image)
            if result:
                predicted_class, confidence = result
                with col2:
                    st.markdown("<div class='result-box'>", unsafe_allow_html=True)
                    st.success("✅ Rice quality detected!")
                    st.markdown(f"<h3>Result: {predicted_class.upper()}</h3>", unsafe_allow_html=True)
                    st.markdown(f"<p>Confidence: {confidence:.2%}</p>", unsafe_allow_html=True)

                    descriptions = {
                        "normal": "This rice appears to be of normal quality with no significant defects.",
                        "damage": "This rice shows signs of damage, possibly due to improper handling or storage.",
                        "chalky": "This rice has chalky areas, which may affect cooking quality and appearance.",
                        "broken": "This rice contains broken grains, which may affect the cooking texture.",
                        "discolored": "This rice shows discoloration, which may indicate aging or improper storage."
                    }
                    desc = descriptions.get(predicted_class, "No description available.")
                    st.markdown(f"<p><b>Description:</b> {desc}</p>", unsafe_allow_html=True)
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

with st.expander("About Rice Quality Classes"):
    st.markdown("""
    ### Rice Quality Classes
    - **Normal**: Rice grains with standard appearance, size, and color.
    - **Damage**: Rice grains that have been physically damaged during harvesting or processing.
    - **Chalky**: Rice grains with opaque white spots due to incomplete maturation.
    - **Broken**: Rice grains that are not whole (less than 3/4 of a whole kernel).
    - **Discolored**: Rice grains with abnormal color, often yellowish or brownish.
    """)

st.markdown("""
<div style='text-align: center; margin-top: 3rem; color: #888;'>
    <p>© 2023 Rice Quality Detection System</p>
</div>
""", unsafe_allow_html=True)


