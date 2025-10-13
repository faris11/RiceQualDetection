import os
import sys
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

# ===============================
# === Constants for inference ===
# ===============================
NUM_CLASSES = 5
CLASS_NAMES = ["normal", "damage", "chalky", "broken", "discolored"]

# ============================
# === Global Config & Paths ==
# ============================
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(APP_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

def _get_secret_like(keys, default=""):
    """Ambil dari st.secrets root, table [model]/[MODEL], atau ENV."""
    val = None
    for k in keys:
        # root
        try:
            v = st.secrets[k]
            if isinstance(v, str):
                val = v
                break
        except Exception:
            pass
        # table
        try:
            tbl = st.secrets.get("model", {}) or st.secrets.get("MODEL", {})
            if isinstance(tbl, dict) and k in tbl and isinstance(tbl[k], str):
                val = tbl[k]
                break
        except Exception:
            pass
        # env
        v = os.environ.get(k)
        if v is not None:
            val = v
            break
    return (val.strip() if isinstance(val, str) else default)

MODEL_URL        = _get_secret_like(["MODEL_URL"], "")
GDRIVE_FILE_ID   = _get_secret_like(["GDRIVE_FILE_ID", "GOOGLE_DRIVE_FILE_ID"], "")
LOCAL_MODEL_NAME = _get_secret_like(["LOCAL_MODEL_NAME"], "rice_model.ts")  # .ts/.pt (TorchScript) atau .pth (state_dict/FULL)
EFFNET_VER       = _get_secret_like(["EFFNET_VER"], "b0").lower()           # b0/b1/b2/b3

LOCAL_MODEL_PATH = os.path.join(MODEL_DIR, LOCAL_MODEL_NAME)

with st.expander("⚙️ Model config (debug aman)"):
    dbg = {
        "has_MODEL_URL": bool(MODEL_URL),
        "has_GDRIVE_FILE_ID": bool(GDRIVE_FILE_ID),
        "LOCAL_MODEL_NAME": LOCAL_MODEL_NAME,
        "EFFNET_VER": EFFNET_VER,
        "MODEL_DIR_exists": os.path.isdir(MODEL_DIR),
        "LOCAL_MODEL_exists": os.path.isfile(LOCAL_MODEL_PATH),
        "LOCAL_MODEL_size_bytes": os.path.getsize(LOCAL_MODEL_PATH) if os.path.isfile(LOCAL_MODEL_PATH) else 0,
    }
    st.code(dbg, language="python")

# ===================================
# === CBAM building blocks & model ===
# ===================================
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
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
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

def _effnet_feature_dim(ver: str) -> int:
    if ver == "b0": return 1280
    if ver == "b1": return 1280
    if ver == "b2": return 1408
    if ver == "b3": return 1536
    raise ValueError(f"Unsupported EfficientNet version: {ver}")

def _effnet_backbone(ver: str, pretrained: bool = True):
    if ver == "b0":
        return models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
    if ver == "b1":
        return models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.DEFAULT if pretrained else None)
    if ver == "b2":
        return models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT if pretrained else None)
    if ver == "b3":
        return models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT if pretrained else None)
    raise ValueError(f"Unsupported EfficientNet version: {ver}")

class EfficientNetWithCBAM(nn.Module):
    """
    EfficientNet backbone + CBAM pada feature map terakhir + GAP + linear head.
    Disediakan juga alias atribut 'avgpool' -> 'avg_pool' agar state_dict lama yang
    menyimpan kunci 'avgpool' tetap kompatibel.
    """
    def __init__(self, num_classes, efficientnet_version='b0', pretrained=True):
        super(EfficientNetWithCBAM, self).__init__()
        self.efficientnet_version = efficientnet_version
        self.efficientnet = _effnet_backbone(efficientnet_version, pretrained=pretrained)
        feature_dim = _effnet_feature_dim(efficientnet_version)

        # gunakan feature extractor resmi dari torchvision
        self.features = self.efficientnet.features  # -> [B, C, H, W]
        self.cbam = CBAM(feature_dim)

        # pooling & head
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.avgpool = self.avg_pool  # alias untuk kompatibilitas state_dict ('avgpool' expected)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(feature_dim, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.cbam(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

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
    session = requests.Session()
    base = "https://docs.google.com/uc?export=download"
    params = {"id": file_id}

    with session.get(base, params=params, stream=True, timeout=60) as r1:
        r1.raise_for_status()
        token = None
        for k, v in r1.cookies.items():
            if k.startswith("download_warning"):
                token = v
                break
        if token is None:
            with open(dst, "wb") as f:
                for chunk in r1.iter_content(8 * 1024 * 1024):
                    if chunk:
                        f.write(chunk)
            return

    params["confirm"] = token
    with session.get(base, params=params, stream=True, timeout=60) as r2:
        r2.raise_for_status()
        with open(dst, "wb") as f:
            for chunk in r2.iter_content(8 * 1024 * 1024):
                if chunk:
                    f.write(chunk)

def _ensure_model_local():
    """Pastikan bobot ada lokal; bila belum, unduh dari URL/GDrive."""
    if os.path.exists(LOCAL_MODEL_PATH) and os.path.isfile(LOCAL_MODEL_PATH):
        if os.path.getsize(LOCAL_MODEL_PATH) > 1024:
            return

    if not MODEL_URL and not GDRIVE_FILE_ID:
        raise RuntimeError(
            "MODEL_URL atau GDRIVE_FILE_ID tidak ditemukan di Secrets.\n"
            "Isi salah satu dan set juga LOCAL_MODEL_NAME (ekstensi .ts/.pt/.pth)."
        )

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = tmp.name
    try:
        if MODEL_URL:
            _download_from_url(MODEL_URL, tmp_path)
        else:
            _download_from_gdrive(GDRIVE_FILE_ID, tmp_path)

        if os.path.getsize(tmp_path) <= 1024:
            raise EOFError("Download gagal/terpotong (<=1KB). Periksa link/permission.")
        shutil.move(tmp_path, LOCAL_MODEL_PATH)
    finally:
        if os.path.exists(tmp_path):
            try: os.remove(tmp_path)
            except Exception: pass

# ==============================
# === FULL-pickle compatibility =
# ==============================
# Jika .pth adalah FULL model yang menyebut kelas 'EfficientNetWithCBAM',
# pastikan unpickler menemukan kelas tersebut.
def _register_class_aliases(alias_cls, names=("__main__", "main", "app", "models.efficientnet_cbam")):
    for mod_name in names:
        mod = sys.modules.get(mod_name) or types.ModuleType(mod_name)
        sys.modules[mod_name] = mod
        setattr(mod, "EfficientNetWithCBAM", alias_cls)

_register_class_aliases(EfficientNetWithCBAM)

class _RemapUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "EfficientNetWithCBAM":
            return EfficientNetWithCBAM
        return super().find_class(module, name)

class _pickle_mod:
    Unpickler = _RemapUnpickler
    loads = pickle.loads

# ==============================
# === Model loading (robust) ===
# ==============================
def _input_size_for_effnet(ver: str) -> int:
    # ukuran input default torchvision: b0/b1=224, b2=260, b3=300
    if ver == "b0" or ver == "b1": return 224
    if ver == "b2": return 260
    if ver == "b3": return 300
    return 224

@st.cache_resource(show_spinner=False)
def load_model_and_preprocess():
    """
    Urutan:
    1) Pastikan file ada (download kalau perlu)
    2) TorchScript (.ts/.pt) → jit.load
    3) .pth → coba state_dict → coba FULL pickle
    4) Fallback: EfficientNetWithCBAM pretrained
    Return: (model.eval(), preprocess_transforms)
    """
    try:
        _ensure_model_local()

        # TorchScript?
        if LOCAL_MODEL_NAME.lower().endswith((".ts", ".pt")):
            m = torch.jit.load(LOCAL_MODEL_PATH, map_location="cpu")
            m.eval()
            # coba tebak ukuran dari secret
            sz = _input_size_for_effnet(EFFNET_VER)
            preprocess = transforms.Compose([
                transforms.Resize((sz, sz)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
            return m, preprocess

        # .pth → state_dict?
        if LOCAL_MODEL_NAME.lower().endswith(".pth"):
            try:
                sd = torch.load(LOCAL_MODEL_PATH, map_location="cpu")
                if isinstance(sd, dict) and any(isinstance(v, torch.Tensor) for v in sd.values()):
                    # dukung {"state_dict": ...}
                    if "state_dict" in sd and isinstance(sd["state_dict"], dict):
                        sd = sd["state_dict"]
                    # strip prefix umum
                    new_sd = {}
                    for k, v in sd.items():
                        for pref in ("module.", "model.", "net."):
                            if k.startswith(pref):
                                k = k[len(pref):]
                        # kompabilitas kunci 'avgpool' -> 'avg_pool'
                        if k.startswith("avgpool"):
                            k = k.replace("avgpool", "avg_pool", 1)
                        new_sd[k] = v

                    m = EfficientNetWithCBAM(num_classes=NUM_CLASSES, efficientnet_version=EFFNET_VER, pretrained=False)
                    missing, unexpected = m.load_state_dict(new_sd, strict=False)
                    if missing:  print("[load_model] missing:", missing)
                    if unexpected: print("[load_model] unexpected:", unexpected)
                    m.eval()
                    sz = _input_size_for_effnet(EFFNET_VER)
                    preprocess = transforms.Compose([
                        transforms.Resize((sz, sz)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225]),
                    ])
                    return m, preprocess
            except Exception:
                pass

            # FULL pickle?
            try:
                m = torch.load(LOCAL_MODEL_PATH, map_location="cpu")
                if isinstance(m, nn.Module):
                    # beri alias avgpool jika perlu
                    if hasattr(m, "avg_pool") and not hasattr(m, "avgpool"):
                        setattr(m, "avgpool", m.avg_pool)
                    m.eval()
                    sz = _input_size_for_effnet(EFFNET_VER)
                    preprocess = transforms.Compose([
                        transforms.Resize((sz, sz)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225]),
                    ])
                    return m, preprocess
            except Exception:
                pass

            try:
                with open(LOCAL_MODEL_PATH, "rb") as f:
                    m = torch.load(f, map_location="cpu", pickle_module=_pickle_mod)
                if isinstance(m, nn.Module):
                    if hasattr(m, "avg_pool") and not hasattr(m, "avgpool"):
                        setattr(m, "avgpool", m.avg_pool)
                    m.eval()
                    sz = _input_size_for_effnet(EFFNET_VER)
                    preprocess = transforms.Compose([
                        transforms.Resize((sz, sz)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225]),
                    ])
                    return m, preprocess
            except Exception as e_full:
                raise RuntimeError(f"Gagal load .pth sebagai state_dict maupun FULL model. Detail: {e_full}")

        raise RuntimeError(
            f"Format model tidak dikenali dari LOCAL_MODEL_NAME='{LOCAL_MODEL_NAME}'. "
            f"Gunakan .ts/.pt (TorchScript) atau .pth (state_dict/FULL)."
        )

    except Exception as e:
        st.warning(f"[Fallback] Memakai EfficientNetWithCBAM pretrained ({EFFNET_VER}). Detail: {e}")
        m = EfficientNetWithCBAM(num_classes=NUM_CLASSES, efficientnet_version=EFFNET_VER, pretrained=True)
        with torch.no_grad():
            # head sudah 5 kelas, biarkan
            pass
        m.eval()
        sz = _input_size_for_effnet(EFFNET_VER)
        preprocess = transforms.Compose([
            transforms.Resize((sz, sz)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        return m, preprocess

# =========================
# === Pre/Post Processing ==
# =========================
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
    model, preprocess = load_model_and_preprocess()
    if model is None:
        return None

    pil_img = image.convert("RGB")
    x = preprocess(pil_img).unsqueeze(0)  # [1,C,H,W]

    with torch.no_grad():
        logits = model(x)
        if not (isinstance(logits, torch.Tensor) and logits.ndim == 2 and logits.size(1) == NUM_CLASSES):
            raise RuntimeError(
                f"Model output shape {tuple(getattr(logits, 'shape', 'UNKNOWN'))} tidak sesuai NUM_CLASSES={NUM_CLASSES}."
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
