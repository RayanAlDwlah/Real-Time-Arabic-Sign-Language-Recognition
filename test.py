# diag_asl.py
import os, json, numpy as np, cv2, joblib
import tensorflow as tf

P = lambda *a: print("[DIAG]", *a)

# ---- Paths ----
KERAS_PATH = "model.InceptionV3_stage2.keras"
FEAT_DIR   = "features_hybrid"            # لو موجود
XGB_PATH   = "xgb_cnn_plus_landmarks.json"
SCALER_PKL = os.path.join(FEAT_DIR, "scaler_lmk.pkl")
CLASSES_JS = os.path.join(FEAT_DIR, "class_names.json")

# ---- Files exist? ----
P("files:")
P("keras:", os.path.exists(KERAS_PATH), KERAS_PATH)
P("xgb :", os.path.exists(XGB_PATH), XGB_PATH)
P("scaler:", os.path.exists(SCALER_PKL), SCALER_PKL)
P("classes.json:", os.path.exists(CLASSES_JS), CLASSES_JS)

# ---- Class names (fallback to hardcoded) ----
CLASS_NAMES = [
    "ain","al","aleff","bb","dal","dha","dhad","fa","gaaf","ghain","ha","haa",
    "jeem","kaaf","khaa","la","laam","meem","nun","ra","saad","seen","sheen",
    "ta","taa","thaa","thal","toot","waw","ya","yaa","zay"
]
if os.path.exists(CLASSES_JS):
    with open(CLASSES_JS, "r") as f:
        saved = json.load(f)
    if isinstance(saved, list) and len(saved) == len(CLASS_NAMES):
        if saved != CLASS_NAMES:
            P("Using class_names.json (training order).")
        CLASS_NAMES = saved
num_classes = len(CLASS_NAMES)
P("num_classes:", num_classes)

# ---- Load Keras + feature extractor ----
model = tf.keras.models.load_model(KERAS_PATH)
gap_layer = model.get_layer("gap")
feature_extractor = tf.keras.Model(inputs=model.input, outputs=gap_layer.output)
P("keras loaded. feature_extractor output shape:", feature_extractor.output_shape)

# ---- Optional hybrid load ----
xgb_model = None
scaler_lmk = None
if os.path.exists(XGB_PATH) and os.path.exists(SCALER_PKL):
    import xgboost as xgb
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(XGB_PATH)
    scaler_lmk = joblib.load(SCALER_PKL)
    P("xgb & scaler loaded.")
else:
    P("Hybrid part incomplete (missing scaler and/or xgb). CNN-only diagnostics will run.")

# ---- Pick a test image ----
CANDIDATES = []
for root in ["samples", "data_splitted/test"]:
    if os.path.isdir(root):
        for c in os.listdir(root):
            p = os.path.join(root, c)
            if os.path.isdir(p):
                for f in os.listdir(p):
                    if f.lower().endswith((".jpg",".jpeg",".png")):
                        CANDIDATES.append(os.path.join(p, f))
                        break
                if CANDIDATES: break
        if CANDIDATES: break

if not CANDIDATES:
    P("No sample image found. Put one image in samples/ or data_splitted/test/*/")
    raise SystemExit

img_path = CANDIDATES[0]
P("using image:", img_path)
img_bgr = cv2.imread(img_path)
if img_bgr is None:
    P("failed to read image"); raise SystemExit(1)

def preprocess(img_bgr, size=(299,299)):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    res = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
    return res.astype(np.float32)

# ---- CNN prediction ----
x = np.expand_dims(preprocess(img_bgr), 0)
probs_cnn = model.predict(x, verbose=0)[0]
top = int(np.argmax(probs_cnn))
P(f"CNN pred: {CLASS_NAMES[top]} ({probs_cnn[top]*100:.1f}%)")

# ---- Hybrid (only if both files exist) ----
if xgb_model is not None and scaler_lmk is not None:
    gap = feature_extractor.predict(x, verbose=0)[0]  # (2048,)
    # للتشخيص فقط: نحط 89 صفر كـ landmarks عشان نتأكد الدمج 2137 شغال
    lmk89_sc = np.zeros((89,), dtype=np.float32)
    feats = np.concatenate([gap, lmk89_sc], axis=0).reshape(1, -1)
    P("combined feats shape:", feats.shape)
    probs_h = xgb_model.predict_proba(feats)[0]
    top_h = int(np.argmax(probs_h))
    P(f"Hybrid (w/zero landmarks): {CLASS_NAMES[top_h]} ({probs_h[top_h]*100:.1f}%)")
else:
    P("Hybrid test skipped.")