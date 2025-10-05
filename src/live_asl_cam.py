# -*- coding: utf-8 -*-
import os, time, collections, json
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

# ========= إعدادات =========
MODEL_PATH   = "model.InceptionV3_stage2.keras"
IMG_SIZE     = (299, 299)
CLASS_NAMES  = [
    "ain","al","aleff","bb","dal","dha","dhad","fa","gaaf","ghain","ha","haa",
    "jeem","kaaf","khaa","la","laam","meem","nun","ra","saad","seen","sheen",
    "ta","taa","thaa","thal","toot","waw","ya","yaa","zay"
]
TRAIN_CLASSES_JSON = "class_names.json"   # اختياري: يثبّت ترتيب الكلاسات من التدريب
ROI_MARGIN   = 0.30                       # جرّبه 0.25–0.40 حسب الكادر
CONF_WARN    = 0.7
EMA_ALPHA    = 0.2
SMOOTH_WIN   = 8
SHOW_MIRROR  = True
USE_TTA_DEF  = False                      # تقدر تبدّله من الكيبورد
CAM_INDEX    = 0

# ========= تحميل الترتيب (اختياري) =========
if os.path.exists(TRAIN_CLASSES_JSON):
    with open(TRAIN_CLASSES_JSON, "r") as f:
        saved_classes = json.load(f)
    if isinstance(saved_classes, list) and len(saved_classes) == len(CLASS_NAMES):
        if saved_classes != CLASS_NAMES:
            print("[WARN] Using class_names.json (training order) instead of hardcoded list.")
        CLASS_NAMES = saved_classes

num_classes = len(CLASS_NAMES)

# ========= تحميل المودل =========
print("[INFO] loading Keras model...")
model = tf.keras.models.load_model(MODEL_PATH)

# ========= MediaPipe =========
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ========= أدوات =========
def compute_hand_roi(frame, hand_landmarks):
    h, w, _ = frame.shape
    xs = [int(lm.x * w) for lm in hand_landmarks.landmark]
    ys = [int(lm.y * h) for lm in hand_landmarks.landmark]
    x1, x2 = max(min(xs), 0), min(max(xs), w - 1)
    y1, y2 = max(min(ys), 0), min(max(ys), h - 1)
    dw, dh = x2 - x1, y2 - y1
    pad_w, pad_h = int(dw * ROI_MARGIN), int(dh * ROI_MARGIN)
    x1, y1 = max(x1 - pad_w, 0), max(y1 - pad_h, 0)
    x2, y2 = min(x2 + pad_w, w - 1), min(y2 + pad_h, h - 1)
    side = max(x2 - x1, y2 - y1, 1)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    x1, y1 = max(0, cx - side // 2), max(0, cy - side // 2)
    x2, y2 = min(w - 1, cx + side // 2), min(h - 1, cy + side // 2)
    return x1, y1, x2, y2

def preprocess_for_model(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_res = cv2.resize(img_rgb, IMG_SIZE, interpolation=cv2.INTER_AREA)
    return img_res.astype(np.float32)

def predict_softmax(img_bgr):
    x = preprocess_for_model(img_bgr)
    x = np.expand_dims(x, 0)
    return model.predict(x, verbose=0)[0]  # (num_classes,)

# ========= تنعيم =========
ema_probs, vote_buffer = None, collections.deque(maxlen=SMOOTH_WIN)
def smooth_and_decode(probs, reset=False):
    global ema_probs, vote_buffer
    if reset or probs is None:
        ema_probs = None
        vote_buffer.clear()
        return None, None, None
    if ema_probs is None:
        ema_probs = probs
    else:
        ema_probs = EMA_ALPHA * probs + (1 - EMA_ALPHA) * ema_probs
    idx = int(np.argmax(ema_probs))
    conf = float(ema_probs[idx])
    vote_buffer.append(idx)
    counts = np.bincount(vote_buffer, minlength=num_classes)
    maj = int(np.argmax(counts))
    return maj, conf, ema_probs

def render_topk(canvas, probs, k=3, x=10, y=20, scale=0.6):
    topk = np.argsort(probs)[-k:][::-1]
    for i, idx in enumerate(topk):
        line = f"Top-{i+1}: {CLASS_NAMES[idx]}  ({probs[idx]*100:.1f}%)"
        color = (0,255,0) if probs[idx] >= CONF_WARN else (0,140,255)
        cv2.putText(canvas, line, (x, y + i*18),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
    if probs[topk[0]] < CONF_WARN:
        cv2.putText(canvas, "LOW CONFIDENCE -> ?????", (x, y + k*18 + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,255), 1, cv2.LINE_AA)

# ========= اللوب الرئيسي =========
def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("❌ ما قدر يفتح الكاميرا.")
        return

    print("Controls: m=mirror | s=TTA | r=reset smoothing | q=quit")
    mirror = SHOW_MIRROR
    use_tta = USE_TTA_DEF
    t0, fps = time.time(), 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        view = cv2.flip(frame, 1) if mirror else frame

        res = hands.process(cv2.cvtColor(view, cv2.COLOR_BGR2RGB))
        if res.multi_hand_landmarks:
            hand = res.multi_hand_landmarks[0]
            x1,y1,x2,y2 = compute_hand_roi(view, hand)
            roi = view[y1:y2, x1:x2]
            if roi.size == 0 or roi.shape[0] < 16 or roi.shape[1] < 16:
                smooth_and_decode(None, reset=True)
                cv2.imshow("ASL Live (CNN)", view)
                if (cv2.waitKey(1) & 0xFF) == ord('q'): break
                continue

            probs = predict_softmax(roi)
            if use_tta:
                roi_flip = cv2.flip(roi, 1)
                probs = 0.5 * (probs + predict_softmax(roi_flip))

            pred_idx, pred_conf, smoothed = smooth_and_decode(probs)
            cv2.rectangle(view, (x1,y1), (x2,y2), (80,220,120), 2)
            mp_draw.draw_landmarks(view, hand, mp_hands.HAND_CONNECTIONS)

            if smoothed is not None:
                render_topk(view, smoothed, k=3, x=10, y=20)
                title = f"{CLASS_NAMES[pred_idx].upper()} ({pred_conf*100:.1f}%)"
            cv2.putText(
                view,
                title,
                (x1, max(40, y1 - 15)),
                cv2.FONT_HERSHEY_SIMPLEX,  # خط واضح وثقيل
                1.8,                        # حجم الخط (كبّره شوي)
                (0, 255, 0) if pred_conf >= CONF_WARN else (0, 0, 255),  # أخضر إذا واثق، أحمر إذا ضعيف
                4,                          # سماكة الخط
                cv2.LINE_AA
            )
        else:
            smooth_and_decode(None, reset=True)

        # FPS
        dt = time.time() - t0
        t0 = time.time()
        fps = 0.9*fps + 0.1*(1.0/max(dt, 1e-6))
        cv2.putText(view, f"FPS: {fps:.1f}", (10, view.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

        cv2.putText(view, f"Mode: CNN-only | TTA:{use_tta}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.imshow("ASL Live (CNN)", view)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('m'): mirror = not mirror
        elif key == ord('s'):
            use_tta = not use_tta
            print(f"[INFO] USE_TTA set to {use_tta}")
        elif key == ord('r'):
            smooth_and_decode(None, reset=True)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()