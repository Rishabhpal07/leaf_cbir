import cv2
import numpy as np
import joblib

# Load model, scaler, and label encoder
model = joblib.load("artifacts/RandomForest_best_model.pkl")
scaler = joblib.load("artifacts/scaler.pkl")
label_encoder = joblib.load("artifacts/label_encoder.pkl")

# ===== Feature extraction functions (same as training) =====

def extract_color_features(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv], [0,1,2], None,
        [8,8,8],
        [0,180,0,256,0,256]
    )
    cv2.normalize(hist, hist)
    return hist.flatten()

def extract_texture_features(gray):
    img = gray.astype("float32") / 255.0
    mean_intensity = float(np.mean(img))
    std_intensity = float(np.std(img))
    hist = cv2.calcHist([gray],[0],None,[16],[0,256])
    cv2.normalize(hist, hist)
    hist_flat = hist.flatten().astype("float32")
    return np.concatenate([[mean_intensity, std_intensity], hist_flat])

def extract_shape_features(gray):
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        return np.zeros(7, dtype=np.float32)
    c = max(cnts, key=cv2.contourArea)
    hu = cv2.HuMoments(cv2.moments(c)).flatten()
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    return hu_log.astype(np.float32)

def extract_features(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (224,224))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    color = extract_color_features(img)
    texture = extract_texture_features(gray)
    shape = extract_shape_features(gray)

    return np.concatenate([color, texture, shape])

# ===== Prediction function =====

def predict(image_path):
    features = extract_features(image_path).reshape(1, -1)
    features_scaled = scaler.transform(features)
    pred = model.predict(features_scaled)[0]
    label = label_encoder.inverse_transform([pred])[0]
    return label

# ===== Test =====
if __name__ == "__main__":
    path = input("Enter image path: ")
    result = predict(path)
    print("\n🌿 Predicted Class:", result)
