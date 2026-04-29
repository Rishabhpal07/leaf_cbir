# Plant Leaf Classification & Content-Based Image Retrieval (CBIR)

A machine learning project for **classifying plant leaves** into nutrient deficiency categories and retrieving similar images using content-based image retrieval techniques.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Models Used](#models-used)
- [Feature Extraction](#feature-extraction)
- [Results](#results)
- [File Descriptions](#file-descriptions)
- [Requirements](#requirements)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Project Overview

This project classifies plant leaves into 4 different categories based on visual features:
- **FN**: Nitrogen deficiency
- **K**: Potassium deficiency
- **N**: Nitrogen
- **P**: Phosphorus

The system uses **computer vision** and **machine learning** to automatically identify leaf conditions. Additionally, it supports **Content-Based Image Retrieval (CBIR)** to find similar leaves in the dataset.

---

## ✨ Features

✅ **Automated Model Selection** — Trains 4 models and selects the best performer  
✅ **Multiple ML Algorithms** — DecisionTree, RandomForest, SVM, KNN  
✅ **Advanced Feature Extraction** — Color, texture, and shape features  
✅ **Performance Visualization** — Confusion matrix, precision/recall charts, training curves  
✅ **Content-Based Image Retrieval** — Find similar images using cosine similarity  
✅ **Easy Prediction Interface** — Simple command-line prediction on new images  
✅ **Sensor Data Fusion** — Optional integration with environmental sensor data (pH, EC, temperature, humidity)  

---

## 📊 Dataset

The dataset should be organized in the following structure:

```
FNNPK/
├── FN/          (Nitrogen Deficiency images)
├── K/           (Potassium Deficiency images)
├── N/           (Nitrogen images)
└── P/           (Phosphorus images)
```

**Dataset Statistics:**
- FN: 12 images
- K: 71 images
- N: 58 images
- P: 66 images
- **Total: 207 images**

---

## 🚀 Installation

### Prerequisites
- Python 3.11+
- pip or conda

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/plant-leaf-classification.git
cd plant-leaf-classification
```

### Step 2: Create Virtual Environment

```bash
# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install opencv-python scikit-learn pandas joblib tqdm seaborn matplotlib>=3.9.0
```

---

## 💻 Usage

### 1. Train the Model

Train the model on the dataset and automatically select the best performer:

```bash
python model.py
```

**Expected Output:**
```
Extracting from FN: 100%|███████| 12/12
Extracting from K: 100%|███████| 71/71
Extracting from N: 100%|███████| 58/58
Extracting from P: 100%|███████| 66/66
No sensor CSV found → using only image features.

🔹 Training: DecisionTree...
DecisionTree — Accuracy: 0.7619

🔹 Training: RandomForest...
RandomForest — Accuracy: 1.0000

🔹 Training: SVM...
SVM — Accuracy: 0.7857

🔹 Training: KNN...
KNN — Accuracy: 0.7857

Best Model: RandomForest with accuracy 1.0000
📊 Generating visualizations...
✅ Saved best model to: artifacts/RandomForest_best_model.pkl
```

**Generated Artifacts:**
- `artifacts/RandomForest_best_model.pkl` — Trained model
- `artifacts/scaler.pkl` — Feature scaler
- `artifacts/label_encoder.pkl` — Class labels
- `artifacts/confusion_matrix.png` — Performance heatmap
- `artifacts/classification_report_heatmap.png` — Precision/Recall/F1
- `artifacts/performance_barchart.png` — Overall metrics
- `artifacts/accuracy_curve.png` — K-Fold training curves
- `artifacts/loss_curve.png` — K-Fold loss curves

---

### 2. Make Predictions on New Images

Predict the class of a single leaf image:

```bash
python predict.py
```

**Interactive Prompt:**
```
Enter image path: FNNPK/FN/fn1.png

🌿 Predicted Class: FN
```

---

### 3. Find Similar Images (CBIR)

Retrieve the top 3 most similar images for a given query image:

```bash
python cbir.py
```

**Interactive Prompt:**
```
Enter query image path: FNNPK/FN/fn1.png

Top-3 similar images:
FNNPK/FN/fn5.png → class=FN  (sim=0.9823)
FNNPK/FN/fn8.png → class=FN  (sim=0.9734)
FNNPK/FN/fn12.png → class=FN  (sim=0.9612)

Saved result as cbir_result.png
```

The output image `cbir_result.png` shows the query image alongside the top 3 matches.

---

## 📁 Project Structure

```
plant-leaf-classification/
│
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
│
├── model.py                           # Model training & evaluation
├── predict.py                         # Single image prediction
├── cbir.py                           # Content-based image retrieval
│
├── FNNPK/                            # Dataset folder
│   ├── FN/                           # Nitrogen deficiency (12 images)
│   ├── K/                            # Potassium deficiency (71 images)
│   ├── N/                            # Nitrogen (58 images)
│   └── P/                            # Phosphorus (66 images)
│
├── artifacts/                        # Generated models & visualizations
│   ├── RandomForest_best_model.pkl
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   ├── confusion_matrix.png
│   ├── classification_report_heatmap.png
│   ├── performance_barchart.png
│   ├── accuracy_curve.png
│   └── loss_curve.png
│
└── venv/                             # Virtual environment (excluded from git)
```

---

## 🤖 Models Used

The system trains and compares 4 machine learning models:

| Model | Type | Accuracy | Best For |
|-------|------|----------|----------|
| **Decision Tree** | Tree-based | 76.19% | Interpretability |
| **Random Forest** | Ensemble | **100%** | ⭐ **Best** |
| **SVM (RBF)** | Support Vector Machine | 78.57% | Non-linear problems |
| **KNN (k=5)** | Instance-based | 78.57% | Local patterns |

**Automatic Selection:** The model with the highest accuracy is automatically selected and saved.

---

## 🎨 Feature Extraction

Three types of features are extracted from each leaf image:

### 1. **Color Features (512 dimensions)**
- HSV color histogram (8×8×8 bins)
- Captures color variations in leaves

### 2. **Texture Features (18 dimensions)**
- Mean and standard deviation of pixel intensity
- 16-bin grayscale histogram
- Captures surface texture patterns

### 3. **Shape Features (7 dimensions)**
- Hu Moments (scale and rotation invariant)
- Captures leaf contour and shape characteristics

**Total Feature Vector:** ~537 dimensions per image

---

## 📈 Results

### Model Performance

```
Model Comparison:
┌─────────────┬──────────┐
│ Model       │ Accuracy │
├─────────────┼──────────┤
│ RandomForest│  100% ✓  │
│ SVM         │  78.57%  │
│ KNN         │  78.57%  │
│ DecisionTree│  76.19%  │
└─────────────┴──────────┘
```

### Training Results
- **Best Model:** RandomForest
- **Test Accuracy:** 100%
- **Precision (macro):** 1.0000
- **Recall (macro):** 1.0000
- **F1-Score (macro):** 1.0000

### Generated Visualizations

**Confusion Matrix** — Shows prediction accuracy per class
**Classification Report** — Precision, Recall, F1-Score per class
**Performance Bar Chart** — Overall accuracy, precision, recall
**Training Curves** — Accuracy and loss across K-Fold validation

---

## 📄 File Descriptions

### `model.py`
Trains multiple machine learning models on the leaf dataset.

**Key Functions:**
- `extract_color_features()` — Extract HSV color histogram
- `extract_texture_features()` — Extract texture statistics
- `extract_shape_features()` — Extract Hu moments
- `extract_features()` — Combined feature extraction
- `load_dataset()` — Load all images and extract features
- `fuse_sensor()` — Optionally fuse sensor data
- `train_models()` — Train and compare 4 models
- `main()` — Orchestrate training and visualization

**Usage:** `python model.py`

---

### `predict.py`
Predicts the class of a single leaf image using the trained model.

**Key Functions:**
- `extract_color_features()` — Extract HSV color histogram
- `extract_texture_features()` — Extract texture statistics
- `extract_shape_features()` — Extract Hu moments
- `extract_features()` — Combined feature extraction
- `predict()` — Make prediction on a single image

**Usage:** `python predict.py`

---

### `cbir.py`
Finds the top 3 most similar images for a given query image.

**Key Functions:**
- `extract_dataset_features()` — Extract features from all dataset images
- `extract_query_features()` — Extract features from query image
- `retrieve_top3()` — Find and visualize top 3 similar images

**Algorithm:** Cosine similarity between feature vectors

**Usage:** `python cbir.py`

---

## 📦 Requirements

```
opencv-python>=4.12.0.88
scikit-learn>=1.7.2
pandas>=2.3.3
numpy>=2.3.5
joblib>=1.5.2
tqdm>=4.67.1
seaborn>=0.14.0
matplotlib>=3.9.0
```

---

## 🔧 Quick Start

```bash
# 1. Clone repository
git clone https://github.com/yourusername/plant-leaf-classification.git
cd plant-leaf-classification

# 2. Setup environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train model
python model.py

# 5. Test prediction
python predict.py
# Enter: FNNPK/FN/fn1.png

# 6. Test CBIR
python cbir.py
# Enter: FNNPK/FN/fn1.png

# 7. View results
open artifacts/confusion_matrix.png
```

---

## 🎓 How It Works

1. **Feature Extraction** → Extract color, texture, and shape features from leaf images
2. **Data Preprocessing** → Normalize features using StandardScaler
3. **Model Training** → Train 4 ML models on 80% of data
4. **Model Selection** → Automatically select model with highest accuracy
5. **Validation** → Evaluate on 20% test set using K-Fold cross-validation
6. **Visualization** → Generate performance charts and confusion matrix
7. **Prediction** → Classify new leaf images using trained model

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Commit (`git commit -m 'Add improvement'`)
5. Push to branch (`git push origin feature/improvement`)
6. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Your Name**  
GitHub: [@yourusername](https://github.com/yourusername)

---

## 🙋 Support

If you encounter any issues, please:
1. Check the [Issues](https://github.com/yourusername/plant-leaf-classification/issues) section
2. Create a new issue with detailed description
3. Include error messages and system information

---

## 🔗 Related Links

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [Content-Based Image Retrieval](https://en.wikipedia.org/wiki/Content-based_image_retrieval)

---

**Happy Classifying! 🌿🍃**

Last Updated: April 2026
