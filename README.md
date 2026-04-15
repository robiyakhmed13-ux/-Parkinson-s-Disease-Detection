# 🧠 Parkinson's Disease Detection

A machine learning project that detects **Parkinson's Disease** from biomedical voice measurements using a **Support Vector Machine (SVM)** classifier with feature scaling.

---

## 📌 Project Overview

Parkinson's Disease affects millions worldwide, and early diagnosis is key to effective treatment. This project uses vocal biomarkers — measurable characteristics of a person's voice — to classify whether a person has Parkinson's Disease, since vocal impairment is one of the earliest and most consistent symptoms.

| Item | Detail |
|------|--------|
| **Algorithm** | Support Vector Machine (Linear Kernel) |
| **Task** | Binary Classification |
| **Dataset** | [Parkinson's Disease Data Set – Kaggle](https://www.kaggle.com/datasets/vikasukani/parkinsons-disease-data-set) |
| **Target** | `status` — Healthy (0) / Has Parkinson's (1) |

---

## 📂 Project Structure

```
parkinsons_disease_detection/
│
├── parkinsons_disease_detection.ipynb   # Jupyter Notebook (full walkthrough)
├── parkinsons_disease_detection.py      # Clean Python script
├── requirements.txt                     # Dependencies
├── parkinsons_data.csv                  # Dataset (download from Kaggle)
├── class_distribution.png              # Target class balance plot
├── correlation_heatmap.png             # Feature correlation heatmap
├── feature_boxplots.png                # Key vocal features vs disease status
├── confusion_matrix.png                # Confusion matrix
└── README.md
```

---

## 📊 Dataset Features

The dataset contains **195 voice recordings** from 31 people, 23 of whom have Parkinson's Disease. Each row represents one voice recording described by 22 biomedical voice measurements.

| Feature Group | Features | Description |
|---------------|----------|-------------|
| **Frequency** | `MDVP:Fo(Hz)`, `MDVP:Fhi(Hz)`, `MDVP:Flo(Hz)` | Average, max, and min vocal fundamental frequency |
| **Jitter** | `MDVP:Jitter(%)`, `MDVP:Jitter(Abs)`, `MDVP:RAP`, `MDVP:PPQ`, `Jitter:DDP` | Variation in fundamental frequency |
| **Shimmer** | `MDVP:Shimmer`, `MDVP:Shimmer(dB)`, `Shimmer:APQ3`, `Shimmer:APQ5`, `MDVP:APQ`, `Shimmer:DDA` | Variation in amplitude |
| **Noise** | `NHR`, `HNR` | Noise-to-harmonics ratio measures |
| **Nonlinear** | `RPDE`, `D2` | Nonlinear dynamical complexity measures |
| **Signal fractal** | `DFA` | Detrended fluctuation analysis |
| **Frequency variation** | `spread1`, `spread2`, `PPE` | Nonlinear measures of fundamental frequency variation |
| **Target** | `status` | ✅ 0 = Healthy, 1 = Parkinson's |

---

## ⚙️ How to Run

### 1. Clone the repository
```bash
git clone https://github.com/your-username/parkinsons-disease-detection.git
cd parkinsons-disease-detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Download `parkinsons_data.csv` from [Kaggle](https://www.kaggle.com/datasets/vikasukani/parkinsons-disease-data-set) and place it in the project root.

### 4. Run
```bash
python parkinsons_disease_detection.py
```

---

## 🔄 Pipeline

```
Raw CSV Data (195 voice recordings)
    │
    ▼
EDA — Class distribution, correlation heatmap, feature boxplots
    │
    ▼
Feature / Target Split (drop name & status)
    │
    ▼
Train / Test Split (80% / 20%, stratified)
    │
    ▼
Feature Scaling — StandardScaler (fit on train only)
    │
    ▼
SVM Training (Linear Kernel)
    │
    ▼
Accuracy + Classification Report + Confusion Matrix
    │
    ▼
Single-patient Parkinson's Prediction (with scaling applied)
```

---

## ⚠️ Important: Why Feature Scaling?

SVM is sensitive to the scale of features. Without scaling, features with large numerical ranges (e.g. `MDVP:Fo(Hz)` up to ~260) would dominate those with tiny ranges (e.g. `MDVP:Jitter(Abs)` near 0.00006).

`StandardScaler` transforms each feature to have **mean = 0** and **standard deviation = 1**, ensuring all features contribute equally to the SVM decision boundary.

> The scaler is **fit on training data only** and then applied to both train and test sets — this prevents data leakage.

---

## 📈 Results

| Split | Accuracy |
|-------|----------|
| Training | ~88% |
| Test | ~87% |

---

## 🔮 Sample Prediction

```python
# 22 biomedical voice measurement values
sample = (
    91.904, 115.871, 86.292, 0.00540, 0.00006, 0.00281, 0.00336, 0.00844,
    0.02752, 0.249, 0.01424, 0.01641, 0.02214, 0.04272, 0.01141,
    21.414, 0.58339, 0.79252, -4.960234, 0.363566, 2.642476, 0.275931
)
result = predict_parkinsons(model, scaler, sample)
# Output: ⚠️  The person HAS Parkinson's disease.
```

---

## 🛠️ Tech Stack

- **Python 3.x**
- **pandas / numpy** — data processing
- **scikit-learn** — SVM, StandardScaler, train/test split, metrics
- **seaborn / matplotlib** — visualization

---

## 🚀 Future Improvements

- [ ] Try RBF kernel SVM and compare with linear kernel
- [ ] Apply cross-validation (k-fold) for more reliable accuracy estimates
- [ ] Use SHAP values to interpret which vocal features are most predictive
- [ ] Test Random Forest or XGBoost for comparison
- [ ] Build a voice recording → prediction pipeline with `librosa`

---

## ⚠️ Disclaimer

This project is for **educational purposes only** and is not a substitute for professional medical diagnosis. Always consult a qualified neurologist for Parkinson's Disease assessment.

---

## 📄 License

MIT License

---

## 🙋 Author

**[Your Name]**  
[GitHub](https://github.com/your-username) | [LinkedIn](https://linkedin.com/in/your-profile)
