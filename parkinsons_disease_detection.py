# =============================================================================
# Parkinson's Disease Detection using Support Vector Machine (SVM)
# Author: [Your Name]
# Dataset: https://www.kaggle.com/datasets/vikasukani/parkinsons-disease-data-set
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)


# =============================================================================
# 1. Data Loading
# =============================================================================

def load_data(filepath: str) -> pd.DataFrame:
    """Load the Parkinson's disease dataset."""
    df = pd.read_csv(filepath)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"\nMissing values:\n{df.isnull().sum().sum()} total missing")
    print(f"\nTarget distribution (status):\n{df['status'].value_counts()}")
    print("  0 = Healthy (no Parkinson's) | 1 = Has Parkinson's")
    return df


# =============================================================================
# 2. Exploratory Data Analysis
# =============================================================================

def plot_eda(df: pd.DataFrame) -> None:
    """Visualise target balance and correlation heatmap."""

    # Target class distribution
    plt.figure(figsize=(5, 4))
    sns.countplot(x='status', data=df, palette=['steelblue', 'salmon'])
    plt.title("Parkinson's Disease — Class Distribution")
    plt.xticks([0, 1], ['Healthy (0)', "Parkinson's (1)"])
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("class_distribution.png", dpi=150)
    plt.show()
    print("Class distribution saved as 'class_distribution.png'")

    # Correlation heatmap (drop name column)
    numeric_df = df.drop(columns=['name'])
    plt.figure(figsize=(18, 14))
    sns.heatmap(
        numeric_df.corr(), annot=True, fmt='.1f',
        cmap='Blues', square=True, annot_kws={'size': 6}
    )
    plt.title("Feature Correlation Heatmap", fontsize=14)
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png", dpi=150)
    plt.show()
    print("Heatmap saved as 'correlation_heatmap.png'")

    # Box plots for a few key vocal features
    key_features = ['MDVP:Fo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Shimmer', 'HNR']
    available = [f for f in key_features if f in df.columns]
    if available:
        fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 5))
        if len(available) == 1:
            axes = [axes]
        for ax, feat in zip(axes, available):
            sns.boxplot(x='status', y=feat, data=df,
                        palette=['steelblue', 'salmon'], ax=ax)
            ax.set_title(feat)
            ax.set_xticklabels(['Healthy', "Parkinson's"])
        plt.suptitle("Key Vocal Features vs Disease Status", fontsize=14)
        plt.tight_layout()
        plt.savefig("feature_boxplots.png", dpi=150)
        plt.show()
        print("Box plots saved as 'feature_boxplots.png'")


# =============================================================================
# 3. Feature / Target Split
# =============================================================================

def split_features_target(df: pd.DataFrame):
    """Drop name and status; return features X and target Y."""
    X = df.drop(columns=['name', 'status'], axis=1)
    Y = df['status']
    print(f"\nFeatures: {X.shape} | Target: {Y.shape}")
    return X, Y


# =============================================================================
# 4. Feature Scaling
# =============================================================================

def scale_features(X_train, X_test):
    """
    Apply StandardScaler: fit on training data only,
    transform both train and test sets to prevent data leakage.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    print("Feature scaling complete (StandardScaler).")
    return X_train_scaled, X_test_scaled, scaler


# =============================================================================
# 5. Train / Test Split
# =============================================================================

def split_data(X, Y, test_size=0.2, random_state=2):
    """
    Split into train (80%) and test (20%), stratified by class.
    Note: the original notebook used train_size=0.2 (i.e. 80% test),
    which is corrected here to the standard test_size=0.2 convention.
    """
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y,
        test_size=test_size,
        stratify=Y,
        random_state=random_state
    )
    print(f"Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")
    return X_train, X_test, Y_train, Y_test


# =============================================================================
# 6. Model Training
# =============================================================================

def train_model(X_train_scaled, Y_train):
    """Train a linear-kernel SVM classifier."""
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train_scaled, Y_train)
    print("SVM model training complete.")
    return model


# =============================================================================
# 7. Model Evaluation
# =============================================================================

def evaluate_model(model, X_train_scaled, Y_train,
                   X_test_scaled, Y_test) -> None:
    """Accuracy, classification report, and confusion matrix."""
    train_preds = model.predict(X_train_scaled)
    test_preds  = model.predict(X_test_scaled)

    print(f"\nTraining Accuracy : {accuracy_score(Y_train, train_preds):.4f}")
    print(f"Test     Accuracy : {accuracy_score(Y_test,  test_preds):.4f}")

    print("\nClassification Report (Test Set):")
    print(classification_report(
        Y_test, test_preds,
        target_names=["Healthy", "Parkinson's"]
    ))

    # Confusion matrix
    cm = confusion_matrix(Y_test, test_preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                xticklabels=["Healthy", "Parkinson's"],
                yticklabels=["Healthy", "Parkinson's"])
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.show()
    print("Confusion matrix saved as 'confusion_matrix.png'")


# =============================================================================
# 8. Predictive System
# =============================================================================

def predict_parkinsons(model, scaler, input_data: tuple) -> str:
    """
    Predict whether a person has Parkinson's disease.

    Parameters
    ----------
    model      : trained SVM model
    scaler     : fitted StandardScaler (must scale input before predicting)
    input_data : tuple of 22 biomedical voice measurement values:
        (MDVP:Fo(Hz), MDVP:Fhi(Hz), MDVP:Flo(Hz), MDVP:Jitter(%),
         MDVP:Jitter(Abs), MDVP:RAP, MDVP:PPQ, Jitter:DDP,
         MDVP:Shimmer, MDVP:Shimmer(dB), Shimmer:APQ3, Shimmer:APQ5,
         MDVP:APQ, Shimmer:DDA, NHR, HNR, RPDE, DFA,
         spread1, spread2, D2, PPE)
    """
    arr    = np.asarray(input_data).reshape(1, -1)
    arr_scaled = scaler.transform(arr)
    prediction = model.predict(arr_scaled)

    if prediction[0] == 0:
        return "✅ The person does NOT have Parkinson's disease."
    else:
        return "⚠️  The person HAS Parkinson's disease."


# =============================================================================
# Main Pipeline
# =============================================================================

if __name__ == "__main__":
    DATA_PATH = "parkinsons_data.csv"   # update path if needed

    # 1. Load
    df = load_data(DATA_PATH)
    print("\nFirst 5 rows:\n", df.head())
    print("\nStatistical summary:\n", df.describe())

    # 2. EDA
    plot_eda(df)

    # 3. Features & Target
    X, Y = split_features_target(df)

    # 4. Train / Test Split
    X_train, X_test, Y_train, Y_test = split_data(X, Y)

    # 5. Scaling (fit on train, transform both)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # 6. Train
    model = train_model(X_train_scaled, Y_train)

    # 7. Evaluate
    evaluate_model(model, X_train_scaled, Y_train, X_test_scaled, Y_test)

    # 8. Predict a sample patient
    sample = (
        91.904, 115.871, 86.292, 0.00540, 0.00006, 0.00281, 0.00336, 0.00844,
        0.02752, 0.24900, 0.01424, 0.01641, 0.02214, 0.04272, 0.01141,
        21.414, 0.583390, 0.792520, -4.960234, 0.363566, 2.642476, 0.275931
    )
    result = predict_parkinsons(model, scaler, sample)
    print(f"\nSample Prediction:\n{result}")
