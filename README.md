# Fraud Detection Analysis

A complete machine learning project for detecting fraudulent financial transactions using Python, data analysis, and machine learning techniques. Perfect for beginners learning data science!

## 🎯 Overview

This project analyzes a dataset of financial transactions to identify patterns and characteristics of fraudulent activity. The goal is to build predictive models that can detect fraud in real-time transaction systems.

## 📊 Dataset

- **Source**: [Kaggle - Fraud Detection Dataset](https://www.kaggle.com/datasets/amanalisiddiqui/fraud-detection-dataset)
- **Size**: ~6.36M transactions (large dataset - may take time to download)
- **Fraud Rate**: Only 0.13% of transactions (8,213 fraudulent out of 6.36M total)
- **File Size**: ~500MB (ensure you have enough disk space)

### Features Explained

| Feature          | Description               | Example                                     |
| ---------------- | ------------------------- | ------------------------------------------- |
| `type`           | Transaction type          | PAYMENT, TRANSFER, CASH_OUT, DEBIT, CASH_IN |
| `amount`         | Transaction amount        | 9839.64                                     |
| `nameOrig`       | Sender account ID         | C1231006815                                 |
| `oldbalanceOrg`  | Sender's balance before   | 170136.0                                    |
| `newbalanceOrig` | Sender's balance after    | 160296.36                                   |
| `nameDest`       | Receiver account ID       | M1979787155                                 |
| `oldbalanceDest` | Receiver's balance before | 0.0                                         |
| `newbalanceDest` | Receiver's balance after  | 0.0                                         |
| `isFraud`        | **Target**: Is it fraud?  | 0 (no) or 1 (yes)                           |
| `isFlaggedFraud` | Old system's fraud flag   | 0 (no) or 1 (yes)                           |

```

The notebook is divided into clear sections:

### Part 1: Data Exploration (Steps 1-10)
**What happens**: Load data, check structure, understand what we're working with
**Why**: Before building models, we need to understand our data
**Key insight**: Only 0.13% fraud - this is extremely imbalanced!

### Part 2: Visualization (Steps 11-15)
**What happens**: Create charts showing fraud patterns
**Why**: Visualizations reveal insights hidden in raw numbers
**Key insight**: TRANSFER and CASH_OUT have highest fraud rates

### Part 3: Feature Engineering (Steps 16-32)
**What happens**: Create new features (balance differences, anomaly detection)
**Why**: New features can improve model performance dramatically
**Key insight**: Accounts draining to zero are highly suspicious

### Part 4: Machine Learning (Steps 33-47)
**What happens**: Build, train, and evaluate fraud detection model
**Why**: Automate fraud detection at scale
**Key result**: 95% recall (catch 95% of frauds) with 94.5% accuracy

## 📈 Model Performance

After training, our model achieves:

| Metric | Score | What it means |
|--------|-------|---------------|
| **Accuracy** | 94.5% | Overall correct predictions |
| **Precision (Fraud)** | 2% | Of flagged frauds, 2% are real (many false alarms) |
| **Recall (Fraud)** | 95% | We catch 95% of all frauds! |
| **F1-Score (Fraud)** | 0.04 | Balance of precision and recall |

### What does this mean?

✅ **Good news**:
- We catch 95% of fraudulent transactions (only miss 5%)
- Much better than the old system (which caught almost nothing)

⚠️ **Trade-off**:
- We generate ~103,000 false alarms on 1.9M test transactions
- This is expected with `class_weight="balanced"` on imbalanced data
- In production, you'd adjust the threshold based on capacity to review alerts

## 💻 Using the Trained Model

After running the notebook, you'll have `fraud_detection_pipeline.pkl` - a trained model ready for predictions!

## 🔑 Key Findings

1. **Class Imbalance is Severe**
   - Only 0.13% of transactions are fraudulent
   - Need special techniques: `class_weight="balanced"`, SMOTE, or undersampling

2. **Transaction Types Matter**
   - TRANSFER and CASH_OUT have highest fraud rates
   - PAYMENT, DEBIT, and CASH_IN are much safer

3. **Balance Anomalies are Red Flags**
   - Negative balance differences (balance increases when sending money)
   - Accounts draining to exactly zero
   - These patterns correlate with fraud

4. **Feature Engineering is Powerful**
   - Created `balanceDiffOrig` and `balanceDiffDest` features
   - These engineered features capture suspicious patterns
   - Often more important than algorithm choice!

5. **Logistic Regression Works as Baseline**
   - Simple, fast, interpretable
   - Achieves 95% recall with proper class weighting
   - Room for improvement with advanced models

## 🛠 Technologies Used

| Technology | Version | Purpose |
|------------|---------|---------|
| Python | 3.8+ | Programming language |
| Pandas | Latest | Data manipulation and analysis |
| NumPy | Latest | Numerical computations |
| Matplotlib | Latest | Basic plotting and visualization |
| Seaborn | Latest | Advanced statistical visualizations |
| Scikit-learn | Latest | Machine learning algorithms |
| Kagglehub | Latest | Dataset management |
| Joblib | Latest | Model serialization |
| Jupyter | Latest | Interactive notebook environment |
```
