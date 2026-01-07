# Fraud Detection Analysis

A machine learning project for detecting fraudulent financial transactions using Python and data analysis techniques.

## Overview

This project analyzes a dataset of financial transactions to identify patterns and characteristics of fraudulent activity. The goal is to build predictive models that can detect fraud in real-time transaction systems.

## Dataset

- **Source**: Kaggle - Fraud Detection Dataset
- **Size**: ~6.36M transactions
- **Fraud Rate**: Only 0.13% of transactions (highly imbalanced dataset)
- **Features**:
  - `type`: Transaction type (PAYMENT, TRANSFER, CASH_OUT, DEBIT, CASH_IN)
  - `amount`: Transaction amount
  - `nameOrig`: Customer who initiated the transaction
  - `oldbalanceOrg`: Initial balance of origin account
  - `newbalanceOrig`: New balance of origin account
  - `nameDest`: Recipient of the transaction
  - `oldbalanceDest`: Initial balance of destination account
  - `newbalanceDest`: New balance of destination account
  - `isFraud`: Target variable (1 = fraud, 0 = legitimate)
  - `isFlaggedFraud`: Transactions flagged by the system

## Key Findings

- **Class Imbalance**: Only 0.13% of transactions are fraudulent
- **Transaction Types**: Certain transaction types (TRANSFER, CASH_OUT) are more susceptible to fraud
- **Balance Anomalies**: Negative balance differences can indicate suspicious activity
- **Amount Distribution**: Transaction amounts follow a highly skewed distribution

## Analysis Steps

1. **Data Loading**: Download and load the fraud detection dataset from Kaggle
2. **Exploratory Data Analysis (EDA)**:
   - Examine data structure and statistics
   - Check for missing values and data quality
   - Analyze class distribution and imbalance
3. **Visualization**:
   - Transaction type distribution
   - Fraud rates by transaction type
   - Amount distribution analysis
   - Temporal fraud patterns
4. **Feature Engineering**:
   - Balance difference calculations
   - Anomaly detection in account balances
5. **Model Development** (to be implemented):
   - Handling class imbalance (SMOTE, undersampling)
   - Feature selection and encoding
   - Model training and evaluation
   - Performance metrics (precision, recall, F1-score, AUC-ROC)

## Technologies Used

- **Python 3.x**
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Matplotlib & Seaborn**: Data visualization
- **Scikit-learn**: Machine learning (upcoming)
- **Kagglehub**: Dataset management
