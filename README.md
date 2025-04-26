# Social Network Ads Classification

## Overview
This project implements a **Support Vector Machine (SVM)** model to predict social network ad purchases based on user demographics. The dataset `Social_Network_Ads.csv` is used for training and evaluation.

## Data Processing
- Dataset is loaded with `pandas.read_csv()`.
- Features (`X`) and labels (`y`) are extracted.
- Data is split into **training** and **testing** sets using `train_test_split()`.
- Standardization is applied via `StandardScaler()`.

## Model Training
The classifier:
- **Support Vector Machine (SVM)** with an **RBF kernel**
- Data is scaled using `StandardScaler()`.
- Training is done using `SVC.fit()`.
- Model evaluation is performed with:
  - Confusion matrix
  - Accuracy score

## Cross Validation
- The model undergoes **10-fold cross-validation** using `cross_val_score`.
- Helps assess **stability** and **generalization**.

## **Hyperparameter Tuning with Grid Search**
To optimize model performance, **Grid Search** is employed:
- `GridSearchCV` tests multiple hyperparameter combinations.
- Parameters tuned:
  - `C` (regularization strength)
  - `gamma` (kernel coefficient for RBF)
  - Kernel type (`linear` or `rbf`)
- Uses **10-fold cross-validation** for robust selection.
- Outputs:
  - **Best Accuracy** achieved during tuning.
  - **Optimal Parameters** for the model.

## Performance Metrics
- Confusion matrix for classification evaluation.
- Accuracy score for prediction quality.
- Cross-validation results:
  - Mean accuracy
  - Standard deviation across folds.

## Results
- **Best accuracy achieved:** **90.67%** 
- **Optimal hyperparameters:** **{'C': 0.5, 'gamma': 0.6, 'kernel': 'rbf'}** 

