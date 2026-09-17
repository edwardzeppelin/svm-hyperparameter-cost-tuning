# MAGIC Telescope SVM Classification & Cost Optimization

A machine learning pipeline evaluating Support Vector Classifiers (SVC) on the MAGIC Gamma Telescope dataset from OpenML. The project covers data standardization, hyperparameter optimization, threshold calibration, and cost-sensitive decision tuning.

## Features

* **Baseline Classification**: Initial evaluation using RBF-kernel SVM on raw dataset features.
* **Feature Scaling**: Data standardization via `StandardScaler` to optimize SVM distance metrics and prevent data leakage.
* **Hyperparameter Tuning**: Logarithmic search over parameters $C$ and $\gamma$ using `RandomizedSearchCV` (cross-validated by ROC-AUC).
* **Threshold Calibration**: Precision-Recall curve analysis to tune decision boundaries.
* **Cost-Sensitive Learning**: Custom loss minimization where False Negatives are penalized twice as severely as False Positives ($Cost = 2 \cdot FN + FP$).

## Pipeline Structure

1. **Raw Evaluation**: Evaluate initial baseline performance ($Accuracy$, $Precision$, $Recall$, $F1$, $ROC\text{-}AUC$).
2. **Preprocessing**: Fit `StandardScaler` exclusively on training data and transform train/test splits.
3. **Random Search**: Sample $C, \gamma \in [2^{-12}, 2^{12}]$ over 50 iterations with 3-fold cross-validation.
4. **Precision-Recall Analysis**: Test probability thresholds ($t \in [0.25, 0.40]$).
5. **Cost Minimization**: Compute optimal threshold from ROC curve coordinates minimizing overall cost.

## Tech Stack

| Component | Library / Framework |
| :--- | :--- |
| **Language** | Python 3.x |
| **Dataset Source** | OpenML (`openml`) |
| **Data Processing** | NumPy, Pandas |
| **ML Engine** | Scikit-Learn (`SVC`, `StandardScaler`, `RandomizedSearchCV`) |
| **Visualization** | Matplotlib |

## Quickstart

### Prerequisites

```bash
pip install numpy pandas matplotlib openml scikit-learn
```
