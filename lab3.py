# General imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import openml as oml
from matplotlib import cm
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Download MAGA Telescope data from OpenML. You can repeat this analysis with any other OpenML classification dataset.
magic = oml.datasets.get_dataset(1120)
X, y, _, _ = magic.get_data(target=magic.default_target_attribute, dataset_format='array'); 
attribute_names = [f.name for i,f in magic.features.items()][:-1][1:]

# Quick visualization of the features (top) and the target (bottom)
magic_df = pd.DataFrame(X, columns=attribute_names)
magic_df.plot(figsize=(12,6))
# Also plot the target: 1 = background, 0 = gamma
pd.DataFrame(y).plot(figsize=(12,1))

#plt.tight_layout()
#plt.show()

# Exercise 1 Metrics

# 1. Split data (25% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 2. Train SVM with RBF kernel (default hyperparameters)
svm = SVC(kernel='rbf', probability=True, random_state=42)
svm.fit(X_train, y_train)

# 3. Predictions
y_pred = svm.predict(X_test)
y_proba = svm.predict_proba(X_test)[:, 1]

# 4. Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

# 5. Report results
print("Raw: ")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 score:", f1)
print("ROC-AUC:", auc)

#Exercise 2 Preprocessing

# 1.
from sklearn.preprocessing import StandardScaler
# Important here is to fit the scaler on the training data alone
# Then, use it to scale both the training set and test set
# This assumes that you named your training set X_train. Adapt if needed.
scaler = StandardScaler().fit(X_train)
Xs_train = scaler.transform(X_train)
Xs_test = scaler.transform(X_test)

# 2. Train SVM with RBF kernel (default hyperparameters)
svm = SVC(kernel='rbf', probability=True, random_state=42)
svm.fit(Xs_train, y_train)

# 3. Predictions
y_pred = svm.predict(Xs_test)
y_proba = svm.predict_proba(Xs_test)[:, 1]

# 4. Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

# 5. Report results
print("Scaled: ")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 score:", f1)
print("ROC-AUC:", auc)

#Exercise 3 Hyperparameter optimization
'''

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform

# 1. Define parameter distribution (log scale 2^-12 to 2^12)
param_dist = {
    'C': loguniform(2**-12, 2**12),
    'gamma': loguniform(2**-12, 2**12)
}

# 2. Base model
svm = SVC(kernel='rbf', probability=True, random_state=42)

# 3. Random Search (50 iterations, 3-fold CV, optimize AUC)
random_search = RandomizedSearchCV(
    svm,
    param_distributions=param_dist,
    n_iter=50,
    scoring='roc_auc',
    cv=3,
    random_state=42,
    n_jobs=-1
)

# 4. Fit on scaled training data
random_search.fit(Xs_train, y_train)

# 5. Best parameters
print("Best parameters:", random_search.best_params_)
print("Best CV AUC:", random_search.best_score_)
print("Best estimator:", random_search.best_estimator_)

# 6. Visualization
results = random_search.cv_results_

C_vals = np.array(results['param_C'], dtype=float)
gamma_vals = np.array(results['param_gamma'], dtype=float)
scores = results['mean_test_score']

plt.figure(figsize=(8,6))
scatter = plt.scatter(C_vals, gamma_vals, c=scores)

plt.xscale('log')
plt.yscale('log')

plt.xlabel('C')
plt.ylabel('gamma')
plt.title('Random Search Samples (C vs gamma)')
plt.colorbar(scatter, label='Mean CV AUC')

# 7. Another try on best
best_svm = random_search.best_estimator_
y_pred = best_svm.predict(Xs_test)
y_proba = best_svm.predict_proba(Xs_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print("Optimized model (test set):")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 score:", f1)
print("ROC-AUC:", auc)

#plt.show()

'''

#Exercise 4 Threshold calibration

from sklearn.metrics import precision_recall_curve

# scaled default 
svm_default = SVC(kernel='rbf', probability=True, random_state=42)
svm_default.fit(Xs_train, y_train)

y_proba = svm_default.predict_proba(Xs_test)[:, 1]

precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

plt.figure(figsize=(8,6))
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve (Scaled Default SVM)")

for t in [0.4, 0.35, 0.3, 0.25]:
    y_pred_new = (y_proba >= t).astype(int)
    print("\nThreshold:", t)
    print("Precision:", precision_score(y_test, y_pred_new))
    print("Recall:", recall_score(y_test, y_pred_new))

#plt.show()


#Exercise 5 Cost function

from sklearn.metrics import roc_curve

fpr, tpr, roc_thresholds = roc_curve(y_test, y_proba)

plt.figure(figsize=(8,6))
plt.plot(roc_thresholds)

# Cost = 2*FN + 1*FP
# FN = (1 - TPR)
# FP = FPR

cost = 2*(1 - tpr) + fpr

optimal_idx = np.argmin(cost)
optimal_threshold = roc_thresholds[optimal_idx]

print("Optimal threshold (cost-sensitive):", optimal_threshold)

y_pred_opt = (y_proba >= optimal_threshold).astype(int)

print("Accuracy:", accuracy_score(y_test, y_pred_opt))
print("Precision:", precision_score(y_test, y_pred_opt))
print("Recall:", recall_score(y_test, y_pred_opt))
print("F1 score:", f1_score(y_test, y_pred_opt))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))

plt.show()