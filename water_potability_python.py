# Import necessary libraries
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Import heatmap visualization
import seaborn as sns
# Import additional necessary libraries for ROC curve
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Load the datasets
X_train = pd.read_csv('Dataset/X_train.csv')
X_val = pd.read_csv('Dataset/X_val.csv')
y_train = pd.read_csv('Dataset/y_train.csv')
y_val = pd.read_csv('Dataset/y_val.csv')

# Set up the parameter grid
param_grid = {
    'learning_rate': [0.01, 0.1],
    'n_estimators': [500, 1000],
    'max_depth': [5, 10],
    'min_child_weight': [2, 4],
    'gamma': [0.5, 1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# Initialize the XGBoost Classifier
xgb = XGBClassifier(objective='binary:logistic', nthread=-2, random_state=42)

# Set up RandomizedSearchCV
random_search = RandomizedSearchCV(xgb, param_distributions=param_grid, n_iter=50, scoring='accuracy', n_jobs=-1, cv=3, verbose=3, random_state=42)

# Fit the model
random_search.fit(X_train, y_train.values.ravel())

# Best estimator
best_xgb = random_search.best_estimator_

# Evaluate the best estimator on validation data
y_val_pred = best_xgb.predict(X_val)
acc_val = accuracy_score(y_val, y_val_pred)
print(f"Best Model Parameters: {random_search.best_params_}\nValidation Accuracy: {acc_val:.4f}")

# Load the test dataset and evaluate the model
X_test = pd.read_csv('Dataset/X_test.csv')
y_test = pd.read_csv('Dataset/y_test.csv')

y_pred = best_xgb.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")

# Generate and print the classification report
report = classification_report(y_test, y_pred)
print(f"Classification Report:\n{report}")

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plotting the confusion matrix
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

# Predict probabilities for the test set
y_probs = best_xgb.predict_proba(X_test)[:, 1]

# Calculate the ROC AUC score
roc_auc = roc_auc_score(y_test, y_probs)
print(f"ROC AUC Score: {roc_auc:.4f}")

# Compute ROC curve data
fpr, tpr, _ = roc_curve(y_test, y_probs)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


