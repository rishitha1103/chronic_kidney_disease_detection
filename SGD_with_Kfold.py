import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve

# **1. Load dataset**
data = pd.read_csv('S:/Documents/6th sem/ckd/kidney_disease.csv')

# **2. Data Cleaning**
data.columns = data.columns.str.strip()  # Remove whitespace from column names
data = data.applymap(lambda x: x.strip() if isinstance(x, str) else x)  # Strip whitespace from values
data.replace(r'^\s*\?$', np.nan, regex=True, inplace=True)  # Replace '?' with NaN
data.replace(r'^\s*\\t\?$', np.nan, regex=True, inplace=True)  # Replace '\t?' with NaN
data.fillna(data.mean(numeric_only=True), inplace=True)  # Replace missing values with mean for numerical data
data.fillna('Unknown', inplace=True)  # Replace missing values in categorical data

# **3. Encode Categorical Features**
categorical_columns = data.select_dtypes(include=['object']).columns  # Find categorical columns
le = LabelEncoder()
for col in categorical_columns:
    data[col] = le.fit_transform(data[col].astype(str))  # Encode categorical values

# **4. Define Features (X) and Target (y)**
target_column = 'classification'
if target_column not in data.columns:
    raise KeyError(f"Target column '{target_column}' not found in dataset.")

X = data.drop(['id', target_column], axis=1, errors='ignore').values  # Drop 'id' and target column
y = data[target_column].values

# **5. K-Fold Cross-Validation**
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # 5-Fold Cross-Validation
fold = 1  # Track fold number
accuracies = []  # Store accuracy of each fold
all_classification_reports = []
feature_importances = []
roc_curves = []
pr_curves = []

for train_index, test_index in kf.split(X, y):
    print(f"\nFold {fold}...")

    # Split data into training and testing sets for the current fold
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Train SGD Classifier
    sgd = SGDClassifier(loss='log_loss', max_iter=1000, random_state=42)
    sgd.fit(X_train, y_train)

    # Evaluate the model
    y_pred = sgd.predict(X_test)
    y_scores = sgd.decision_function(X_test)  # For ROC and PR curves
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

    # Save classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    all_classification_reports.append(pd.DataFrame(report).transpose())

    # Save feature importance
    feature_importances.append(np.abs(sgd.coef_).flatten())

    # Print metrics for the fold
    print(f"Accuracy for Fold {fold}: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))

    # **Confusion Matrix Heatmap**
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix Heatmap (Fold {fold})")
    plt.show()

    # **ROC Curve**
    if len(np.unique(y)) == 2:  # Binary classification check
        fpr, tpr, _ = roc_curve(y_test, y_scores)
        roc_auc = auc(fpr, tpr)
        roc_curves.append((fpr, tpr, roc_auc))

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve (Fold {fold})")
        plt.legend()
        plt.show()

    # **Precision-Recall Curve**
    precision, recall, _ = precision_recall_curve(y_test, y_scores)
    pr_curves.append((recall, precision))

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.', color='red', label='Precision-Recall Curve')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve (Fold {fold})")
    plt.legend()
    plt.show()

    fold += 1

# **6. Overall Performance Metrics**
print("\n--- Overall Performance ---")
print(f"Mean Accuracy: {np.mean(accuracies):.4f}")
print(f"Standard Deviation of Accuracy: {np.std(accuracies):.4f}")

# **7. Boxplot of Cross-Validation Accuracies**
plt.figure(figsize=(8, 6))
sns.boxplot(data=accuracies, palette="coolwarm")
plt.ylabel("Accuracy")
plt.title("Cross-Validation Accuracy Distribution")
plt.show()

# **8. Mean Feature Importance Bar Plot**
mean_feature_importance = np.mean(feature_importances, axis=0)
feature_names = data.drop(['id', target_column], axis=1, errors='ignore').columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': mean_feature_importance})
importance_df = importance_df.sort_values(by="Importance", ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x=importance_df['Importance'], y=importance_df['Feature'], palette="coolwarm")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.title("Mean Feature Importance Across Folds")
plt.show()

# **9. Overall ROC Curve**
if len(np.unique(y)) == 2:
    plt.figure(figsize=(8, 6))
    for fpr, tpr, auc_value in roc_curves:
        plt.plot(fpr, tpr, alpha=0.5)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Overall ROC Curve Across Folds")
    plt.show()

# **10. Overall Precision-Recall Curve**
plt.figure(figsize=(8, 6))
for recall, precision in pr_curves:
    plt.plot(recall, precision, alpha=0.5)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Overall Precision-Recall Curve Across Folds")
plt.show()

# **11. Overall Classification Report Table**
final_report = pd.concat(all_classification_reports).groupby(level=0).mean()
print("\n--- Overall Classification Report ---")
print(final_report)
