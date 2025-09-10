import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc, precision_recall_curve

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

X = data.drop(['id', target_column], axis=1, errors='ignore')  # Drop 'id' and target column
y = data[target_column].values

# **5. Split Dataset**
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# **6. Train SGD Classifier**
sgd = SGDClassifier(loss='log_loss', max_iter=1000, random_state=42)
sgd.fit(X_train, y_train)

# **7. Evaluate Model**
y_pred = sgd.predict(X_test)
y_scores = sgd.decision_function(X_test)  # Get decision scores for ROC curve

print("\n--- Model Evaluation ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# **8. Generate Visualizations**

## **A. Heatmap of Feature Correlations**
plt.figure(figsize=(10, 8))
sns.heatmap(pd.DataFrame(X).corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

## **B. Confusion Matrix Heatmap**
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix Heatmap")
plt.show()

## **C. Classification Report as Table**
report = classification_report(y_test, y_pred, output_dict=True)
df_report = pd.DataFrame(report).transpose()
print("\n--- Classification Report Table ---")
print(df_report)

## **D. Feature Importance Bar Plot**
feature_importance = np.abs(sgd.coef_).flatten()
feature_names = data.drop(['id', target_column], axis=1, errors='ignore').columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
importance_df = importance_df.sort_values(by="Importance", ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x=importance_df['Importance'], y=importance_df['Feature'], palette="coolwarm")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.title("Feature Importance (SGD Coefficients)")
plt.show()

## **E. ROC Curve (without scikitplot)**
if len(np.unique(y)) == 2:  # Check if binary classification
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

## **F. Precision-Recall Curve (without scikitplot)**
precision, recall, _ = precision_recall_curve(y_test, y_scores)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, marker='.', color='red', label='Precision-Recall Curve')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.show()
