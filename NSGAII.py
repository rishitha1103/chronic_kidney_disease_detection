import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import LabelEncoder

# **1. Load and preprocess dataset**
data = pd.read_csv('S:/Documents/6th sem/ckd/kidney_disease.csv')

# Clean column names and values
data.columns = data.columns.str.strip()
data = data.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# Replace problematic values with NaN and handle missing data
data.replace(r'^\s*\?$', np.nan, regex=True, inplace=True)
data.fillna(data.mean(numeric_only=True), inplace=True)
data.fillna('Unknown', inplace=True)

# Encode categorical columns
categorical_columns = data.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in categorical_columns:
    data[col] = le.fit_transform(data[col].astype(str))

# Define features and target
target_column = 'classification'
X = data.drop(['id', target_column], axis=1, errors='ignore').values
y = data[target_column].values

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# **2. Define NSGA-II Problem**
class MultiObjectiveFeatureSelection(ElementwiseProblem):
    def __init__(self):
        super().__init__(
            n_var=X_train.shape[1],  # Number of features
            n_obj=2,  # Objectives: maximize accuracy, minimize features
            n_constr=0,
            xl=0,
            xu=1,
            type_var=np.bool_,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        selected_features = np.where(x == 1)[0]

        if len(selected_features) == 0:
            out["F"] = [1.0, X_train.shape[1]]
            return

        X_train_selected = X_train[:, selected_features]
        X_test_selected = X_test[:, selected_features]

        model = SGDClassifier(loss="log_loss", max_iter=1000, random_state=42)
        model.fit(X_train_selected, y_train)

        y_pred = model.predict(X_test_selected)
        accuracy = accuracy_score(y_test, y_pred)

        obj1 = -accuracy  # Negative accuracy (minimizing negative accuracy = maximizing accuracy)
        obj2 = len(selected_features)

        out["F"] = [obj1, obj2]

# **3. Run NSGA-II Optimization**
algorithm = NSGA2(
    pop_size=50,
    sampling=BinaryRandomSampling(),
    crossover=SimulatedBinaryCrossover(prob=0.9, eta=15),
    mutation=BitflipMutation(prob=0.1),
    eliminate_duplicates=True,
)

problem = MultiObjectiveFeatureSelection()
res = minimize(problem, algorithm, termination=("n_gen", 50), seed=42, verbose=True)

# **4. Display Results**
print("\n--- Results ---")
for i, sol in enumerate(res.X):
    selected_features = np.where(sol == 1)[0]
    print(f"Solution {i + 1}: Selected Features = {selected_features}, Objectives = {res.F[i]}")

# **5. Find the best solution (highest accuracy)**
best_solution = res.X[np.argmax(res.F[:, 0])]  # Select the solution with the highest accuracy
selected_features = np.where(best_solution == 1)[0]

X_train_selected = X_train[:, selected_features]
X_test_selected = X_test[:, selected_features]

# Train model using the best feature set
model = SGDClassifier(loss="log_loss", max_iter=1000, random_state=42)
model.fit(X_train_selected, y_train)
y_pred = model.predict(X_test_selected)

# **6. Print Classification Report**
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# **7. Accuracy**
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")

# **8. Confusion Matrix Heatmap**
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix Heatmap (Best Solution)")
plt.show()

# **9. ROC Curve**
if len(np.unique(y)) == 2:
    y_scores = model.decision_function(X_test_selected)
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Best Solution)")
    plt.legend()
    plt.show()

# **10. Precision-Recall Curve**
precision, recall, _ = precision_recall_curve(y_test, y_scores)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, marker='.', color='red', label='Precision-Recall Curve')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve (Best Solution)")
plt.legend()
plt.show()

# **11. Pareto Front: Accuracy vs. Number of Features**
plt.figure(figsize=(8, 6))
plt.scatter(-res.F[:, 0], res.F[:, 1], c='blue', label="Solutions")
plt.scatter(-max(res.F[:, 0]), min(res.F[:, 1]), color='red', marker='x', s=100, label="Best Solution")
plt.xlabel("Accuracy")
plt.ylabel("Number of Features")
plt.title("Pareto Front: Accuracy vs. Number of Features")
plt.legend()
plt.show()

# **12. Feature Selection Distribution**
feature_selection_counts = np.sum(res.X, axis=0)
feature_names = data.drop(['id', target_column], axis=1, errors='ignore').columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Selection Count': feature_selection_counts})
importance_df = importance_df.sort_values(by="Selection Count", ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x=importance_df['Selection Count'], y=importance_df['Feature'], palette="coolwarm")
plt.xlabel("Selection Count")
plt.ylabel("Feature")
plt.title("Feature Selection Frequency Across Solutions")
plt.show()
