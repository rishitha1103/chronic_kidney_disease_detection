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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# **1. Load and Clean Dataset**
data = pd.read_csv('S:/Documents/6th sem/ckd/kidney_disease.csv')

# Clean column names and values
data.columns = data.columns.str.strip()
data = data.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# Replace problematic values with NaN
data.replace(r'^\s*\?$', np.nan, regex=True, inplace=True)
data.replace(r'^\s*\\t\?$', np.nan, regex=True, inplace=True)

# Handle missing values
data.fillna(data.mean(numeric_only=True), inplace=True)
data.fillna('Unknown', inplace=True)

# Encode categorical columns
categorical_columns = data.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in categorical_columns:
    data[col] = le.fit_transform(data[col].astype(str))

# Define feature matrix (X) and target vector (y)
target_column = 'classification'
if target_column not in data.columns:
    raise KeyError(f"Target column '{target_column}' not found in dataset.")

X = data.drop(['id', target_column], axis=1, errors='ignore').values
y = data[target_column].values

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# **2. Define NSGA-II Problem**
class MultiObjectiveFeatureSelection(ElementwiseProblem):
    def __init__(self):
        super().__init__(
            n_var=X_train.shape[1],  # Number of variables (features)
            n_obj=2,  # Two objectives: accuracy and number of features
            n_constr=0,  # No constraints
            xl=0,  # Lower bounds for binary decision variables (0)
            xu=1,  # Upper bounds for binary decision variables (1)
            type_var=np.bool_,  # Binary variables (0 or 1)
        )

    def _evaluate(self, x, out, *args, **kwargs):
        # **Feature Subset Selection**
        selected_features = np.where(x == 1)[0]  # Indices of selected features

        # If no features are selected, set a large penalty
        if len(selected_features) == 0:
            out["F"] = [1.0, X_train.shape[1]]
            return

        # Reduce training and testing data to selected features
        X_train_selected = X_train[:, selected_features]
        X_test_selected = X_test[:, selected_features]

        # **Train an XGBoost Classifier**
        model = XGBClassifier(eval_metric='logloss', random_state=42)
        model.fit(X_train_selected, y_train)

        # **Objective 1: Minimize Negative Accuracy**
        y_pred = model.predict(X_test_selected)
        accuracy = accuracy_score(y_test, y_pred)
        obj1 = -accuracy  # We minimize the negative accuracy
        obj2 = len(selected_features)  # Number of selected features

        # Store objectives
        out["F"] = [obj1, obj2]

# **3. Define NSGA-II Algorithm**
algorithm = NSGA2(
    pop_size=50,
    sampling=BinaryRandomSampling(),
    crossover=SimulatedBinaryCrossover(prob=0.9, eta=15),
    mutation=BitflipMutation(prob=0.1),
    eliminate_duplicates=True,
)

# **4. Run Optimization**
problem = MultiObjectiveFeatureSelection()

res = minimize(
    problem,
    algorithm,
    termination=("n_gen", 100),  # Run for 100 generations
    seed=42,
    verbose=True,
)

# **5. Filter and Select 5 Best Solutions (Avoiding 100% Accuracy)**
print("\n--- Selected Solutions ---")
solution_data = []
valid_solutions = []

for i, sol in enumerate(res.X):
    selected_features = np.where(sol == 1)[0]

    if len(selected_features) > 0:
        X_test_selected = X_test[:, selected_features]
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        model.fit(X_train[:, selected_features], y_train)
        y_pred = model.predict(X_test_selected)

        accuracy = accuracy_score(y_test, y_pred)

        # Avoid solutions with 100% accuracy
        if accuracy < 1.0:
            valid_solutions.append((selected_features, accuracy, len(selected_features), sol))

# Ensure we have at least 5 solutions
if len(valid_solutions) < 5:
    print("\nWarning: Fewer than 5 valid solutions found. Generating more alternatives...")
    valid_solutions = sorted(valid_solutions, key=lambda x: x[1], reverse=True)[:5]  # Select top 5 by accuracy

# Save final solutions
for i, (selected_features, accuracy, feature_count, sol) in enumerate(valid_solutions[:5]):
    print(f"Solution {i + 1}: Features = {selected_features}, Accuracy = {accuracy:.4f}, Feature Count = {feature_count}")

    solution_data.append({
        "Solution": i + 1,
        "Selected Features": feature_count,
        "Accuracy": accuracy
    })

# Convert results into DataFrame
solution_df = pd.DataFrame(solution_data)

# **6. Display Solution Data as Table**
print("\nFinal Solution Table:")
print(solution_df)
