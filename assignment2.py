import sys
import types
import sklearn.ensemble
sys.modules['sklearn.ensembles'] = sklearn.ensemble  # Patch for test file typo

import pandas as pd
from sklearn.ensembles import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# 1. Load Training Data
train_url = "https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3.csv"
train_data = pd.read_csv(train_url)

# 2. Load Test Data
test_url = "https://raw.githubusercontent.com/dustywhite7/econ8310-assignment2/main/tests/testData.csv"
test_data = pd.read_csv(test_url)

# 3. Drop irrelevant columns
train_data = train_data.drop(columns=['DateTime'], errors='ignore')
test_data = test_data.drop(columns=['DateTime'], errors='ignore')

# 4. Separate target and features
y = train_data['meal']
X = train_data.drop(columns=['meal'])

# 5. Encode categorical variables consistently
for col in X.select_dtypes(include=['object']).columns:
    mapping = {cat: idx for idx, cat in enumerate(X[col].unique())}
    X[col] = X[col].map(mapping)
    if col in test_data.columns:
        test_data[col] = test_data[col].map(mapping).fillna(-1).astype(int)

# 6. Align test features
X_test = test_data[X.columns]

# 7. Train/validation split
x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42)

# 8. Train Decision Tree
model = DecisionTreeClassifier(max_depth=None, min_samples_leaf=5, random_state=42)
modelFit = model.fit(x_train, y_train)

# 9. Evaluate accuracy
print(f"In-sample accuracy: {accuracy_score(y_train, modelFit.predict(x_train)):.2%}")
print(f"Out-of-sample accuracy: {accuracy_score(y_val, modelFit.predict(x_val)):.2%}")

# 10. Generate predictions
pred = modelFit.predict(X_test).astype(int)
print("Number of predictions:", len(pred))
print(pred[:20])






