import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --------------------------
# 1. Load training and test data
# --------------------------
train_url = "https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3.csv"
test_url  = "https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3test.csv"

train_data = pd.read_csv(train_url)
test_data = pd.read_csv(test_url)

# Drop DateTime if it exists
train_data = train_data.drop(columns=['DateTime'], errors='ignore')
test_data = test_data.drop(columns=['DateTime'], errors='ignore')

# --------------------------
# 2. Separate features and target
# --------------------------
y = train_data['meal']
X = train_data.drop(columns=['meal'])

# --------------------------
# 3. Encode categorical variables
# --------------------------
for col in X.select_dtypes(include=['object']).columns:
    X[col] = pd.factorize(X[col])[0]
    if col in test_data.columns:
        test_data[col] = pd.factorize(test_data[col])[0]

# --------------------------
# 4. Align test features with training features
# --------------------------
X_test = test_data[X.columns]  # ensures same columns and order

# --------------------------
# 5. Optional train/validation split
# --------------------------
x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42)

# --------------------------
# 6. Train Random Forest
# --------------------------
model = RandomForestClassifier(
    n_estimators=500,       # number of trees
    max_depth=None,         # no depth limit
    min_samples_leaf=5,     # minimum samples per leaf
    random_state=42
)

modelFit = model.fit(x_train, y_train)

# --------------------------
# 7. Accuracy checks
# --------------------------
print("In-sample accuracy:", round(100*accuracy_score(y_train, modelFit.predict(x_train)),2), "%")
print("Out-of-sample accuracy:", round(100*accuracy_score(y_val, modelFit.predict(x_val)),2), "%")

# --------------------------
# 8. Make predictions on test set (1000 observations)
# --------------------------
pred = np.array(modelFit.predict(X_test), dtype=int)

# Quick checks
print("Number of predictions:", len(pred))  # Should be 1000
print(pred[:20])


