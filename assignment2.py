import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ===============================
# 1. Load training and test data
# ===============================
train_url = "https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3.csv"
test_url  = "https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3test.csv"

train_data = pd.read_csv(train_url)
test_data  = pd.read_csv(test_url)

# Drop irrelevant columns
train_data = train_data.drop(columns=['DateTime'], errors='ignore')
test_data  = test_data.drop(columns=['DateTime'], errors='ignore')

# ===============================
# 2. Separate target and features
# ===============================
y = train_data['meal']
X = train_data.drop(columns=['meal'])

# Encode categorical variables as integers
for col in X.select_dtypes(include=['object']).columns:
    X[col], uniques = pd.factorize(X[col])
    if col in test_data.columns:
        test_data[col] = pd.factorize(test_data[col])[0]

# Align test features with training features
common_cols = [col for col in X.columns if col in test_data.columns]
X_test = test_data[common_cols]

# ===============================
# 3. Optional train/validation split
# ===============================
x_train, x_val, y_train, y_val = train_test_split(
    X, y, test_size=0.33, random_state=42
)

# ===============================
# 4. Define the forecasting model
# ===============================
model = DecisionTreeClassifier(
    max_depth=None,          # fully grow the tree to capture patterns
    min_samples_leaf=5,      # avoid very small leaves
    random_state=42
)

# ===============================
# 5. Fit the model
# ===============================
modelFit = model.fit(x_train, y_train)

# Optional: print accuracies
print("In-sample accuracy:", round(100*accuracy_score(y_train, modelFit.predict(x_train)),2), "%")
print("Out-of-sample accuracy:", round(100*accuracy_score(y_val, modelFit.predict(x_val)),2), "%")

# ===============================
# 6. Generate predictions for test data
# ===============================
pred = modelFit.predict(X_test).astype(int)  # ensure integer 0/1

# Preview first 30 predictions
print(pred[:30])




