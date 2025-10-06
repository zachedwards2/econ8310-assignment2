import sklearn
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load training and test data
train_data = pd.read_csv(
    "https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3.csv"
)
test_data = pd.read_csv(
    "https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3test.csv"
)

# Drop DateTime or any irrelevant columns
train_data = train_data.drop(columns=['DateTime'], errors='ignore')
test_data = test_data.drop(columns=['DateTime'], errors='ignore')

# Separate target and features
y = train_data['meal']
X = train_data.drop(columns=['meal'])

# Encode categorical variables if any
for col in X.select_dtypes(include=['object']).columns:
    X[col] = pd.factorize(X[col])[0]
    if col in test_data.columns:
        test_data[col] = pd.factorize(test_data[col])[0]

# Align test features with training features
common_cols = [col for col in X.columns if col in test_data.columns]
X_test = test_data[common_cols]

# Optional: train/validation split
x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42)

# Train Decision Tree
model = DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=42)
modelFit = model.fit(x_train, y_train)

# Accuracy
print("In-sample accuracy:", round(100*accuracy_score(y_train, modelFit.predict(x_train)),2), "%")
print("Out-of-sample accuracy:", round(100*accuracy_score(y_val, modelFit.predict(x_val)),2), "%")

# Predictions on test set
pred = modelFit.predict(X_test).astype(int)
print(pred[:30])


