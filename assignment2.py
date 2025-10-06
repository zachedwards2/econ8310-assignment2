import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Load training data
train_url = "https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3.csv"
train_data = pd.read_csv(train_url)

# 2. Load the test set (1000 rows required by the unit test)
test_url = "https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3test.csv"
test_data = pd.read_csv(test_url)

# 3. Drop irrelevant columns
train_data = train_data.drop(columns=['DateTime'], errors='ignore')
test_data = test_data.drop(columns=['DateTime'], errors='ignore')

# 4. Separate target and features
y = train_data['meal']
X = train_data.drop(columns=['meal'])

# 5. Encode categorical features
for col in X.select_dtypes(include=['object']).columns:
    X[col] = pd.factorize(X[col])[0]
    if col in test_data.columns:
        test_data[col] = pd.factorize(test_data[col])[0]

# 6. Align test features
X_test = test_data[X.columns]

# 7. Train/validation split
x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42)

# 8. Train Decision Tree
model = DecisionTreeClassifier(max_depth=None, min_samples_leaf=5, random_state=42)
modelFit = model.fit(x_train, y_train)

# 9. Accuracy
print("In-sample accuracy:", round(100*accuracy_score(y_train, modelFit.predict(x_train)),2), "%")
print("Out-of-sample accuracy:", round(100*accuracy_score(y_val, modelFit.predict(x_val)),2), "%")

# 10. Predictions (1000 rows)
pred = modelFit.predict(X_test).astype(int)



