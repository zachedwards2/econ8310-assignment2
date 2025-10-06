import pandas as pd 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

#Load Training Data
train_url = "https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3.csv"
train_data = pd.read_csv(train_url)

#Load Test Data
test_url = "https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3test.csv"
test_data = pd.read_csv(test_url)

#Drop irrelevant columns
train_data = train_data.drop(columns=['DateTime'], errors='ignore')
test_data = test_data.drop(columns=['DateTime'], errors='ignore')

#Separate target and features
y = train_data['meal']
X = train_data.drop(columns=['meal'])

#Encode categorical variables consistently
for col in X.select_dtypes(include=['object']).columns:
    mapping = {cat: idx for idx, cat in enumerate(X[col].unique())}
    X[col] = X[col].map(mapping)
    if col in test_data.columns:
        test_data[col] = test_data[col].map(mapping).fillna(-1).astype(int)

#Align test features
X_test = test_data[X.columns]

#Train/validation split
x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42)

#Train Decision Tree
model = DecisionTreeClassifier(max_depth=None, min_samples_leaf=5, random_state=42)
modelFit = model.fit(x_train, y_train)

#Predictions
pred = modelFit.predict(X_test).astype(int).tolist()










