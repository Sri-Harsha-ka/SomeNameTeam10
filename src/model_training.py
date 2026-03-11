import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Load processed data from processed folder 

X_train_processed = pd.read_csv("../data/processed/X_train.csv")
X_test_processed = pd.read_csv("../data/processed/X_test.csv")
y_train_processed = pd.read_csv("../data/processed/y_train.csv")
y_test_processed = pd.read_csv("../data/processed/y_test.csv")

# Create the model and train data

lr = LinearRegression()
lr.fit(X_train_processed , y_train_processed)

with open("../artifacts/model.pkl" , "wb") as f:
    pickle.dump(lr, f)