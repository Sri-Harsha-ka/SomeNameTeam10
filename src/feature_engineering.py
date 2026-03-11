import data_preprocessing 
from sklearn.preprocessing import StandardScaler 
import pandas as pd

import pickle

X_train , X_test , y_train , y_test = data_preprocessing.load_split_data() 


sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)


pd.DataFrame(X_train_scaled).to_csv("../data/processed/X_train.csv",index=False)
pd.DataFrame(X_test_scaled).to_csv("../data/processed/X_test.csv",index=False)
pd.DataFrame(y_train).to_csv("../data/processed/y_train.csv",index=False)
pd.DataFrame(y_test).to_csv("../data/processed/y_test.csv",index=False)

with open("../artifacts/scaler.pkl" , "wb") as f:
    pickle.dump(sc , f)