import pandas as pd
from sklearn.model_selection import train_test_split


def load_split_data():
    # Load the Data
    data = pd.read_csv("../data/raw/dataset.csv")

    # Identifiy X and y 
    X = data[['Age',"Annual_Income_LPA" , 'Policy_Term_Years' , 'Sum_Assured_Lakhs' ]]
    y = data['Annual_Premium_Thousands']

    # split the data to train test split
    X_train , X_test , y_train , y_test = train_test_split(X,y , test_size=0.2 )
    return X_train , X_test , y_train , y_test

