import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import argparse

def preprocess_dataset(dataset_name):
    if dataset_name == "australian":
        df = pd.read_csv('data/Australian_Credit.csv', header=None)
        df.columns = ['Gender', 'Age', 'Debt', 'Married', 'BankCustomer', 'EducationLevel',
                      'Ethnicity', 'YearsEmployed', 'PriorDefault', 'Employed', 'CreditScore',
                      'DriverLicense', 'Citizen', 'ZipCode', 'Income', 'ApprovalStatus']
    else:
        df = pd.read_excel('data/German_Credit.xlsx')
    
    # Common preprocessing steps
    df.replace('?', np.nan, inplace=True)
    for col in df.select_dtypes(include=['object']).columns:
        df[col].fillna(df[col].mode()[0], inplace=True)
    
    # Dataset-specific processing
    if dataset_name == "australian":
        le = LabelEncoder()
        for col in ['ApprovalStatus', 'Gender', 'PriorDefault', 'Employed', 'DriverLicense']:
            df[col] = le.fit_transform(df[col])
        df['ApprovalStatus'] = 1 - df['ApprovalStatus']
        df = pd.get_dummies(df, columns=['Married', 'BankCustomer', 'EducationLevel', 'Ethnicity', 'Citizen'])
        target = 'ApprovalStatus'
    else:
        df['Class'] -= 1
        target = 'Class'
        df = pd.get_dummies(df, columns=['checking account', 'Credit_his', 'Purpose', 'Savings account',
                                         'Present_emp', 'sex', 'other_debtor', 'Property', 'Other_install',
                                         'Housing', 'Job'])
    
    X = df.drop(target, axis=1)
    y = df[target]
    
    X = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=9 if dataset_name == "australian" else 0
    )
    
    # Save processed data
    for data, name in zip([X_train, X_test, y_train, y_test], 
                        ['x_train', 'x_test', 'y_train', 'y_test']):
        with open(f'{name}_{dataset_name}.pickle', 'wb') as f:
            pickle.dump(data, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['australian', 'german'], required=True)
    args = parser.parse_args()
    preprocess_dataset(args.dataset)