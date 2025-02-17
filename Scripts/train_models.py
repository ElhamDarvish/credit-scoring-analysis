import pickle
import argparse
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
                             ExtraTreesClassifier)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier

MODEL_CONFIG = {
    'Logistic_Regression': {
        'model': LogisticRegression(),
        'params': {
            'solver': ['liblinear', 'saga'],
            'C': [0.1, 1, 10],
            'max_iter': [1000]
        }
    },
    # Include similar configurations for all 9 models
    # ... (other model configurations)
}

def train_model(dataset_name, model_name):
    # Load data
    with open(f'x_train_{dataset_name}.pickle', 'rb') as f:
        X_train = pickle.load(f)
    with open(f'y_train_{dataset_name}.pickle', 'rb') as f:
        y_train = pickle.load(f)
    
    # Get model config
    config = MODEL_CONFIG[model_name]
    pipeline = make_pipeline(config['model'])
    
    # Train with GridSearchCV
    grid = GridSearchCV(
        pipeline,
        {f'{config["model"].__class__.__name__.lower()}__{k}': v 
         for k, v in config['params'].items()},
        cv=KFold(n_splits=5, shuffle=True, random_state=0),
        n_jobs=-1
    )
    grid.fit(X_train, y_train)
    
    # Save model
    with open(f'models/{dataset_name}_{model_name}.pkl', 'wb') as f:
        pickle.dump(grid.best_estimator_, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['australian', 'german'], required=True)
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    train_model(args.dataset, args.model)