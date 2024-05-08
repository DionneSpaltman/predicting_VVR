"""
Baseline 
This file contains the logistic regression algorithm
"""

# Import necessary libraries
import logging
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import make_pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pickle

# import my functions 
from models.evaluation import evaluation

def main():
    # Set logging level to INFO
    logging.getLogger().setLevel(logging.INFO)

    file_path = '/Users/dionnespaltman/Desktop/V3/merged_df.csv'
    
    data = pd.read_csv(file_path, sep=',')

    # Separate features (action units) and target variable (fainting occurrence)
    # Do I have to drop ID too? 
    X = data.drop(['VVR_group', 'sum_12', 'sum_4567', 'sum_456', 'Condition', 'VVR_1', 'VVR_2'], axis=1)
    y = data['VVR_group']

    print(X)
    print(y)
    
    # This can be deleted I think? 
    # Define a pipeline for a dummy classifier that predicts the most frequent class
    dummy = make_pipeline(StandardScaler(), DummyClassifier(strategy='most_frequent'))
    
    # Define a pipeline for a logistic regression classifier
    logistic = make_pipeline(StandardScaler(), LogisticRegression())
    
    # Create a 5-fold cross-validation iterator
    kf = KFold(n_splits=5, shuffle=True, random_state=123)
    
    # Log the fitting of models using cross-validation
    logging.info("Fitting models using 5-fold cross-validation")
    
    # This can also be deleted 
    # Evaluate the dummy regressor
    logging.info("Evaluation for Dummy Regressor:")
    evaluation(dummy, X, y)
    
    # Evaluate the logistic regression classifier
    logging.info("Evaluation for logistic regression classifier:")
    evaluation(logistic, X, y)

if __name__ == "__main__":
    main()



