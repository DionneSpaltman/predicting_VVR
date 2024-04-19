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

    file_path = '/Users/dionnespaltman/Desktop/downloading/big_dataframe.pkl'
    
    # Open the pickle file in read-binary mode
    with open(file_path, "rb") as f:
        # Load the object from the pickle file
        data = pickle.load(f)
    
    # Load the file containing information about fainting occurrences and gender
    labels = pd.read_csv(data_dir + 'labels.csv')

    print(labels)
    
    # # Merge the datasets based on a common identifier (e.g., participant ID)
    # merged_data = pd.merge(data, labels, on='participant_id')
    
    # # Separate features (action units) and target variable (fainting occurrence)
    # X = merged_data.drop(['fainted'], axis=1)
    # y = merged_data['fainted']
    
    # # This can be deleted I think? 
    # # Define a pipeline for a dummy classifier that predicts the most frequent class
    # dummy = make_pipeline(StandardScaler(), DummyClassifier(strategy='most_frequent'))
    
    # # Define a pipeline for a logistic regression classifier
    # logistic = make_pipeline(StandardScaler(), LogisticRegression())
    
    # # Create a 5-fold cross-validation iterator
    # kf = KFold(n_splits=5, shuffle=True, random_state=123)
    
    # # Log the fitting of models using cross-validation
    # logging.info("Fitting models using 5-fold cross-validation")
    
    # # This can also be deleted 
    # # Evaluate the dummy regressor
    # logging.info("Evaluation for Dummy Regressor:")
    # evaluation(dummy, X, y)
    
    # # Evaluate the logistic regression classifier
    # logging.info("Evaluation for logistic regression classifier:")
    # evaluation(logistic, X, y)

if __name__ == "__main__":
    main()



