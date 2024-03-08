"""
This code is all from the group assignment for the course Machine Learning
It might be useful to use to set up my project 

I will only use it as a reference, and the file will be deleted once I have set up my project 
"""

# To evaluate 
import json
from sklearn.metrics import mean_absolute_error
import numpy as np
import json

# From the baseline file to set up the assignment 
import pandas as pd
import logging
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

def evaluate(gold_path, pred_path):
    gold = np.array([ x['year'] for x in json.load(open(gold_path)) ]).astype(float)
    pred = np.array([ x['year'] for x in json.load(open(pred_path)) ]).astype(float)

    return mean_absolute_error(gold, pred)


def main():
    """
    This code is from the baseline.py file for the Machine Learning group assignment
    I made ChatGPT add notes to the code 
    """
    # Set logging level to INFO
    logging.getLogger().setLevel(logging.INFO)
    
    # Log the loading of training/test data
    logging.info("Loading training/test data")
    
    # Load training and test data from JSON files into Pandas DataFrames and fill missing values with empty strings
    train = pd.DataFrame.from_records(json.load(open('train.json'))).fillna("")
    test = pd.DataFrame.from_records(json.load(open('test.json'))).fillna("")
    
    # Log the splitting of validation data
    logging.info("Splitting validation")
    
    # Split the training data into training and validation sets, using stratified sampling based on the 'year' column
    train, val = train_test_split(train, stratify=train['year'], random_state=123)
    
    # Define a featurizer using CountVectorizer to transform text data in the 'title' column
    featurizer = ColumnTransformer(
        transformers=[("title", CountVectorizer(), "title")],
        remainder='drop')
    
    # Define a pipeline for a dummy regressor that predicts the mean value of the target variable
    dummy = make_pipeline(featurizer, DummyRegressor(strategy='mean'))
    
    # Define a pipeline for a Ridge regression model
    ridge = make_pipeline(featurizer, Ridge())
    
    # Log the fitting of models
    logging.info("Fitting models")
    
    # Fit the dummy regressor to the training data
    dummy.fit(train.drop('year', axis=1), train['year'].values)
    
    # Fit the Ridge regression model to the training data
    ridge.fit(train.drop('year', axis=1), train['year'].values)
    
    # Log the evaluation on validation data
    logging.info("Evaluating on validation data")
    
    # Calculate the mean absolute error of the dummy regressor on the validation data
    err = mean_absolute_error(val['year'].values, dummy.predict(val.drop('year', axis=1)))
    logging.info(f"Mean baseline MAE: {err}")
    
    # Calculate the mean absolute error of the Ridge regression model on the validation data
    err = mean_absolute_error(val['year'].values, ridge.predict(val.drop('year', axis=1)))
    logging.info(f"Ridge regress MAE: {err}")
    
    # Log the prediction on the test data
    logging.info(f"Predicting on test")
    
    # Make predictions using the Ridge regression model on the test data
    pred = ridge.predict(test)
    
    # Add the predicted 'year' values to the test DataFrame
    test['year'] = pred
    
    # Log the writing of the prediction file
    logging.info("Writing prediction file")
    
    # Write the test DataFrame with predicted values to a JSON file
    test.to_json("predicted.json", orient='records', indent=2)

    
main()

