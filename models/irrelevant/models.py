"""
Info about this file
"""

# Import necessary libraries
import json
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, auc, precision_recall_curve, confusion_matrix
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report


def load_data():
    """
    Load the training/test data
    """
    logging.info("Loading training/test data")
    merged_df = pd.read_csv('/Users/dionnespaltman/Desktop/V3/merged_df.csv', sep=',')
    return merged_df


def split_data(merged_df):
    """
    Split the data into train, validation, and test sets
    """
    logging.info("Splitting validation")
    train, test = train_test_split(merged_df, test_size=0.2, random_state=123)
    train, val = train_test_split(train, stratify=train['VVR_group'], random_state=123)
    return train, val, test


def load_columns():
    """
    Load the list of columns_action_units from the JSON file
    """
    with open('/Users/dionnespaltman/Desktop/V3/columns_action_units.json', 'r') as f:
        columns_action_units = json.load(f)
    return columns_action_units


def define_featurizer(columns_action_units):
    """
    Define featurizer using StandardScaler to standardize numerical columns
    """
    numerical_column_names = columns_action_units
    featurizer = ColumnTransformer(transformers=[("numeric", StandardScaler(), numerical_column_names)], remainder='drop')
    return featurizer


def define_models(featurizer):
    """
    Define models with pipelines
    """
    dummy = make_pipeline(featurizer, DummyClassifier(strategy='most_frequent'))
    rf = make_pipeline(featurizer, RandomForestClassifier())
    svm = make_pipeline(featurizer, SVC())
    multiclass_svm = make_pipeline(featurizer, SVC(decision_function_shape='ovr'))
    xgb = make_pipeline(featurizer, XGBClassifier())
    mlp = make_pipeline(featurizer, MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000))
    
    models = {
        "Dummy": dummy,
        "RandomForest": rf,
        "SVM": svm,
        "Multiclass SVM": multiclass_svm,
        "XGBoost": xgb,
        "MLP": mlp
    }
    return models


def evaluate_model(model, X, y):
    """
    Evaluate the model's performance
    """
    model.fit(X, y)
    y_probs = model.predict_proba(X)[:, 1]
    precision = precision_score(y, model.predict(X))
    recall = recall_score(y, model.predict(X))
    f1 = f1_score(y, model.predict(X))
    precision_curve, recall_curve, _ = precision_recall_curve(y, y_probs)
    auc_pr = auc(recall_curve, precision_curve)
    cm = confusion_matrix(y, model.predict(X))
    
    logging.info(f"Precision: {precision}")
    logging.info(f"Recall: {recall}")
    logging.info(f"F1-score: {f1}")
    logging.info(f"AUC-PR score: {auc_pr}")
    logging.info(f"Confusion Matrix:")
    logging.info(cm)


def main():
    """
    Main function to train and evaluate models
    """
    # Set logging level to INFO
    logging.getLogger().setLevel(logging.INFO)
    
    merged_df = load_data()
    train, val, test = split_data(merged_df)
    columns_action_units = load_columns()
    featurizer = define_featurizer(columns_action_units)
    models = define_models(featurizer)
    
    logging.info("Fitting models")
    
    for name, model in models.items():
        model.fit(train.drop('VVR_group', axis=1), train['VVR_group'].values)
        logging.info(f"Evaluating {name} on validation data")
        pred = model.predict(val.drop('VVR_group', axis=1))
        accuracy = accuracy_score(val['VVR_group'].values, pred)
        report = classification_report(val['VVR_group'].values, pred)
        cm = confusion_matrix(val['VVR_group'].values, pred)
        
        logging.info(f"{name} Accuracy: {accuracy}")
        logging.info(f"{name} Classification Report:")
        logging.info(report)
        logging.info(f"{name} Confusion Matrix:")
        logging.info(cm)
    
    best_model_name = max(models, key=lambda x: accuracy_score(val['VVR_group'].values, models[x].predict(val.drop('VVR_group', axis=1))))
    best_model = models[best_model_name]
    
    logging.info(f"Predicting on test using {best_model_name}")
    
    pred = best_model.predict(test.drop('VVR_group', axis=1))
    
    accuracy = accuracy_score(test['VVR_group'].values, pred)
    report = classification_report(test['VVR_group'].values, pred)
    cm = confusion_matrix(test['VVR_group'].values, pred)
    
    logging.info(f"{best_model_name} Accuracy on Test Data: {accuracy}")
    logging.info(f"{best_model_name} Classification Report on Test Data:")
    logging.info(report)
    logging.info(f"{best_model_name} Confusion Matrix:")
    logging.info(cm)

    evaluate_model(best_model, test.drop('VVR_group', axis=1), test['VVR_group'].values)
    
main()