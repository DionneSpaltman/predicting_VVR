"""

"""

import logging
from sklearn.metrics import precision_score, recall_score, f1_score, auc, precision_recall_curve

def evaluation(model, X, y):
    # Fit the model
    model.fit(X, y)
    
    # Predict probabilities
    y_probs = model.predict_proba(X)[:, 1]
    
    # Compute precision, recall, and F1-score
    precision = precision_score(y, model.predict(X))
    recall = recall_score(y, model.predict(X))
    f1 = f1_score(y, model.predict(X))
    
    # Compute precision-recall curve and AUC-PR score
    precision_curve, recall_curve, _ = precision_recall_curve(y, y_probs)
    auc_pr = auc(recall_curve, precision_curve)
    
    # Log the evaluation metrics
    logging.info(f"Precision: {precision}")
    logging.info(f"Recall: {recall}")
    logging.info(f"F1-score: {f1}")
    logging.info(f"AUC-PR score: {auc_pr}")
