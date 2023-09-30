#!/usr/bin/env python
# coding: utf-8


from sklearn.metrics import roc_auc_score, matthews_corrcoef, average_precision_score, confusion_matrix
from sklearn.metrics import roc_curve
import math
import numpy as np


def evaluate_classifier(true_labels, predictions, probs):
    auc = roc_auc_score(true_labels, probs)
    mcc = matthews_corrcoef(true_labels, predictions)
    avg_precision = average_precision_score(true_labels, probs)
    tn, fp, fn, tp = confusion_matrix(true_labels, predictions).ravel()
    spe = tn / (tn + fp)
    sen = tp / (tp + fn)
    ba = (spe + sen)/2
    return {'Held_out_TP': tp, 'Held_out_TN': tn,
            'Held_out_FP': fp, 'Held_out_FN': fn,
            'Held_out_BA': ba,
            'Held_out_AUC': auc, 'Held_out_MCC': mcc, 
            'Held_out_AUCPR': avg_precision, 'Held_out_Specificity': spe,
            'Held_out_Sensitivity': sen}


def optimize_threshold_j_statistic(y_true, y_probs):
    # Example usage:
    # y_true is the true labels (binary)
    # y_probs is the predicted probabilities
    # best_threshold = optimize_threshold_j_statistic(y_true, y_probs)

    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    
    # Calculate J statistic values
    j_statistic = tpr - fpr
    
    # Find the index of the threshold that maximizes J statistic
    best_threshold_idx = j_statistic.argmax()
    
    # Get the best threshold
    best_threshold = thresholds[best_threshold_idx]
    
    return best_threshold