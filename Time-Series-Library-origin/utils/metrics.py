import numpy as np
from sklearn.metrics import roc_auc_score, log_loss, precision_recall_curve, auc as sk_auc


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(true - pred))


def MSE(pred, true):
    return np.mean((true - pred) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((true - pred) / true))


def MSPE(pred, true):
    return np.mean(np.square((true - pred) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe


def AUC(y_true, y_pred):
    """Area Under the ROC Curve"""
    return roc_auc_score(y_true, y_pred)


def PRAUC(y_true, y_pred):
    """Area Under the Precision-Recall Curve"""
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    return sk_auc(recall, precision)


def LogLoss(y_true, y_pred, eps=1e-7):
    """Logarithmic Loss (Binary Cross Entropy)"""
    # sklearn's log_loss doesn't accept eps parameter
    # Manually clip predictions to avoid log(0)
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return log_loss(y_true, y_pred)


def binary_classification_metrics(y_true, y_pred):
    """
    Compute binary classification metrics.

    Args:
        y_true: Ground truth labels (numpy array)
        y_pred: Predicted probabilities (numpy array)

    Returns:
        dict: Dictionary containing AUC, PRAUC, and LogLoss
    """
    prauc = PRAUC(y_true, y_pred)
    auc_score = AUC(y_true, y_pred)
    logloss = LogLoss(y_true, y_pred)

    return {
        'PRAUC': prauc,
        'AUC': auc_score,
        'LogLoss': logloss
    }
