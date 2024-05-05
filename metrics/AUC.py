from sklearn.metrics import roc_auc_score

def getAUC(predResult) -> float:
    try:
        return roc_auc_score(predResult[0], predResult[1])
    except ValueError:
        return 0.0
