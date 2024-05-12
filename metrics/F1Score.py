from sklearn.metrics import f1_score

def getF1Score(predResult) -> float:
    TN = 0
    TP = 0
    FP = 0
    FN = 0
    for i in range(len(predResult[0])):
        if predResult[0][i]:
            if predResult[0][i] != predResult[1][i]:
                FN += 1
            else:
                TP += 1
        else:
            if predResult[0][i] != predResult[1][i]:
                FP += 1
            else:
                TN += 1
    if TP == 0:
        return 0

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    if precision*recall == 0:
        return 0
    return 2*precision*recall/(precision+recall)