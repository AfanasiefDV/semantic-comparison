def getPredictResult(results: [], threshold) -> []:
    predResult = []
    allPredict = [[], []]
    for rpd in results:
        rdpPredResult = [[], []]
        for keyword in rpd.keywords:
            rdpPredResult[1].append(keyword.similarity >= threshold)
            rdpPredResult[0].append(keyword.isMatch)
            allPredict[0].append(keyword.similarity >= threshold)
            allPredict[1].append(keyword.isMatch)
        predResult.append(rdpPredResult)
    return predResult, allPredict
