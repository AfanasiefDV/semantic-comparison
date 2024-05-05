from methods.IMethod import IMethod
from metrics.dataForMatch import dataForMatch, dataRPDForMatch
from metrics.dataAfterMatch import dataAfterMatch, dataRPDAfterMatch
from metrics.dataAfterResearch import dataRPDAfterResearch
from metrics.util import getPredictResult
from metrics.F1Score import getF1Score
from metrics.AUC import getAUC
from methods.MethodsSentenceTransformer import MethodsEnum, MethodsSentenceTransformer
from methods.Word2VecMethod import Word2VecMethod
from methods.Word2VecWithTfIdf import Word2VecWithTfIdf
from common.TextPreproccessing import TextPreprocessing

import matplotlib.pyplot as plt
import json
import os
import pandas as pd


def match(method: IMethod, data: [dataForMatch]) -> []:
    result = []
    if os.path.exists(method.getCollector().getRelativePathForMatch()):
        with open(method.getCollector().getRelativePathForMatch(), 'r', encoding='utf-8') as file:
            sims = json.load(file)
            i = 0
            for rpd in data:
                keywords = []
                j = 0
                for keyword in rpd.keywords:
                    keywords.append(
                        dataAfterMatch(
                            keyword=keyword.keyword,
                            isMatch=keyword.isMatch,
                            similarity=float(sims[i][j])
                        ))
                    j += 1
                result.append(dataRPDAfterMatch(
                    text=rpd.text,
                    nameText=rpd.nameText,
                    keywords=keywords
                ))
                i += 1
    else:
        i = 0
        simsForDump = []
        for rpd in data:
            print((i + 1) / len(data))
            i += 1
            keywords = []
            rpdForDump = []
            j = 0
            for keyword in rpd.keywords:
                print(j)
                j += 1
                sim = method.match(keyword.keyword, rpd.text, rpd.nameText, keyword.isMatch)
                keywords.append(
                    dataAfterMatch(
                        keyword=keyword.keyword,
                        isMatch=keyword.isMatch,
                        similarity=sim
                    ))
                rpdForDump.append(str(sim))
            result.append(dataRPDAfterMatch(
                text=rpd.text,
                nameText=rpd.nameText,
                keywords=keywords
            ))
            simsForDump.append(rpdForDump)
        #method.calculateMetrics()
        with open(method.getCollector().getRelativePathForMatch(), 'w', encoding='utf-8') as file:
            json.dump(simsForDump, file, ensure_ascii=False, indent=4)
    return result


def getBemchmark() -> list:
    with open("benchmark/benchmark.json", 'r') as file:
        data = json.load(file)
    benchmark = []
    exclude = [] #["Распределенные алгоритмы", "Физические основы информационных технологий"]
    for elem in data:
        if elem["title"] not in exclude:
            keywords = []
            for i in range(len(elem["keywords"])):
                keywords.append(
                    dataForMatch(
                        keyword=elem["keywords"][i]["word"],
                        isMatch=elem["keywords"][i]["isMatch"]
                    )
                )
            benchmark.append(
                dataRPDForMatch(
                    text=elem["text"],
                    nameText=elem["title"],
                    keywords=keywords
                )
            )
    return benchmark


def getImgMatchBenchmark(method: IMethod, matchBenchmark: []):
    for rpd in matchBenchmark:
        getImgMatchRPD(method, rpd)


def getImgMatchRPD(method: IMethod, rpd):
    fig, ax = plt.subplots()
    for i in range(len(rpd.keywords)):
        plt.scatter(i, rpd.keywords[i].similarity, color='g' if rpd.keywords[i].isMatch else 'r')
        ax.annotate(rpd.keywords[i].keyword, (i, rpd.keywords[i].similarity))
    plt.title(rpd.nameText)
    plt.xlabel('number keyword')
    plt.ylabel('sim')
    plt.grid()
    plt.savefig(method.getCollector().getRelativePathForRpdImages(rpd.nameText))
    plt.close()


# TODO
def researchMethod(method: IMethod, benchmark: []):
    matchBenchmark = match(method, benchmark)
    researchResult = []
    getImgMatchBenchmark(method, matchBenchmark)
    for i in range(len(matchBenchmark)):
        researchResult.append(
            dataRPDAfterResearch(
                text=matchBenchmark[i].text,
                nameText=matchBenchmark[i].nameText,
                keywords=matchBenchmark[i].keywords,
                thresholds=[],
                f1=[],
                auc=[]
            )
        )
    resAll = {"thresholds": [], "f1": [], "auc": []}
    bestF1 = 0.0
    bestAuc = 0.0
    for threshold in range(0, 10001, 4):
        pred, allPred = getPredictResult(matchBenchmark, threshold / 10000)
        currentfF1 = getF1Score(allPred)
        currentfAuc = getAUC(allPred)
        resAll["thresholds"].append(threshold/ 10000)
        resAll["f1"].append(currentfF1)
        resAll["auc"].append(currentfAuc)
        bestF1 = max(currentfF1, bestF1)
        bestAuc = max(bestAuc, currentfAuc)
        for i in range(len(pred)):
            researchResult[i].thresholds.append(threshold/10000)
            researchResult[i].f1.append(getF1Score(pred[i]))
            researchResult[i].auc.append(getAUC(pred[i]))
    print(method.name() + ": " + str(bestF1))
    plt.plot(resAll["thresholds"], resAll["f1"])
    plt.title(method.name() + ' research abs f1')
    plt.xlabel('thresholds')
    plt.ylabel('f1')
    plt.grid()
    plt.savefig(method.getCollector().getRelativePathForResultMetricImage('f1 global'))
    plt.close()

    plt.plot(resAll["thresholds"], resAll["auc"])
    plt.title(method.name() + ' research abs auc')
    plt.xlabel('thresholds')
    plt.ylabel('auc')
    plt.grid()
    plt.savefig(method.getCollector().getRelativePathForResultMetricImage('auc global'))
    plt.close()
    return researchResult, bestF1, bestAuc


# TODO
def research():
    methods = [Word2VecMethod(), MethodsSentenceTransformer(MethodsEnum.method1),
               MethodsSentenceTransformer(MethodsEnum.method2), MethodsSentenceTransformer(MethodsEnum.method3)]
#    Word2VecWithTfIdf(),

    bench = getBemchmark()
    modelsResult = []
    print('load')
    proc = TextPreprocessing()
    for method in methods:
        print(method.name())
        researchResult, bestF1, bestAuc = researchMethod(
            method,
            bench
        )
        sumf1 = 0
        sumAuc = 0
        f1 = []
        auc = []
        bestThresholds = []
        quantityWords = []
        names = []
        for rpd in researchResult:
            f1.append(max(rpd.f1))
            auc.append(max(rpd.auc))
            sumf1 += max(rpd.f1)
            sumAuc += max(rpd.auc)
            bestThreshold = 0.0
            maxf1 = 0.0
            for j in range(len(rpd.f1)):
                if rpd.f1[j] > maxf1:
                    bestThreshold = rpd.thresholds[j]
                    maxf1 = rpd.f1[j]
            quantityWords.append(len(proc.preprocess(rpd.text).split(' ')))
            bestThresholds.append(bestThreshold)
            names.append(rpd.nameText)
            plt.plot(rpd.thresholds, rpd.f1)
            plt.title(method.name() + ' f1-score ' + rpd.nameText)
            plt.xlabel('thresholds')
            plt.ylabel('f1')
            plt.grid()
            plt.savefig(method.getCollector().getRelativePathForF1RpdImages(rpd.nameText))
            plt.close()
            plt.plot(rpd.thresholds, rpd.auc)
            plt.title(method.name() + ' auc ' + rpd.nameText)
            plt.xlabel('thresholds')
            plt.ylabel('auc')
            plt.grid()
            plt.savefig(method.getCollector().getRelativePathForAUCRpdImages(rpd.nameText))
            plt.close()
        modelsResult.append({
            "model": method.name(),
            "metrics": [
                {
                    "name": "Avg f1",
                    "value": sumf1 / len(researchResult)
                },
                {
                    "name": "Avg auc",
                    "value": sumAuc / len(researchResult)
                },
                {
                    "name": "f1 global",
                    "value": bestF1
                },
                {
                    "name": "auc global",
                    "value": bestAuc
                }
            ]
        })
        d = {"threshold": bestThresholds, "num words": quantityWords, "names": names}
        df = pd.DataFrame(d)
        df = df.sort_values(by=['num words'])
        plt.plot(df["num words"], df["threshold"])
        with pd.option_context('display.max_rows', None,
                               'display.max_columns', None,
                               'display.precision', 3,
                               ):
            print(df)
        '''
        for k in range(len(quantityWords)):
            plt.scatter(quantityWords[k], bestThresholds[k])
        '''
        plt.title(method.name() + ' research')
        plt.xlabel('num words')
        plt.ylabel('threshold')
        plt.grid()
        plt.savefig(method.name().replace("/", "-") + ' research.png')
        plt.close()

        plt.plot(range(len(f1)), f1)
        plt.title(method.name() + ' f1-score')
        plt.xlabel('num rpd')
        plt.ylabel('f1')
        plt.grid()
        plt.savefig(method.getCollector().getRelativePathForResultMetricImage('f1'))
        plt.close()

        plt.plot(range(len(auc)), auc)
        plt.title(method.name() + ' auc')
        plt.xlabel('num rpd')
        plt.ylabel('auc')
        plt.grid()
        plt.savefig(method.getCollector().getRelativePathForResultMetricImage('auc'))
        plt.close()
    with open(methods[0].getCollector().getRelativePathForResultAbsolut(), 'w', encoding='utf-8') as file:
        json.dump(modelsResult, file, ensure_ascii=False, indent=4)
