import json
from gensim.models import Word2Vec
import os
import pandas as pd
import numpy as np
import time

from methods.IMethod import IMethod
from common.TextPreproccessing import TextPreprocessing
from common.CollectorData import CollectorData
from metrics.dependencySearch import DependencySearch


class Word2VecMethod(IMethod):
    def __init__(self, sizeVec=500, minCount=6, alpha=0.05, hs=1, window=2):
        self.preprocessing = TextPreprocessing()
        self.sizeVec = sizeVec
        self.minCount = minCount
        self.alpha = alpha
        self.window = window
        self.hs = hs
        self.nameModel = 'models/w2v_{}_{}_{}_{}_{}'.format(sizeVec, minCount, window, str(alpha)[:6], hs)
        self.w2vModel = Word2Vec()
        self.collector = CollectorData(self.nameModel, '01-05-2024-1')
        self.search = DependencySearch()
        if os.path.exists(self.nameModel + '.model'):
            self.w2vModel = Word2Vec.load(self.nameModel + '.model')
        else:
            self.__learn()

    #keywords, text
    def match(self, text1: str, text2: str, key = None, isMatch = None) -> float:
        text1Split = self.preprocessing.preprocess(text1).split(' ')
        text2Split = self.preprocessing.preprocess(text2).split(' ')
        #TODO if contains = 1
        #if word in w2v_model.wv.vocab:+
        maxSims = [0.0] * len(text1Split)
        for word in text2Split:
            if self.w2vModel.wv.has_index_for(word):
                for i in range(len(text1Split)):
                    if self.w2vModel.wv.has_index_for(text1Split[i]):
                        maxSims[i] = max(maxSims[i], self.w2vModel.wv.similarity(word, text1Split[i]))
        result = np.average(maxSims) #self.w2vModel.wv.n_similarity(text1Split, text2Split)

        self.search.increment(len(text1Split), len(text2Split), result, isMatch)
        return result

    def __learn(self):
        print('start learn')
        self.w2vModel = Word2Vec(
            min_count=self.minCount,
            window=self.window,
            vector_size=self.sizeVec,
            negative=10,
            alpha=self.alpha,
            min_alpha=0.0007,
            sg=1,
            hs=self.hs
        )
        print('load all data')
        data = self.__getData()
        # [[],[текст]]
        start = time.time()
        self.w2vModel.build_vocab(data)
        print('train')
        self.w2vModel.train(data, total_examples=self.w2vModel.corpus_count, epochs=40, report_delay=1)
        self.w2vModel.init_sims(replace=True)
        print('end train')
        finish = time.time()
        self.w2vModel.save(self.nameModel + '.model')
        res = finish - start
        res_msec = res * 1000
        print('Время обучения ', res_msec)

    def __getData(self):
        df_cv_short = pd.read_csv('dataVacancies/CV_short.csv', usecols=['text'], encoding='utf-8',
                                  encoding_errors='ignore')
        dataset = pd.read_csv('dataVacancies/prep_2.csv', encoding_errors='ignore')
        data = [doc[0].split() for doc in df_cv_short.to_numpy()]
        for index, row in dataset.iterrows():
            data.append(row['resume_lem'].split())
            data.append(row['vacancy_lem'].split())
        with open('dt-recom-data/docLema.json', 'r', errors='ignore', encoding='utf-8') as file:
            [data.append(word) for word in json.load(file)]
        with open('dt-recom-data/extraRpdLem.json', 'r', errors='ignore', encoding='utf-8') as file:
            [data.append(word) for word in json.load(file)]
        return data

    def name(self):
        return "w2v"

    def getCollector(self):
        return self.collector

    def calculateMetrics(self):
        self.search.calculate()
