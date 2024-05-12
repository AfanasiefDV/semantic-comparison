import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import json
from gensim.models import Word2Vec
import os
import pandas as pd
import numpy as np
from methods.IMethod import IMethod
from common.TextPreproccessing import TextPreprocessing
from common.CollectorData import CollectorData


class Word2VecWithTfIdf(IMethod):
    def __init__(self, sizeVec=300, minCount=6, alpha=0.05, hs=1, window=2, ngramRange=(1, 1)):
        self.preprocessing = TextPreprocessing()
        self.sizeVec = sizeVec
        self.minCount = minCount
        self.alpha = alpha
        self.window = window
        self.hs = hs
        self.ngramRange = ngramRange
        self.nameModelW2V = 'models/w2v_{}_{}_{}_{}_{}'.format(sizeVec, minCount, window, str(alpha)[:6], hs)
        self.nameModelTfIdf = 'models/tfidf_{}_{}'.format(ngramRange[0], ngramRange[1])
        self.nameModel = 'w2v_with_tfidf'
        self.w2vModel = Word2Vec()
        self.collector = CollectorData(self.nameModel)
        self.tfidf = None

        if os.path.exists(self.nameModelTfIdf + '.pickle'):
            self.w2vModel = Word2Vec.load(self.nameModelW2V + '.model')
            self.tfidf = pickle.load(open(self.nameModelTfIdf + '.pickle', "rb"))
        else:
            self.__learn()

    def match(self, text1: str, text2: str, key=None, isMatch = None) -> float:
        text1Split = self.preprocessing.preprocess(text1).split(' ')
        text2Split = self.preprocessing.preprocess(text2).split(' ')
        maxSims = [0.0] * len(text1Split)
        for word in text2Split:
            if self.w2vModel.wv.has_index_for(word):
                for i in range(len(text1Split)):
                    if self.w2vModel.wv.has_index_for(text1Split[i]):
                        maxSims[i] = max(maxSims[i], self.w2vModel.wv.similarity(word, text1Split[i]))
        w = self.tfidf.transform([" ".join(text1Split)]).toarray()
        s = self.tfidf.transform([" ".join(text2Split)]).toarray()
        return np.average(maxSims)

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
        if os.path.exists(self.nameModelW2V + '.model'):
            self.w2vModel = Word2Vec.load(self.nameModelW2V + '.model')
        else:
            self.w2vModel.build_vocab(data)
            self.w2vModel.train(data, total_examples=self.w2vModel.corpus_count, epochs=40, report_delay=1)
            self.w2vModel.init_sims(replace=True)
            self.w2vModel.save(self.nameModelW2V + '.model')
        print('learn tfidf')
        self.tfidf = TfidfVectorizer(ngram_range=self.ngramRange)
        self.tfidf.fit_transform(self.__getDataN())
        print('end learn tfidf')
        pickle.dump(self.tfidf, open(self.nameModelTfIdf + '.pickle', "wb"))

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
        return data

    def __getDataN(self):
        df_cv_short = pd.read_csv('dataVacancies/CV_short.csv', usecols=['text'], encoding='utf-8',
                                  encoding_errors='ignore')
        dataset = pd.read_csv('dataVacancies/prep_2.csv', encoding_errors='ignore')
        data = [doc[0] for doc in df_cv_short.to_numpy()]
        for index, row in dataset.iterrows():
            data.append(row['resume_lem'])
            data.append(row['vacancy_lem'])
        with open('dt-recom-data/docLema.json', 'r', errors='ignore', encoding='utf-8') as file:
            [data.append(" ".join(word)) for word in json.load(file)]
        return data

    def name(self):
        return self.nameModel

    def getCollector(self):
        return self.collector
