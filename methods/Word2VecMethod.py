import json
from gensim.models import Word2Vec
import os

import IMethod
from common.TextPreproccessing import TextPreprocessing


class Word2VecMethod(IMethod):
    def __init__(self, sizeVec=300, minCount=6, alpha=0.05, hs=1, window=2):
        super(self)
        self.preprocessing = TextPreprocessing()
        self.sizeVec = sizeVec
        self.minCount = minCount
        self.alpha = alpha
        self.window = window
        self.hs = hs
        self.nameModel = 'models/w2v_{}_{}_{}_{}_{}_{}'.format(sizeVec, minCount, window, str(alpha)[:6], hs)
        self.w2vModel = Word2Vec()
        if os.path.exists(self.name_model + '.model'):
            self.w2vModel = Word2Vec.load(self.name_model + '.model')
        else:
            self.__learn()

    def match(self, text1: str, text2: str) -> float:
        text1Split = self.preprocessing.preprocess(text1).split(' ')
        text2Split = self.preprocessing.preprocess(text2).split(' ')
        return self.w2v_model.wv.n_similarity(text1Split, text2Split)

    def __learn(self):
        self.w2vModel = Word2Vec(
            min_count=self.min_count,
            window=self.widow,
            vector_size=self.size_vec,
            negative=10,
            alpha=self.alpha,
            min_alpha=0.0007,
            sg=1,
            hs=self.hs
        )
        with open('../dt-recom-data/docLema.json', 'r') as file:
            data = json.load(file)
        # [[],[текст]]
        self.w2vModel.build_vocab(data)
        self.w2vModel.train(data, total_examples=self.w2v_model.corpus_count, epochs=40, report_delay=1)
        self.w2vModel.init_sims(replace=True)
        self.w2vModel.save(self.name_model + '.model')
