from methods.IMethod import IMethod
from common.Translator import Translator
from common.CollectorData import CollectorData

from enum import Enum
from sentence_transformers import SentenceTransformer, util


class MethodsEnum(str, Enum):
    method1 = 'all-MiniLM-L6-v2'
    method2 = 'msmarco-distilbert-base-v3'
    method3 = 'symanto/sn-xlm-roberta-base-snli-mnli-anli-xnli'


class MethodsSentenceTransformer(IMethod):
    def __init__(self, method: str):
        self.collector = CollectorData(method, '01-05-2024-1')
        self.method = method
        self.model = SentenceTransformer(method)
        self.translator = Translator()

    def match(self, text1: str, text2: str, key = None, isMatch = None) -> float:
        if self.method == 'symanto/sn-xlm-roberta-base-snli-mnli-anli-xnli':
            return float(util.cos_sim(self.model.encode(text1), self.model.encode(text2)))
        text1Encode = self.model.encode(self.translator.translate(text1, 'keyword_'+key))
        text2Encode = self.model.encode(self.translator.translate(text2, 'text_'+key))
        return float(util.cos_sim(text1Encode, text2Encode))

    def name(self):
        return self.method

    def getCollector(self):
        return self.collector

    def calculateMetrics(self):
        print("calc")