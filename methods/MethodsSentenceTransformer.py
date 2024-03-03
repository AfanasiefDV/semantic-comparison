import IMethod
from common.Translator import Translator

from enum import Enum
from sentence_transformers import SentenceTransformer, util


class MethodsEnum(str, Enum):
    method1 = 'all-MiniLM-L6-v2'
    method2 = 'msmarco-distilbert-base-v3'

class MethodsSentenceTransformer(IMethod):
    def __init__(self, method: str):
        super(self)
        self.model = SentenceTransformer(method)
        self.translator = Translator()


    def match(self, text1: str, text2: str) -> float:
        text1Encode = self.model.encode(self.translator.translate(text1))
        text2Encode = self.model.encode(self.translator.translate(text2))
        return util.cos_sim(text1Encode, text2Encode)