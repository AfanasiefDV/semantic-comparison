from pymystem3 import Mystem
from nltk.corpus import stopwords


class TextPreprocessing:
    def __init__(self):
        self.mystem = Mystem()
        self.russian_stopwords = set(stopwords.words("russian"))

    def preprocess(self, doc: str) -> str:
        stop_free = " ".join([i for i in doc.lower().split() if i not in self.russian_stopwords])
        punc_free = " ".join(
            [''.join([ch for ch in i if ch not in '!"$%&\'(),.:;<=>?@[\]^_`{|}~']) for i in stop_free.split()])
        tokens = self.mystem.lemmatize(punc_free)
        tokens = [token for token in tokens if token != " "]
        text = " ".join(tokens)
        text = text.replace("\n", " ").replace("-", " ").split(' ')
        text = ' '.join([word for word in text if word != ''])
        return text
