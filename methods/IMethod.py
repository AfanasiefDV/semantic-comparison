
class IMethod:
    def __init__(self):
        pass

    def match(self, text1: str, text2: str, key = None, isMatch = None) -> float:
        raise Exception('NotImplementedException')

    def name(self) -> str:
        raise Exception('NotImplementedException')

    def getCollector(self):
        raise Exception('NotImplementedException')

    def calculateMetrics(self):
        raise Exception('NotImplementedException')