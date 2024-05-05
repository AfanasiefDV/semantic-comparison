from common.Metrics3d import img3d


class DependencySearch:
    def __init__(self):
        self.keywordsQuantityTokens = []
        self.textQuantityTokens = []
        self.sims = []
        self.markers = []

    def increment(self, quantityTokensKeyword: int, quantityTokensText: int, sim: float, isMatch):
        self.keywordsQuantityTokens.append(quantityTokensKeyword)
        self.textQuantityTokens.append(quantityTokensText)
        self.sims.append(sim)
        self.markers.append(isMatch)


    def calculate(self):
        img3d(self.keywordsQuantityTokens, self.textQuantityTokens, self.sims, self.markers)
