import datetime as dt
import os

root = 'collect'
resultMatch = 'resultMatch'
resultMatchRpd = 'rpdMatchImages'
resultImgMetricF1 = 'rpdF1Images'
resultImgMetricAUC = 'rpdAUCImages'
resultMetric = 'metrics'
resultAbsolut = 'absolutResult'

patternDate = "%d-%m-%Y"

formatImage = '.png'
formatData = '.json'


class CollectorData:
    def __init__(self, modelName: str, dateCollect=None):
        self.modelName = modelName.replace('/', '-').replace('.', '-')
        if not dateCollect:
            self.dateCollect = dt.datetime.now().strftime(patternDate)
        else:
            self.dateCollect = dateCollect
        self.basePath = os.path.join(root, self.dateCollect)

    def __base(self, currentDir, format, indexName = None):
        path = os.path.join(self.basePath, currentDir)
        os.makedirs(path, exist_ok=True)
        if not indexName:
            return os.path.join(path, self.modelName + format)
        else:
            path = os.path.join(path, self.modelName)
            os.makedirs(path, exist_ok=True)
            return os.path.join(path, indexName + format)

    def getRelativePathForMatch(self):
        return self.__base(resultMatch, formatData)

    def getRelativePathForRpdImages(self, indexName = None):
        return self.__base(resultMatchRpd, formatImage, indexName)

    def getRelativePathForF1RpdImages(self, indexName = None):
        return self.__base(resultImgMetricF1, formatImage, indexName)

    def getRelativePathForAUCRpdImages(self, indexName = None):
        return self.__base(resultImgMetricAUC, formatImage, indexName)

    def getRelativePathForMetrics(self):
        return self.__base(resultMetric, formatData)

    def getRelativePathForResultAbsolut(self):
        os.makedirs(self.basePath, exist_ok=True)
        return os.path.join(self.basePath, resultAbsolut + formatData)

    def getRelativePathForResultMetricImage(self, metricName):
        return self.__base(resultAbsolut, formatImage, metricName)

    # TODO open and save JSON