from .nGramsFeaturesBuilder import nGramsFeaturesBuilder
from .lexicFeatures import lexicFeatures
from .semanticFeatures import semanticFeatures
from common.debug import *
from common import textPreps


class featuresExtractor:
    def __init__(self, trainMessages, trainClasses):
        """
        инициализация - получение данных анализа по входным сообщениям
        :param trainMessages: list
        :param trainClasses: list
        """
        trainMessagesAnalytics = list()
        # для каждого сообщения из обучающей выборки
        for trainMessage in trainMessages:
            # получаем данные аналитики
            analytic, lemmas = textPreps.analyzeMessage(trainMessage)
            trainMessagesAnalytics.append(analytic)
        # и затем строим список n-грам
        obj = nGramsFeaturesBuilder(trainMessagesAnalytics, trainClasses)
        self.ngrams = obj.getNGramsList()

        self.lf = lexicFeatures()
        self.sf = semanticFeatures()

    def getFeaturesVector(self, message):
        """
        построение вектора значений признаков для сообщения
        :param message: string
        :return: list
        """
        # лемматизация сообщения
        # lemmas = m.lemmatize(message)
        # получение разбора сообщения
        # details = m.analyze(message)

        details, lemmas = textPreps.analyzeMessage(message)

        vector = list()

        # определение значений семантических признаков
        semanticFeatures = self.__getSemanticFeatures(message, lemmas, details)
        # определение значений лексических признаков
        lexicFeatures = self.__getLexicFeatures(message, lemmas, details)
        # определение значений признаков на n-граммах
        nGramsFeatures = self.__getNGramsRating(lemmas)

        # запись полученных значений в один список
        vector.extend(semanticFeatures)
        vector.extend(lexicFeatures)
        vector.extend(nGramsFeatures)

        return vector

    def __getNGramsRating(self, lemmas):
        """
        построение вектора значений по n-граммам
        :param lemmas: list
        :return: list
        """
        # для каждой n-граммы - если она есть в списке лемм, то признак равен 1, иначе 0
        vector = [ 1 if ngram in lemmas else 0 for i, ngram in enumerate(self.ngrams)]
        return vector

    def __getSemanticFeatures(self, message, lemmas, analysis):
        """
        построение вектора значений по семантическим признакам
        :param message: string
        :param lemmas: list
        :param analysis: list
        :return: list
        """
        vector = list()
        vector.append(self.sf.getOpinionWordsRating(lemmas))
        vector.append(self.sf.getSmilesRating(message, lemmas))
        vector.append(self.sf.getVulgarWordsRating(analysis, lemmas))
        vector.extend(self.sf.getSpeechActVerbsRatings(analysis, lemmas))
        return vector

    def __getLexicFeatures(self, message, lemmas, analysis):
        """
        построение вектора значений по лексическим признакам
        :param message: string
        :param lemmas: list
        :param analysis: list
        :return: list
        """
        vector = list()
        vector.extend(self.lf.getAbbreviationsRatings(lemmas))
        vector.extend(self.lf.getPunctuantionRatings(message))
        vector.extend(self.lf.getPOSRatings(analysis))
        vector.extend(self.lf.getTwitterSpecificsRatings(message))
        return vector

