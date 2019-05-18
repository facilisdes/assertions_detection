from .nGramsFeaturesBuilder import nGramsFeaturesBuilder
from .lexicFeatures import lexicFeatures
from .semanticFeatures import semanticFeatures
from common import textPreps


class featuresExtractor:
    def __init__(self, trainMessages, trainClasses):
        """
        инициализация объекта - получение данных анализа по входным сообщениям
        :param trainMessages: сообщения обучающей выборки
        :type trainMessages: list
        :param trainClasses: классы сообщений обучающей выборки
        :type trainClasses: list
        """
        trainMessagesAnalytics = list()
        # для каждого сообщения из обучающей выборки
        for trainMessage in trainMessages:
            # получаем данные аналитики
            textAnalytic, wordsAnalytic, lemmas = textPreps.analyzeText(trainMessage, caching=True)
            trainMessagesAnalytics.append(textAnalytic)
        # и затем строим список n-грам
        obj = nGramsFeaturesBuilder(trainMessagesAnalytics, trainClasses)
        self.ngrams = obj.getNGramsList()

        self.lf = lexicFeatures()
        self.sf = semanticFeatures()
        q = semanticFeatures()

    def getFeaturesVector(self, message):
        """
        построение вектора значений признаков для сообщения
        :param message: текст анализируемого сообщения
        :type message: string
        :return: вектор признаков данного сообщения
        :rtype: list
        """

        # получение разбора сообщения и лемматизация
        textAnalytics, wordsAnalytics, lemmas = textPreps.analyzeText(message)

        vector = list()

        # определение значений семантических признаков
        semanticFeaturesList = self.__getSemanticFeatures(message, lemmas, wordsAnalytics)
        # определение значений лексических признаков
        lexicFeaturesList = self.__getLexicFeatures(message, lemmas, wordsAnalytics)
        # определение значений признаков на n-граммах
        nGramsFeaturesList = self.__getNGramsRating(lemmas)

        # запись полученных значений в один список
        vector.extend(semanticFeaturesList)
        vector.extend(lexicFeaturesList)
        vector.extend(nGramsFeaturesList)

        return vector

    def __getNGramsRating(self, lemmas):
        """
        проверка на наличие n-грам в тексте сообщения
        :param lemmas: список лемм сообщения
        :type lemmas: list
        :return: результат проверки в виде массива
        :rtype: list
        """
        # для каждой n-граммы - если она есть в списке лемм, то признак равен 1, иначе 0
        vector = [1 if ngram in lemmas else 0 for i, ngram in enumerate(self.ngrams)]
        return vector

    def __getSemanticFeatures(self, message, lemmas, analysis):
        """
        построение вектора значений по семантическим признакам
        :param message: текст сообщения
        :type message: string
        :param lemmas: список лемм сообщения
        :type lemmas: list
        :param analysis: данные слов сообщения
        :type analysis: list
        :return: вектор семантических признаков
        :rtype: list
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
        :param message: текст сообщения
        :type message: string
        :param lemmas: список лемм сообщения
        :type lemmas: list
        :param analysis: данные слов сообщения
        :type analysis: list
        :return: вектор лексических признаков
        :rtype: list
        """
        vector = list()
        vector.extend(self.lf.getAbbreviationsRatings(lemmas))
        vector.extend(self.lf.getPunctuantionRatings(message))
        vector.extend(self.lf.getPOSRatings(analysis))
        vector.extend(self.lf.getTwitterSpecificsRatings(message))
        return vector

