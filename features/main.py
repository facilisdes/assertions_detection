from .nGramsFeaturesBuilder import nGramsFeaturesBuilder
from .lexicFeatures import lexicFeatures
from .semanticFeatures import semanticFeatures
from common import textPreps
from common import caching
import hashlib


class featuresExtractor:
    def __init__(self):
        """
        инициализация объекта
        """
        self.lf = lexicFeatures()
        self.sf = semanticFeatures()

    def isTrainingRequired(self, testsHash):
        """
        возвращает результат проверки наличия кеша результата расчёта n-грам с указанным идентификатором
        :param testsHash: идентификатор
        :type testsHash: string
        :return: результат проверки
        :rtype: bool
        """
        ngrams = caching.readVar(testsHash)
        if ngrams is not None:
            self.ngrams = ngrams
            return False
        return True

    def trainNGrams(self, trainMessages, trainClasses, cache=True, testsHash=None):
        """
        построение списка n-грам по размеченным сообщениям
        :param trainMessages: сообщения обучающей выборки
        :type trainMessages: list
        :param trainClasses: классы сообщений обучающей выборки
        :type trainClasses: list
        :param cache: флаг необходимости кеширования результата расчётов
        :type cache: bool
        :param testsHash: идентификатор искомого результата, может быть пустым
        :type testsHash: string
        :return: идентификатор сохранённого результата
        :rtype: string
        """
        ngrams = None
        if cache:
            if testsHash is None:
                testsHash = hashlib.md5(''.join(trainMessages).encode("utf-8")).hexdigest()
            ngrams = caching.readVar(testsHash)

        if ngrams is None:
            trainMessagesAnalytics = list()
            # для каждого сообщения из обучающей выборки
            for trainMessage in trainMessages:
                # получаем данные аналитики
                textAnalytic, wordsAnalytic, lemmas = textPreps.analyzeText(trainMessage)
                trainMessagesAnalytics.append(textAnalytic)
            # и затем строим список n-грам
            obj = nGramsFeaturesBuilder(trainMessagesAnalytics, trainClasses)
            ngrams = obj.getNGramsList()

        if cache:
            if testsHash is None:
                testsHash = hashlib.md5(''.join(trainMessages).encode("utf-8")).hexdigest()
            caching.saveVar(testsHash, ngrams)

        self.ngrams = ngrams

        return testsHash

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

