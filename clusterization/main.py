from sklearn import cluster
from common import textPreps
from common import caching
import math
import os
import csv

class clusterizer:
    def __init__(self, distance_threshold=0.7):
        self.clusterizer = cluster.AgglomerativeClustering(n_clusters=None, affinity='cosine',
                                       compute_full_tree=True, linkage='complete', distance_threshold=distance_threshold)

        # читаем прилагательные выражения мнения
        curDir = os.path.dirname(__file__)
        opinionAdjectivesList = "data/opinionAdjectives.csv"
        opinionAdjectivesList = os.path.join(curDir, opinionAdjectivesList)
        if not os.path.isfile(opinionAdjectivesList):
            raise Exception("Opinion adjectives file (%s) does not exists" % opinionAdjectivesList)
        with open(opinionAdjectivesList, 'r', encoding='utf-8-sig') as f:
            words, opinions = zip(*[[el[0], int(el[1])] for el in csv.reader(f, delimiter=";")])
            self.opinionAdjectives = {'words': words, 'opinions': opinions}

    def clusterize(self, messages, groupByOpinionWords = False, cacheHash = None, resetCache = True):
        # если нужно сбросить кеш - сбрасываем
        if resetCache:
            caching.saveVar("prepared messages for " + cacheHash, None)
            caching.saveVar("list of words for " + cacheHash, None)
            caching.saveVar("list IDFs of words for " + cacheHash, None)
            caching.saveVar("messages vectors for " + cacheHash, None)
            caching.saveVar("clusters of " + cacheHash, None)
        # собираем разборы слов текстов сообщений - они уже закешировались при построении вектора признаков
        analytics = self.__prepareMessages(messages, cacheHash="prepared messages for " + cacheHash)
        # получаем список слов для построения векторов сообщений в пространстве этих слов
        if not groupByOpinionWords:
            wordsList = self.__buildWordsListFromTexts(analytics, cacheHash="list of words for " + cacheHash)
        else:
            wordsList = self.__buildWordsListFromOpinionWords(analytics, cacheHash="list of words for " + cacheHash)
        # считаем IDF для каждого слова из построенного списка
        wordsIDFs = self.__calculateIDFForWords(wordsList, analytics, cacheHash="list IDFs of words for " + cacheHash)
        # строим вектор в пространстве извлечённых слов
        messagesVectors = self.__vectorizeMessages(wordsList, wordsIDFs, analytics, cacheHash="messages vectors for " +
                                                                                              cacheHash)
        # фильтруем те сообщения, у которых вектора полностью нулевые
        filteredMessages, filteredAnalytics, filteredMessagesVectors = self.__filterEmptyMessages(messages, analytics,
                                                                                                  messagesVectors)
        # и кластеризуем отфильтрованное
        resultMessages, resultClasses = self.__clusterize(filteredMessages, filteredMessagesVectors,
                                                          cacheHash="clusters of " + cacheHash)

        # извлекаем суммарные данные по группам
        groupsLeadWords, groupsLeadOpinionWords, groupsLeadOpinions = self.__extractGroupsSummaries(filteredAnalytics,
                                                                                                 resultClasses)
        # подготавливаем собранные данные к выводу
        result = self.__prepareResult(resultClasses, resultMessages, groupsLeadWords, groupsLeadOpinionWords,
                                      groupsLeadOpinions)

        return result

    @caching.cacheFuncOutput
    def __vectorizeMessages(self, wordsList, wordsIDFs, messages, cacheHash=None):
        result = []

        for message in messages:
            v = self.__vectorizeMessage(wordsList, wordsIDFs, message)
            result.append(v)

        return result

    @caching.cacheFuncOutput
    def __prepareMessages(self, messages, cacheHash=None):
        # игнорируем некоторые части речи
        ignoredPOSes = ["CONJ", "PR", "INTJ", "NUM", "PART", "ADV", "P", "S", "SPRO", "APRO", "ADVPRO"]
        userPOSes = ["A", "ADV", 'APRO', 'ADVPRO', 'ANUM']

        # собираем все слова каждого текста
        preparedMessages = []
        for message in messages:
            # получаем данные аналитики - нас интересуют только слова
            textAnalytics, wordsAnalytics, lemmas = textPreps.analyzeText(message)

            preparedMessage = []
            # перебираем все извлечённые слова
            for wordsAnalytic in wordsAnalytics:
                # исключаем игнорируемые части речи
                if wordsAnalytic['POS'] not in userPOSes:
                    continue
                # неотфильтрованное записываем
                preparedMessage.append(wordsAnalytic)

            # итог - аналитика по весомым словам текста
            preparedMessages.append(preparedMessage)

        # возвращаем итоговое представление сообщений
        return preparedMessages

    @caching.cacheFuncOutput
    def __buildWordsListFromTexts(self, messagesAnalytics, cacheHash=None):
        #построение списка всех слов во всех текстах

        # фильтруем собранные слова
        uniqueWords = set()
        for messageAnalytics in messagesAnalytics:
            for analytics in messagesAnalytics:
                for analytic in analytics:
                    uniqueWords.add(analytic['text'])
        return list(uniqueWords)

    @caching.cacheFuncOutput
    def __buildWordsListFromOpinionWords(self, messagesAnalytics, cacheHash=None):
        # получаем список всех слов с выражением мнения
        wordsList = set(self.opinionAdjectives['words'])
        # получаем все слова текста
        uniqueWords = set()
        for messageAnalytics in messagesAnalytics:
            for analytics in messagesAnalytics:
                for analytic in analytics:
                    uniqueWords.add(analytic['text'])
        # фильтруем слова выражения мнения по признаку наличия в тексте
        result = wordsList.intersection(uniqueWords)
        return list(result)

    def __vectorizeMessage(self, wordsList, wordsIDFs, messageAnalytics):
        result = []
        message = []
        for analytic in messageAnalytics:
            message.append(analytic['text'])

        for i, word in enumerate(wordsList):
            if word in message:
                tf = message.count(word) / len(message)
                idf = wordsIDFs[i]
                tdidf = tf*idf
                result.append(tdidf)
            else:
                result.append(0)
        return result

    @caching.cacheFuncOutput
    def __calculateIDFForWords(self, wordsList, textsAnalytics, cacheHash=None):
        idfs = []
        for word in wordsList:
            documentFrequency = 0
            for textAnalytics in textsAnalytics:
                for analytic in textAnalytics:
                    if analytic['text'] != word:
                        continue
                    else:
                        documentFrequency += 1
                        break
            idf = len(textsAnalytics) / (1 + documentFrequency)
            idfs.append(idf)
        return idfs

    def __filterEmptyMessages(self, messages, analytics, messagesVectors):
        resultMessages = []
        resultAnalytics = []
        resultVectors = []
        for i, vector in enumerate(messagesVectors):
            isVectorEmpty = True
            for measurement in vector:
                if measurement > 0:
                    isVectorEmpty = False
                    break

            if not isVectorEmpty:
                resultMessages.append(messages[i])
                resultAnalytics.append(analytics[i])
                resultVectors.append(messagesVectors[i])

        return resultMessages, resultAnalytics, resultVectors

    @caching.cacheFuncOutput
    def __extractGroupsSummaries(self, analytics, classes):
        # пытаемся найти главные слова
        opinionWords = self.opinionAdjectives['words']
        opinionRatings = self.opinionAdjectives['opinions']
        classesLeadWords = {}
        for classIndex in classes:
            # собираем все слова из аналитики всех сообщений класса и переводим в наборы
            classMessages = [set([word['text'] for word in analytics[i]]) for i in classes[classIndex]]
            # переводим полученное в общий набор - набор всех слов всех текстов класса
            opinionWordsInMessage = set([word for message in classMessages for word in message])
            # и ищем те слова, которые встречаются только во всех сообщениях
            for message in classMessages:
                opinionWordsInMessage = opinionWordsInMessage.intersection(message)
            classesLeadWords[classIndex] = list(opinionWordsInMessage)

        # затем ищем полученные слова в словах выражения мнения
        OWcache = {}
        classesLeadOpinion = {}
        classesLeadOpinionWords = {}
        # перебираем все классы
        for classIndex in classesLeadWords:
            summarOpinion = 0
            classOpinionWords = []
            # и все слова в классе
            for word in classesLeadWords[classIndex]:
                # пытаемся взять результат из локального кеша - вдруг это слово уже обрабатывалось в другом сообщении
                if word in OWcache:
                    classOpinionWords.append(word)
                    summarOpinion += OWcache[word]['rating']
                else:
                    # если кеша нет, ищем слово в списке слов выражения мнения
                    try:
                        OWindex = opinionWords.index(word)
                    except ValueError:
                        # такого слова в списке нет, идём дальше
                        continue
                    # если слово нашлось, берём его самого и его рейтинг
                    classOpinionWords.append(word)
                    summarOpinion += opinionRatings[OWindex]
                    # и сохраняем кеш
                    OWcache[word] = {'word': word, 'rating': opinionRatings[OWindex]}

            classesLeadOpinion[classIndex] = summarOpinion
            classesLeadOpinionWords[classIndex] = classOpinionWords

        return classesLeadWords, classesLeadOpinionWords, classesLeadOpinion

    def __clusterize(self, messages, messagesVector, minGroupSize=2, cacheHash=None):
        # кластеризуем
        self.clusterizer.fit_predict(messagesVector)

        # переводим результат из вида "номер соообщения: класс" к двум словарям вида "класс: [сообщения/тексты]"
        classes = {}
        classesMessages = {}
        for messageIndex, classIndex in enumerate(self.clusterizer.labels_):
            if classIndex in classes:
                classes[classIndex].append(messageIndex)
                classesMessages[classIndex].append(messages[messageIndex])
            else:
                classes[classIndex] = [messageIndex]
                classesMessages[classIndex] = [messages[messageIndex]]

        # фильтруем результат - рассматриваем только группы из minGroupSize и более сообщений
        resultMessages = {}
        resultClasses = {}

        for classIndex in classes:
            if len(classes[classIndex]) >= minGroupSize:
                resultMessages[classIndex] = classesMessages[classIndex]
                resultClasses[classIndex] = classes[classIndex]

        return resultMessages, resultClasses

    def __prepareResult(self, classes, classesMesages, classesLeadWords, classesLeadOW, classesOpinions):
        result = {}

        for classIndex in classes:
            classRepresentation = {
                'messages': classesMesages[classIndex],
                'leadWords': classesLeadWords[classIndex],
                'leadOpinionWords': classesLeadOW[classIndex],
                'opinionScore': classesOpinions[classIndex],
            }
            result[classIndex] = classRepresentation

        return result
