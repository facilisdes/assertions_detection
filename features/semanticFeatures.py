import os
import csv
from emoji import UNICODE_EMOJI


class semanticFeatures:
    def __init__(self, speechActVerbsList=False, opinionWordsList=False, smilesList=False, vulgarWordsList=False):
        """
        инициализация - чтение данных из файлов
        :param speechActVerbsList: (опционально) путь к .csv-файлу со словарём глаголов,
            соответствующих опред. речевым актам
        :type speechActVerbsList: string
        :param opinionWordsList: (опционально) путь к .csv-файлу со словарём слов с яркой эмоциональной окраской
        :type opinionWordsList: string
        :param smilesList: (опционально) путь к .csv-файлу со списком смайликов
        :type smilesList: string
        :param vulgarWordsList: (опционально) путь к .csv-файлу со словарём вульгарных слов
        :type vulgarWordsList: string
        """
        curDir = os.path.dirname(__file__)
        # берём текущий путь и считаем пути относительно его
        if speechActVerbsList is False:
            speechActVerbsList = 'data/speechActVerbs.csv'
            speechActVerbsList = os.path.join(curDir, speechActVerbsList)
        if opinionWordsList is False:
            opinionWordsList = 'data/opinionWords.csv'
            opinionWordsList = os.path.join(curDir, opinionWordsList)
        if smilesList is False:
            smilesList = 'data/smiles.csv'
            smilesList = os.path.join(curDir, smilesList)
        if vulgarWordsList is False:
            vulgarWordsList = 'data/vulgarWords.csv'
            vulgarWordsList = os.path.join(curDir, vulgarWordsList)

        if not os.path.isfile(speechActVerbsList):
            raise Exception("Speech act verbs file (%s) does not exists" % speechActVerbsList)
        if not os.path.isfile(opinionWordsList):
            raise Exception("Opinions words file (%s) does not exists" % opinionWordsList)
        if not os.path.isfile(smilesList):
            raise Exception("Smiles file (%s) does not exists" % smilesList)
        if not os.path.isfile(vulgarWordsList):
            raise Exception("Smiles file (%s) does not exists" % vulgarWordsList)

        # читаем нулевой столбец каждого файла в массив
        with open(speechActVerbsList, 'r', encoding='utf-8-sig') as f:
            self.speechActVerbsList = [el[0] for el in csv.reader(f, delimiter=";")]
        with open(opinionWordsList, 'r', encoding='utf-8-sig') as f:
            self.opinionWordsList = [[el[0], el[1]] for el in csv.reader(f, delimiter=";")]
        with open(smilesList, 'r', encoding='utf-8-sig') as f:
            self.smilesList = [el[0] for el in csv.reader(f, delimiter=";")]
        with open(vulgarWordsList, 'r', encoding='utf-8-sig') as f:
            self.vulgarWordsList = [el[0] for el in csv.reader(f, delimiter=";")]

    def getOpinionWordsRating(self, lemmas):
        """
        проверка на наличие и подсчёт в тексте слов с яркой эмоциональной окраской
        :param lemmas: список лемм сообщения
        :type lemmas: list
        :return: результат проверки в виде числа (0 или 1)
        :rtype: int
        """
        opinionWordsCount = 0
        opinionWords, opinionRatings = zip(*self.opinionWordsList)
        opinionRatingsResult = {'-2': 0, '2': 0}

        for i in range(len(opinionWords)):
            word = opinionWords[i]
            rating = opinionRatings[i]
            if word in lemmas:
                opinionWordsCount += 1
                opinionRatingsResult[rating] += 1

        # нормируем результаты
        if opinionWordsCount > 1:
            if opinionWordsCount < 4:
                opinionWordsCount = 2
            else:
                opinionWordsCount = 3
        for rating in opinionRatingsResult:
            if opinionRatingsResult[rating] > 3:
                opinionRatingsResult[rating] = 3

        # после нормирования переводим в булевый вектор
        opinionWordsResult = [False] * 3
        for mark in range(opinionWordsCount):
            opinionWordsResult[mark] = True
        opinionRatingsResultBool = []
        for rating in opinionRatingsResult:
            opinionRatingsResultBoolTmp = [False] * 3
            for mark in range(opinionRatingsResult[rating]):
                opinionRatingsResultBoolTmp[mark] = True
            opinionRatingsResultBool.extend(opinionRatingsResultBoolTmp)

        return [*opinionWordsResult, *opinionRatingsResultBool]

    def getVulgarWordsRating(self, analysis, lemmas):
        """
        проверка на наличие в тексте обсценной лексики
        :param analysis: данные по анализу слов текста
        :type analysis: list
        :param lemmas: список лемм слов текста
        :type lemmas: list
        :return: результат проверки в виде числа (0 или 1)
        :rtype: int
        """
        result = False
        # проверяем каждое обсценное слово из словаря на наличие в тексте
        for vulgarWord in self.vulgarWordsList:
            if vulgarWord in lemmas:
                result = True
                break
        if result == False:
            for word in analysis:
                if word['isObscene']:
                    result = True
                    break

        return result

    def getSmilesRating(self, message, lemmas):
        """
        проверка на наличие в тексте смайликов и эмодзи
        :param message: текст сообщения
        :type message: string
        :param lemmas: список лемм слов текста
        :type lemmas: list
        :return: результат проверки в виде числа (0 или 1)
        :rtype: int
        """
        result = [False]
        # Проверяем все смайлики из словаря на наличие в тексте сообщения. Самая лёгкая проверка в надежде исключить
        # сообщение до начала проверки по большому списку эмодзи
        for smile in self.smilesList:
            if smile in message:
                result[0] = True
                break
        # если обычных смайликов всё же не нашлось, ищем эмодзи
        if result[0] == True:
            for lemma in lemmas:
                if lemma in UNICODE_EMOJI:
                    result[0] = True
                    break

        # затем ищем скобки "((" или "))"
        brackets = {
            "(": {},
            ")": {}
        }
        for bracket in brackets:
            # считаем общее количество
            brackets[bracket]['count'] = message.count(bracket)
            # и количество "сильных", т.е. повторяющихся
            brackets[bracket]['strongCount'] = message.count(bracket*2)

            # нормируем кол-во сильных
            if brackets[bracket]['strongCount'] > 0:
                if brackets[bracket]['strongCount'] < 3:
                    # на один или два знака ставим признак, равный 1
                    brackets[bracket]['strongCount'] = 1
                else:
                    # на три и более ставим 2
                    brackets[bracket]['strongCount'] = 2

        # уменьшаем количество вхождений каждой скобки на количество противоположной, т.е. исключаем скобки в их
        # нормальном использовании
        t0 = brackets[list(brackets.keys())[0]]['count']
        t1 = brackets[list(brackets.keys())[1]]['count']
        brackets[list(brackets.keys())[0]]['count'] = max(0, brackets[list(brackets.keys())[0]]['count'] - t1)
        brackets[list(brackets.keys())[1]]['count'] = max(0, brackets[list(brackets.keys())[1]]['count'] - t0)

        # нормируем кол-во обычных скобок
        for bracket in brackets:
            if brackets[bracket]['count'] > 0:
                if brackets[bracket]['count'] < 3:
                    # на один или два знака ставим признак, равный 1
                    brackets[bracket]['count'] = 1
                else:
                    # на три и более ставим 2
                    brackets[bracket]['count'] = 2

        # переводим в булевый вектор
        for bracket in brackets:
            tmpResultCount = [False] * 2
            tmpResultStrongCount = [False] * 2
            for i in range(brackets[bracket]['count']):
                tmpResultCount[i] = True
            for i in range(brackets[bracket]['strongCount']):
                tmpResultStrongCount[i] = True

            # добавляем данные по скобкам в результат
            result.extend(tmpResultCount)
            result.extend(tmpResultStrongCount)

        return result

    def getSpeechActVerbsRatings(self, analysis, lemmas):
        """
        Проверка на наличие в тексте глаголов, соответствующих определённым речевым актам.
        Дополнительно реализована проверка на наличие глаголов в определённых формах и наклонениях
        :param analysis: данные по анализу слов текста
        :type analysis: list
        :param lemmas: список лемм слов текста
        :type lemmas: list
        :return: результат проверки в виде массива
        :rtype: list
        """
        # список обозначений искомых форм и наклонений согласно формата mystem
        verbFeatures = ["isImperative", "isIndicative", "isGerund", "isParticiple", "isInfinitive"]
        vector = [False] * len(self.speechActVerbsList)

        # первым делом проверяем слова соообщения на их наличие в словаре
        for index, lemma in enumerate(self.speechActVerbsList):
            if lemma in lemmas:
                vector[index] = True

        # затем итеративно ищем искомые формы и наклонения
        POSVector = [False] * len(verbFeatures)
        fullPOSVector = [True] * len(verbFeatures)
        for word in analysis:
            # если все искомые формы уже встретились, завершаем поиск
            if POSVector == fullPOSVector:
                break
            if word['POS'] != 'V':
                continue
            for index, verbFeature in enumerate(verbFeatures):
                if ('verbsCharacteristics' in word and
                        verbFeature in word['verbsCharacteristics'] and
                        word['verbsCharacteristics'][verbFeature]):
                    POSVector[index] = True

        vector.extend(POSVector)

        return vector
