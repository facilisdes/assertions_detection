from common.debug import *
import math

class nGramsFeaturesBuilder:
    @staticmethod
    def train(self, analyticsList, classesList):
        """
        процесс обучения - эмпирического определения n-грам по размеченным текстам
        :return:
        :rtype:
        """
        groupsList = list()
        flatGroupsList = list()
        candidatesGroupsList = list()

        for analytic in analyticsList:
            wordsGroup = self.__buildWordsGroups(analytic)
            groupsList.append(wordsGroup)
            flatGroupsList.extend(wordsGroup)

        ngramsCandidates = self.__buildRawNGrams(flatGroupsList, 3)

        # модифицируем группы, прописывая в них частоты
        ngramsCandidates = self.__countWordsDocumentFrequency(ngramsCandidates, groupsList)

        filteredCandidates = self.__filterNGramsByFrequency(ngramsCandidates, inMessagesThreshold=2, minFreqThreshold=3)
        self.__calculateNGramsEntropy(filteredCandidates, classesList)
        filteredCandidates = self.__filterNGramsByEntropy(filteredCandidates, entropyThreshold=0.45)

        return filteredCandidates

    @staticmethod
    def __buildWordsGroups(analytics):
        """
        поиск групп слов, разделённых только пробелами
        :param analytics: данные по анализу слов текста
        :type analytics: list
        :return: список групп слов в тексте
        :rtype: list
        """
        nonBreakingTypes = ['space', 'word', 'enText', 'ruText', 'number']
        ignoredTypes = ['space']

        wordsBlocks = list()
        wordsBlock = list()
        # перебираем все объекты текста
        for analytic in analytics:
            # если тип объекта не разрывает группу
            if analytic['type'] in nonBreakingTypes:
                # записываем объект в группу, если он не должен игнорироваться
                if analytic['type'] not in ignoredTypes:
                    wordsBlock.append(analytic)
            # иначе, если группа не пуста, записываем её в список групп и сбрасываем
            elif len(wordsBlock) > 0:
                wordsBlocks.append(wordsBlock)
                wordsBlock = list()

        # по окончании работ записываем оставшуюся группу, если она не пуста
        if len(wordsBlock) > 0:
            wordsBlocks.append(wordsBlock)

        return wordsBlocks

    @staticmethod
    def getNGramsList(self):
        return ['a', 'b']

    @staticmethod
    def __buildRawNGrams(self, wordsGroups, maxLength):
        """
        поиск "сырых" n-грамм в группах слов текста
        :param wordsGroups: список разделяемых пробелами групп слов
        :type wordsGroups: list
        :param maxLength: максимальная длина n-грамм
        :type maxLength: int
        :return: словарь сырых n-грамм вида "длина: список n-грам"
        :rtype: list
        """
        ngrams = list()
        ignoredPOSes = ["CONJ", "PR", "INTJ", "NUM", "PART"]
        ngramsWords = list()

        # запускаем построение для каждой группы текста
        for wordsGroup in wordsGroups:
            # перебираем все нужные длины n-грам
            for length in range(1, maxLength + 1):
                groups = list()
                group = list()
                # перебираем слова в группе
                for word in wordsGroup:
                    # хак - пропускаем предлоги и другие служебные части речи в качестве n-граммы длины 1
                    if length == 1 and (word['type'] != 'word' or ('POS' in word.keys() and word['POS'] in ignoredPOSes)):
                        continue
                    # добавляем слово в группу
                    group.append(word)
                    # и, если длина группы равна макс. длине n-граммы
                    if len(group) == length:
                        # строим текстовое представление группы для проверки на уникальность
                        groupTextRepresentation = ' '.join(word['text'] for word in group)
                        # и, если группа уникальна
                        if groupTextRepresentation not in ngramsWords:
                            # пишем группу в список групп и в список текстовых представлений групп
                            groups.append(group)
                            ngramsWords.append(groupTextRepresentation)
                        # затем сдвигаем индексы, затирая первый элемент группы
                        group = group[1:]
                # по итогу добавляем найденные группы в общий список
                ngrams.extend(groups)

        return ngrams

    @staticmethod
    def __countWordsDocumentFrequency(self, groupsToCount, groupsByTexts):
        """
        расчёт частот встречаемости групп слов (n-грам) в группах слов текста
        :param groupsToCount: список n-грам
        :type groupsToCount: list
        :param groupsByTexts: список групп слов, сгруппированных по текстам
        :type groupsByTexts: list
        :return: словарь с перечислением элементов n-граммы и частот их встречаемости
        :rtype: list
        """
        result = list()

        # перебираем список групп, частоты которых нужно посчитать
        for groupIndex, groupToCount in enumerate(groupsToCount):
            # берём первое слово группы
            wordToFind = groupToCount[0]
            # и пропускаем его, если оно - имя собственное
            if 'isPersonal' in wordToFind.keys() and wordToFind['isPersonal']:
                continue
            freqByGroups = list()
            inMessages = 0
            # и затем ищем то же слово во всех текстах
            for groups in groupsByTexts:
                freq = 0
                # перебираем группы текущего текста
                for group in groups:
                    # в группе перебираем слова
                    for i, word in enumerate(group):
                        # если текущее слово совпало с первым
                        if word['text'] == wordToFind['text']:
                            found = True
                            # проверяем последующие слова проверяемой группы
                            for j, wordToCheck in enumerate(groupToCount[1:]):
                                # поскольку индексы начинаются с нуля, а по факту должны с единицы (ведь проверяем не
                                # первое, а последующие слова), увеличиваем индекс
                                j = j+1
                                # проверяем, кончается ли проверяемая группа раньше
                                # и, если нет, проверяем, не различаются ли слова групп
                                if (i+j) >= len(group) or group[i+j]['text'] != groupToCount[j]['text']:
                                    # и, если это так, бракуем текущую проверку
                                    found = False
                                    break

                                # также бракуем её для имён собственных
                                if 'isPersonal' in wordToFind.keys() and wordToFind['isPersonal']:
                                    found = False
                                    break

                            # если все слова блока нашлись, увеличиваем счётчик находок
                            if found:
                                freq = freq + 1

                # по окончании проверки всех групп текста записываем частоту вхождений проверяемой группы в
                # проверенный текст
                freqByGroups.append(freq)
                if freq > 0:
                    inMessages = inMessages + 1

            # и после перебора всех текстов формируем массив с результатом проверки, эл-ты n-граммы записываем по
            # ссылке чтобы не тратить память
            result.append({'items': groupsToCount[groupIndex], 'freqByGroups': freqByGroups,
                           'totalFreq': sum(freqByGroups), 'inMessages': inMessages, })

        return result

    @staticmethod
    def __filterNGramsByFrequency(self, ngramsList, inMessagesThreshold=2, minFreqThreshold=3):
        filteredNGrams = list()
        for ngram in ngramsList:
            if ngram['inMessages'] < inMessagesThreshold or ngram['totalFreq'] < minFreqThreshold:
                continue
            filteredNGrams.append(ngram)
        return filteredNGrams

    @staticmethod
    def __filterNGramsByEntropy(self, ngramsList, entropyThreshold=0.2):
        filteredNGrams = list()
        for ngram in ngramsList:
            if ngram['normalizedEntropy'] < entropyThreshold:
                filteredNGrams.append(ngram)
        return filteredNGrams

    @staticmethod
    def __calculateNGramsEntropy(self, ngramsList, classes):
        messagesInClasses = dict()

        # формируем список сообщений в каждом классе
        for i, textClass in enumerate(classes):
            if textClass not in messagesInClasses.keys():
                messagesInClasses[textClass] = list()
            messagesInClasses[textClass].append(classes[i])

        # перебираем все n-граммы
        for ngram in ngramsList:
            ngramEntropyByClass = dict()
            countOfMessagesInClasses = dict()
            freqByClasses = dict()

            # для каждой n-граммы перебираем тексты и их классы
            for i, textClass in enumerate(classes):
                # и считаем, в скольки сообщениях каждого класса встречается эта n-грамма (документная частотоа),
                # а также частоту включений n-граммы в сообщения каждого класса
                isInMessage = 1 if ngram['freqByGroups'][i] > 0 else 0
                if textClass in countOfMessagesInClasses.keys():
                    countOfMessagesInClasses[textClass] = countOfMessagesInClasses[textClass] + isInMessage
                    freqByClasses[textClass] = freqByClasses[textClass] + ngram['freqByGroups'][i]
                else:
                    countOfMessagesInClasses[textClass] = isInMessage
                    freqByClasses[textClass] = ngram['freqByGroups'][i]

            # записываем количество сообщений класса, в которых встречается эта n-грамма
            ngram['freqOfMessagesInClasses'] = countOfMessagesInClasses
            # а также количество включений n-граммы в сообщения класса
            ngram['freqByClasses'] = freqByClasses

            sumEntropy = 0
            # перебираем классы, для которых считали документную частоту
            for textClass in countOfMessagesInClasses:
                # если n-грамма встречалась в сообщениях класса хоть раз,
                if countOfMessagesInClasses[textClass] > 0 and ngram['inMessages'] > 0:
                    # то считаем вероятность того, что в конкретном сообщении встретится текущая n-грамма
                    # то есть делим число сообщений, в которых есть n-грамма, на общее число сообщений
                    # p = countOfMessagesInClasses[textClass] / ngram['inMessages']
                    p = freqByClasses[textClass] / ngram['totalFreq']
                    # затем находим энтропию
                    entropy = -1 * p * math.log(p, 2)
                    ngramEntropyByClass[textClass] = entropy
                    # и суммарную энтропию по всем классам
                    sumEntropy = sumEntropy + entropy
                else:
                    # если n-грамма не встретилась в сообщениях класса ни разу, то и энтропия равна нулю
                    ngramEntropyByClass[textClass] = 0

            # по окончании проверки всех классов нормализуем суммарную энтропию - делим её на общее количество сообщений
            normalizedEntropy = sumEntropy / math.log(len(classes), math.e)
            ngram['normalizedEntropy'] = normalizedEntropy
