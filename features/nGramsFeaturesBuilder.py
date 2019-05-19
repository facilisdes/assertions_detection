from common.debug import *

class nGramsFeaturesBuilder:
    def __init__(self, analyticsList, classesList):
        """
        инициализация
        :param messagesList: list
        :type messagesList: list
        :param classesList: list
        :type classesList: list
        """
        self.analytics = analyticsList
        self.__train()

    def __train(self):
        """
        процесс обучения - эмпирического определения n-грам по размеченным текстам
        :return:
        :rtype:
        """
        analyticsList = self.analytics
        groupsList = list()
        flatGroupsList = list()
        candidatesGroupsList = list()

        for analytic in analyticsList:
            wordsGroup = self.__buildWordsGroups(analytic)
            groupsList.append(wordsGroup)
            flatGroupsList.extend(wordsGroup)
            ngramsCandidates = self.__buildRawNGrams(wordsGroup, 3)
            candidatesGroupsList.append(ngramsCandidates)

        for i, candidatesGroups in enumerate(candidatesGroupsList):
            analytic = analyticsList[i]
            wordsGroup = groupsList[i]

            for candidateKey in candidatesGroups:
                # модифицируем группы, прописывая в них частоты
                candidatesGroups[candidateKey] = self.__countWordsDocumentFrequency(candidatesGroups[candidateKey], groupsList)

        filteredCandidates = self.__filterNGramsByFrequency(candidatesGroupsList)
        q=1

    def __buildWordsGroups(self, analytics):
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


    def getNGramsList(self):
        return ['a', 'b']

    def __buildRawNGrams(self, wordsGroups, maxLength):
        """
        поиск "сырых" n-грамм в группах слов текста
        :param wordsGroups: список разделяемых пробелами групп слов
        :type wordsGroups: list
        :param maxLength: максимальная длина n-грамм
        :type maxLength: int
        :return: словарь сырых n-грамм вида "длина: список n-грам"
        :rtype: dict
        """
        ngrams = dict()
        for length in range(1, maxLength + 1):
            ngrams[length] = list()

        # запускаем построение для каждой группы текста
        for wordsGroup in wordsGroups:
            # перебираем все нужные длины n-грам
            for length in range(1, maxLength + 1):
                groups = list()
                group = list()
                # перебираем слова в группе
                for word in wordsGroup:
                    # хак - пропускаем предлоги в качестве n-граммы длины 1
                    if length == 1 and (word['type'] != 'word' or ('POS' in word.keys() and word['POS'] == 'PR')):
                        continue
                    # добавляем слово в группу
                    group.append(word)
                    # и, если длина группы равна макс. длине n-граммы
                    if len(group) == length:
                        # пишем группу в список групп
                        groups.append(group)
                        # и сдвигаем индексы, затирая первый элемент
                        group = group[1:]
                # по итогу добавляем найденные группы в общий список
                ngrams[length].extend(groups)

        return ngrams

    def __countWordsDocumentFrequency(self, groupsToCount, groupsByTexts):
        """
        расчёт частот встречаемости групп слов (n-грам) в группах слов текста
        :param groupsToCount: список n-грам
        :type groupsToCount: list
        :param groupsByTexts: список групп слов, сгруппированных по текстам
        :type groupsByTexts: list
        :return: словарь с перечислением элементов n-граммы и частот их встречаемости
        :rtype: dict
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
                           'inMessages': inMessages, })

        return result

    def __filterNGramsByFrequency(self, ngramsByTexts, inMessagesThreshold=2, minFreqThreshold=3):
        ngrams = list()
        for i, ngramsByLengths in enumerate(ngramsByTexts):
            for j in ngramsByLengths:
                for k, ngram in enumerate(ngramsByLengths[j]):
                    if ngram['inMessages'] < inMessagesThreshold or sum(ngram['freqByGroups']) < minFreqThreshold:
                        continue
                    ngrams.append(ngram)
        return ngrams

    def __buildWordsPOSList(self):
        return 0

    def __filterWordsByPOS(self):
        return 0

    def __buildWordsEntropyMatrix(self):
        return 0

    def __buildWordsTotalEntropyList(self):
        return 0

    def __filterWordsByEntropy(self):
        return 0