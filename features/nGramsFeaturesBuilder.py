class nGramsFeaturesBuilder:
    def __init__(self, messagesList, classesList):
        """
        инициализация
        :param messagesList: list
        :param classesList: list
        """
        self.messages = messagesList
        for message in messagesList:
            wordsGroup = self.__buildWordsGroups(message)
            ngrams = self.__buildRawNGrams(wordsGroup)


    def __buildWordsGroups(self, message):
        """
        поиск групп слов, разделённых только пробелами - кандидатов в n-граммы
        :param message: list
        :return: list
        """
        wordsBlocks = list()
        wordsBlock = list()
        for analytic in message:
            if len(wordsBlock) > 0:
                wordsBlocks.append(wordsBlock)
                wordsBlock = list()
            else:
                # если данные по анализу есть, фиксируем текст в блоке
                wordsBlock.append(analytic)

        if len(wordsBlock) > 0:
            wordsBlocks.append(wordsBlock)
        return wordsBlocks

    def getNGramsList(self):
        return ['a', 'b']

    def __buildRawNGrams(self, wordsGroups):
        wordsList = list()
        unograms = list()
        bigrams = list()
        trigrams = list()
        for wordsGroup in wordsGroups:
            for i, word in enumerate(wordsGroup):
                unograms.append(word)
                if i+2 < len(wordsGroup):
                    trigrams.append(wordsGroup[i:i+3])
                    bigrams.append(wordsGroup[i:i+2])
                elif i+1 < len(wordsGroup):
                    bigrams.append(wordsGroup[i:i+2])

        wordsList.extend(unograms)
        wordsList.extend(bigrams)
        wordsList.extend(trigrams)

        return {'ngrams': wordsList, 'unograms':unograms, 'bigrams':bigrams, 'trigrams':trigrams}

    def __countWordsDocumentFrequency(self):
        return 0

    def __filterWordsByFrequency(self):
        return 0

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

    class Word:
        def __init__(self, text):
            self.text = text

    class Message:
        def __init__(self, words, messageClass):
            self.words = []
            self.wordsRaw = []
            self.messageClass = messageClass
            for word in words:
                wordObj = nGramsFeaturesBuilder.Word(word)
                self.words.append(wordObj)
                self.wordsRaw.append(word)

