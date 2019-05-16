import os, csv
from emoji import UNICODE_EMOJI


class semanticFeatures:
    def __init__(self, speechActVerbsList=False, opinionWordsList=False, smilesList=False, vulgarWordsList=False):
        """
        инициализация - чтение данных из файлов
        :param speechActVerbsList: string
        :param opinionWordsList: string
        :param smilesList: string
        :param vulgarWordsList: string
        """

        curDir = os.path.dirname(__file__)
        # берём текущий путь и считаем пути относительно его
        if speechActVerbsList == False:
            speechActVerbsList = 'data/speechActVerbs.csv'
            speechActVerbsList = os.path.join(curDir, speechActVerbsList)
        if opinionWordsList == False:
            opinionWordsList = 'data/opinionWords.csv'
            opinionWordsList = os.path.join(curDir, opinionWordsList)
        if smilesList == False:
            smilesList = 'data/smiles.csv'
            smilesList = os.path.join(curDir, smilesList)
        if vulgarWordsList == False:
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
            self.opinionWordsList = [el[0] for el in csv.reader(f, delimiter=";")]
        with open(smilesList, 'r', encoding='utf-8-sig') as f:
            self.smilesList = [el[0] for el in csv.reader(f, delimiter=";")]
        with open(vulgarWordsList, 'r', encoding='utf-8-sig') as f:
            self.vulgarWordsList = [el[0] for el in csv.reader(f, delimiter=";")]


    def getOpinionWordsRating(self, lemmas):
        """
        проверка на наличие в тексте слов с яркой эмоциональной окраской
        :param lemmas: list
        :return: int
        """
        result = 0
        for lemma in self.opinionWordsList:
            if lemma in lemmas:
                result = 1
                break

        return result

    def getVulgarWordsRating(self, analysis, lemmas):
        """
        проверка на наличие в тексте обсценной лексики
        :param analysis: list
        :param lemmas: list
        :return: int
        """
        result = 0
        # проверяем каждое обсценное слово из словаря на наличие в тексте
        for vulgarWord in self.vulgarWordsList:
            if vulgarWord in lemmas:
                result = 1
                break
        if result == 0:
            for word in analysis:
                if word['isObscene']:
                    result = 1
                    break

        return result

    def getSmilesRating(self, message, lemmas):
        """
        проверка на наличие в тексте смайликов и эмодзи
        :param message: list
        :param lemmas: list
        :return: int
        """
        result = 0
        # Проверяем все смайлики из словаря на наличие в тексте сообщения. Самая лёгкая проверка в надежде исключить
        # сообщение до начала проверки по большому списку эмодзи
        for smile in self.smilesList:
            if smile in message:
                result = 1
                break
        # если обычных смайликов всё же не нашлось, ищем эмодзи
        if result == 0:
            for lemma in lemmas:
                if lemma in UNICODE_EMOJI:
                    result = 1
                    break

        return result

    def getSpeechActVerbsRatings(self, analysis, lemmas):
        """
        Проверка на наличие в тексте глаголов, соответствующих определённым речевым актам.
        Дополнительно реализована проверка на наличие глаголов в определённых формах и наклонениях
        :param analysis: list
        :param lemmas: list
        :return: list
        """
        # список обозначений искомых форм и наклонений согласно формата mystem
        verbFeatures = ["isImperative","isIndicative","isGerund","isParticiple","isInfinitive"]
        vector = [0] * len(self.speechActVerbsList)

        # первым делом проверяем слова соообщения на их наличие в словаре
        for index, lemma in enumerate(self.speechActVerbsList):
            if lemma in lemmas:
                vector[index] = 1

        # затем итеративно ищем искомые формы и наклонения
        POSVector = [0] * len(verbFeatures)
        fullPOSVector = [1] * len(verbFeatures)
        for word in analysis:
            # если все искомые формы уже встретились, завершаем поиск
            if POSVector == fullPOSVector:
                break
            if word['POS'] != 'V': continue;
            for index, verbFeature in enumerate(verbFeatures):
                if (
                'verbsCharacteristics' in word and
                verbFeature in word['verbsCharacteristics'] and
                word['verbsCharacteristics'][verbFeature]
                ):
                    POSVector[index] = 1


        vector.extend(POSVector)

        return vector
