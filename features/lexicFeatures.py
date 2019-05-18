import csv
import os


class lexicFeatures:
    def __init__(self, abbreviationsFile='data/abbreviations.csv'):
        """
        инициализация объекта
        :param abbreviationsFile: (опционально) путь к файлу с сокращениями
        :type abbreviationsFile: string
        """
        curDir = os.path.dirname(__file__)
        abbreviationsFile = os.path.join(curDir, abbreviationsFile)

        if not os.path.isfile(abbreviationsFile):
            raise Exception("Abbreviations file (%s) does not exists" % abbreviationsFile)
        with open(abbreviationsFile, 'r', encoding='utf-8-sig') as f:
            self.abbreviationsList = list(csv.reader(f, delimiter=";"))

    def getPunctuantionRatings(self, message):
        """
        проверка на наличие определенных знаков пунктуации в тексте сообщения
        :param message: текст сообщения
        :type message: string
        :return: результат проверки в виде массива
        :rtype: list
        """
        a = 0
        b = 0
        if '!' in message:
            a = 1
        if '?' in message:
            b = 1

        return [a, b]

    def getTwitterSpecificsRatings(self, message):
        """
        проверка на наличие twitter-специфичных символов в тексте сообщения, а также их нахождения в начале сообщения
        :param message: текст сообщения
        :type message: string
        :return: результат проверки в виде массива
        :rtype: list
        """
        a = 0
        aLocation = 0
        b = 0
        bLocation = 0
        c = 0
        cLocation = 0

        indexA = message.find('@')
        indexB = message.find('#')
        indexC = message.find('RT')
        if indexA >= 0:
            a = 1
            if indexA == 0:
                aLocation = 1
        if indexB >= 0:
            b = 1
            if indexB == 0:
                bLocation = 1
        if indexC >= 0:
            c = 1
            if indexC == 0:
                cLocation = 1

        return [a, aLocation, b, bLocation, c, cLocation]

    def getAbbreviationsRatings(self, lemmas):
        """
        проверка на наличие аббревиатур в тексте сообщения
        :param lemmas: список лемм сообщения
        :type lemmas: list
        :return: результат проверки в виде массива
        :rtype: list
        """
        # по умолчанию все признаки равны 0
        result = [0] * len(self.abbreviationsList)

        for index, abbreviation in enumerate(self.abbreviationsList):
            # если элемент списка сокращений не список, значит, это единичная аббревиатура
            if type(abbreviation) != list:
                # если аббревиатура равна проверяемому слову сообщения, то признак равен 1
                if abbreviation in lemmas:
                    result[index] = 1
            # иначе элемент это список, тогда проверяем все аббревиатуры в этом списке
            else:
                for abbreviationElement in abbreviation:
                    if abbreviationElement in lemmas:
                        # for  abbreviation in lemmas://////////////////ур, то признак равен 1
                        result[index] = 1
                        break
        return result

    def getPOSRatings(self, analysis):
        """
        проверка на наличие определенных частей речи
        :param analysis: данные по анализу слов сообщения
        :type analysis: list
        :return: результат проверки в виде массива
        :rtype: list
        """
        result = [0, 0]
        INTJFound = False
        AFound = False
        # проверяем каждое слово сообщения
        for word in analysis:
            POS = word['POS']
            if '=' in POS:
                POS = POS.split('=')[0]

            if POS == 'INTJ':
                INTJFound = True
                result[0] = 1

            if POS == 'A' or POS == 'ANUM' or POS == 'APRO':
                AFound = True
                result[1] = 1

            if INTJFound and AFound:
                break

        return result
