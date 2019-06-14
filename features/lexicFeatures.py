import csv
import os


class lexicFeatures:
    def __init__(self, abbreviationsFile='data/abbreviations.csv'):
        """
        инициализация объекта
        :param abbreviationsFile: (опционально) путь к файлу с сокращениями
        :type abbreviationsFile: str
        """
        curDir = os.path.dirname(__file__)
        abbreviationsFile = os.path.join(curDir, abbreviationsFile)

        if not os.path.isfile(abbreviationsFile):
            raise Exception("Abbreviations file (%s) does not exists" % abbreviationsFile)
        with open(abbreviationsFile, 'r', encoding='utf-8-sig') as f:
            self.abbreviationsList = list(csv.reader(f, delimiter=";"))

    def getPunctuantionRatings(self, message):
        """
        проверка на наличие и размещение определенных знаков пунктуации в тексте сообщения
        :param message: текст сообщения
        :type message: str
        :return: результат проверки в виде массива
        :rtype: list
        """

        result = list()
        marks = {
            '!' : {
                'count': message.count('!'),
                'repeatedness': False
            },
            '?' : {
                'count': message.count('?'),
                'repeatedness': False
            }
        }
        marks = ['!', '?']

        for mark in marks:
            markCount = message.count(mark)
            repeatedness = False

            if markCount > 0:
                if markCount < 3:
                    # на один или два знака ставим признак, равный 1
                    markCount = 1
                else:
                    # на три и более ставим 2
                    markCount = 2

                if mark*2 in message:
                    # считаем, есть ли что-то вроде "!!!!!" в тексте
                    repeatedness = True

            # представление результата в виде булевого вектора
            marksResult = [False] * 3
            for mark in range(markCount+1):
                marksResult[mark] = True

            result.append(repeatedness)
            result.extend(marksResult)

        return result

    def getAbbreviationsRatings(self, lemmas):
        """
        проверка на наличие аббревиатур в тексте сообщения
        :param lemmas: список лемм сообщения
        :type lemmas: list
        :return: результат проверки в виде массива
        :rtype: list
        """
        # по умолчанию все признаки равны 0
        result = [False] * len(self.abbreviationsList)

        for index, abbreviation in enumerate(self.abbreviationsList):
            # если элемент списка сокращений не список, значит, это единичная аббревиатура
            if type(abbreviation) != list:
                # если аббревиатура равна проверяемому слову сообщения, то признак равен 1
                if abbreviation in lemmas:
                    result[index] = True
            # иначе элемент это список, тогда проверяем все аббревиатуры в этом списке
            else:
                for abbreviationElement in abbreviation:
                    if abbreviationElement in lemmas:
                        # for  abbreviation in lemmas://////////////////ур, то признак равен 1
                        result[index] = True
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

        INTJFound = False
        AFound = False
        imperative_verb_found = False
        indicative_verb_found = False
        gerund_verb_found = False
        participle_verb_found = False
        infinitive_verb_found = False

        # проверяем каждое слово сообщения
        for word in analysis:
            POS = word['POS']
            if '=' in POS:
                POS = POS.split('=')[0]

            if POS == 'INTJ':
                INTJFound = True

            if POS == 'A' or POS == 'ANUM' or POS == 'APRO':
                AFound = True

            if POS == 'V':
                imperative_verb_found = imperative_verb_found or word['verbsCharacteristics']['isImperative']
                indicative_verb_found = indicative_verb_found or word['verbsCharacteristics']['isIndicative']
                gerund_verb_found = gerund_verb_found or word['verbsCharacteristics']['isGerund']
                participle_verb_found = participle_verb_found or word['verbsCharacteristics']['isParticiple']
                infinitive_verb_found = infinitive_verb_found or word['verbsCharacteristics']['isInfinitive']

            if INTJFound and AFound and imperative_verb_found and indicative_verb_found and gerund_verb_found and \
                    participle_verb_found and infinitive_verb_found:
                break

        result = [INTJFound, AFound, imperative_verb_found, indicative_verb_found, gerund_verb_found,
                  participle_verb_found, infinitive_verb_found]
        return result
