from pymystem3 import Mystem
from common import caching as cachingWorker
import pickle
import hashlib
import re


def prepareText(text):
    """
    очистка текста
    :param text: текст для обработки
    :type text: string
    :return: обработанный текст
    :rtype: string
    """
    # первый шаг - удаление текстов в скобках
    text = __removeBrackets(text)

    return text


def analyzeText(text, caching=True):
    """
    разбор и анализ слов текста
    :param text: текст для разбора
    :type text: string
    :param caching: кешировать ли результат разбор
    :type caching: bool
    :return:
        - textAnalytics - данные по анализу всего текста (слова плюс другие символы)
        - wordsAnalytics - данные по анализу слов текста
        - lemmas - леммы слов текста
    :rtype: tuple
    """
    result = None

    # если кеширование включено
    if caching:
        # строим хеш
        hash = hashlib.md5(text.encode("utf-8")).hexdigest()
        hash = hash + hashlib.sha1(text.encode("utf-8")).hexdigest()

        # пытаемся читать
        rawResult = cachingWorker.readByte(hash)
        if rawResult is not None:
            result = pickle.loads(rawResult)

    # если ничего не прочитано, получаем данные обычным способом
    if result is None:
        result = __mystemWrapper(text)

    # и, если включено кеширование,
    if caching:
        # сохраняем данные
        rawResult = pickle.dumps(result)
        cachingWorker.saveByte(hash, rawResult)

    return result


def __mystemWrapper(text):
    """
    обёртка для разбора текста через mystem
    :param text: текст для разбора
    :type text: string
    :return:
        - textAnalytics - данные по анализу всего текста (слова плюс другие символы)
        - wordsAnalytics - данные по анализу слов текста
        - lemmas - леммы слов текста
    :rtype: tuple
    """
    # символы, разбивающие блоки текста внутри предложения
    blockBreakers = [",", ":", ";", "(", ")", "\"", "-"]
    # символы, разбивающие предложения
    sentenceBreakers = [".", "!", "?"]
    # символы, разбивающие слова внутри блока
    wordsBreakers = [" ", "-"]
    m = Mystem()
    rawAnalytics = m.analyze(text)

    textAnalytics = list()
    wordsAnalytics = list()
    lemmas = list()

    # перебираем выход mystem
    for rawAnalytic in rawAnalytics:
        # если по аналитике есть данные, копаемся в них
        if 'analysis' in rawAnalytic and len(rawAnalytic['analysis']) > 0:
            # первым делом определяем часть речи
            POS = rawAnalytic['analysis'][0]['gr'].split(',')[0]
            if '=' in POS:
                POS = POS.split('=')[0]

            # затем признак обсценности
            isObscene = "обсц" in rawAnalytic['analysis'][0]['gr']

            # затем лемму
            lemma = rawAnalytic['analysis'][0]['lex']

            # cтроим результирующий словарь
            analytic = {
                'POS': POS,
                'text': lemma,
                'rawText': rawAnalytic['text'],
                'isObscene': isObscene
            }

            # для глаголов расширяем его другими признаками
            if POS == 'V':
                isImperative = "пов" in rawAnalytic['analysis'][0]['gr']
                isIndicative = "изъяв" in rawAnalytic['analysis'][0]['gr']
                isGerund = "деепр" in rawAnalytic['analysis'][0]['gr']
                isParticiple = "прич" in rawAnalytic['analysis'][0]['gr']
                isInfinitive = "инф" in rawAnalytic['analysis'][0]['gr']

                analytic['verbsCharacteristics'] = {
                    'isImperative': isImperative,
                    'isIndicative': isIndicative,
                    'isGerund': isGerund,
                    'isParticiple': isParticiple,
                    'isInfinitive': isInfinitive
                }

            wordsAnalytics.append(analytic)
            lemmas.append(lemma)
            analytic['type'] = 'word'
            textAnalytics.append(analytic)
        else:
            # для текста, не разобранного mystem как слово, определяем тип
            char = rawAnalytic['text']
            charTrimmed = char.strip()
            charType = "unrecognized"
            if char in wordsBreakers:
                charType = "space"
                charTrimmed = char
            elif charTrimmed in blockBreakers:
                charType = "blockBreaker"
            elif charTrimmed in sentenceBreakers:
                charType = "sentenceBreaker"
            elif re.match(r"^[a-zA-Z]+$", char):
                charType = "enText"
                # cлова на английском mystem не обрабатывает, так что вручную записываем их в леммы
                lemmas.append(char)
            elif re.match(r"^[а-яА-ЯёЁ]+$", char):
                charType = "ruText"
                # нераспознанные cлова на русском
                lemmas.append(char)
            elif re.match(r"^[0-9]+$", char):
                charType = "number"
                # число
                lemmas.append(char)

            analytic = {'type': charType, 'rawText': char, 'text': charTrimmed}
            textAnalytics.append(analytic)

    return textAnalytics, wordsAnalytics, lemmas


def __removeBrackets(text):
    """
    удаление текстов в скобках
    :param text: текст для очистки
    :type text: string
    :return: очищенный текст
    :rtype: string
    """
    openingTags = ['\"', '\'', '(', '[']
    closingTags = ['\"', '\'', ')', ']']
    level = 0
    brackets = []
    bracket = []
    currentTag = ''
    for i, char in enumerate(text):
        if char in openingTags and level == 0:
            level = level + 1
            bracket = [i]
            currentTag = closingTags[openingTags.index(char)]

        else:
            if char in closingTags and char == currentTag and level == 1:
                level = level - 1
                bracket.append(i)
                brackets.append(bracket)

    for bracket in reversed(brackets):
        startIndex = bracket[0]
        endIndex = bracket[1] + 1
        if startIndex > 0 and text[bracket[0] - 1] == ' ':
            startIndex = startIndex - 1
        else:
            if endIndex < len(text) - 1 and text[endIndex + 1] == ' ':
                endIndex = endIndex + 1

        text = text[:startIndex] + text[endIndex:]

    return text

