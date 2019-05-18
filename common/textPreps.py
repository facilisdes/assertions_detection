from pymystem3 import Mystem


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


def analyzeText(text):
    """
    разбор и анализ слов текста
    :param text: текст для разбора
    :type text: string
    :return:
        - textAnalytics - данные по анализу всего текста (слова плюс другие символы)
        - wordsAnalytics - данные по анализу слов текста
        - lemmas - леммы слов текста
    :rtype: tuple
    """
    m = Mystem()
    rawAnalytics = m.analyze(text)

    textAnalytics = list()
    wordsAnalytics = list()
    lemmas = list()

    for rawAnalytic in rawAnalytics:
        if 'analysis' in rawAnalytic and len(rawAnalytic['analysis']) > 0:
            POS = rawAnalytic['analysis'][0]['gr'].split(',')[0]
            if '=' in POS:
                POS = POS.split('=')[0]

            isObscene = "обсц" in rawAnalytic['analysis'][0]['gr']

            lemma = rawAnalytic['analysis'][0]['lex']

            analytic = {
                'POS': POS,
                'lemma': lemma,
                'rawText': rawAnalytic['text'],
                'isObscene': isObscene
            }

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
            char = rawAnalytic['text']
            charType = "unrecognized"
            if char == " ":
                charType = "space"
            elif char in [",", ":", ";", "(", ")", "\""]:
                charType = "blockBreaker"
            elif char in ["."]:
                charType = "sentenceBreaker"

            analytic = {'type': 'separator', 'rawText': charType}
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

