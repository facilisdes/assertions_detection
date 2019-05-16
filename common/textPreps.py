from pymystem3 import Mystem


def prepareMessage(message):
    # step 1 - remove citations, etc
    message = __removeBrackets(message)

    return message


def analyzeMessage(message):
    m = Mystem()
    rawAnalytics = m.analyze(message)

    analytics = list()
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

            analytics.append(analytic)
            lemmas.append(lemma)

    return analytics, lemmas


def __removeBrackets(message):
    openingTags = ['\"', '\'', '(', '[']
    closingTags = ['\"', '\'', ')', ']']
    level = 0
    brackets = []
    bracket = []
    currentTag = ''
    for i, char in enumerate(message):
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
        if startIndex > 0 and message[bracket[0] - 1] == ' ':
            startIndex = startIndex - 1
        else:
            if endIndex < len(message) - 1 and message[endIndex + 1] == ' ':
                endIndex = endIndex + 1

        message = message[:startIndex] + message[endIndex:]

    return message

