import os, csv
import re
from emoji import UNICODE_EMOJI


class semanticFeatures:
    def __init__(self,
                 speechActVerbsList='data/speechActVerbs.csv',
                 opinionWordsList='data/opinionWords.csv',
                 smilesList='data/smiles.csv',
                 vulgarWordsList='data/vulgarWords.csv',
                 ):

        curDir = os.path.dirname(__file__)
        SAVFile = os.path.join(curDir, speechActVerbsList)
        OWFile = os.path.join(curDir, opinionWordsList)
        smilesFile = os.path.join(curDir, smilesList)
        VWFile = os.path.join(curDir, vulgarWordsList)

        if not os.path.isfile(SAVFile):
            raise Exception("Speech act verbs file (%s) does not exists" % SAVFile)
        if not os.path.isfile(OWFile):
            raise Exception("Opinions words file (%s) does not exists" % OWFile)
        if not os.path.isfile(smilesFile):
            raise Exception("Smiles file (%s) does not exists" % smilesFile)
        if not os.path.isfile(VWFile):
            raise Exception("Smiles file (%s) does not exists" % VWFile)

        with open(SAVFile, 'r', encoding='utf-8-sig') as f:
            self.speechActVerbsList = [el[0] for el in csv.reader(f, delimiter=";")]

        with open(OWFile, 'r', encoding='utf-8-sig') as f:
            self.opinionWordsList = [el[0] for el in csv.reader(f, delimiter=";")]

        with open(smilesFile, 'r', encoding='utf-8-sig') as f:
            self.smilesList = [el[0] for el in csv.reader(f, delimiter=";")]

        with open(VWFile, 'r', encoding='utf-8-sig') as f:
            self.vulgarWordsList = [el[0] for el in csv.reader(f, delimiter=";")]


    def getOpinionWordsRating(self, lemmas):
        result = 0
        for lemma in self.opinionWordsList:
            if lemma in lemmas:
                result = 1
                break

        return result

    def getVulgarWordsRating(self, analysis, lemmas):
        result = 0
        for vulgarWord in self.vulgarWordsList:
            if vulgarWord in lemmas:
                result = 1
                break
        # проверяем каждое слово сообщения
        if result == 0:
            for word in analysis:
                # опираемся на формат выхода mystem - если в массиве на сообщение есть индекс analysis и в нём больше 0 эл-тов
                if 'analysis' in word and len(word['analysis']) > 0:
                    # , то пометка об обсценности находится в первом подмассиве под индексом gr, проверяем её наличие
                    if("обсц" in word['analysis'][0]['gr']):
                        result = 1
                        break

        return result

    def getSmilesRating(self, message, lemmas):
        result = 0
        for smile in self.smilesList:
            if smile in message:
                result = 1
                break
        if result == 0:
            for lemma in lemmas:
                if lemma in UNICODE_EMOJI:
                    result = 1
                    break

        return result

    def getSpeechActVerbsRatings(self, analysis):
        verbFeatures = ["пов", "изъяв", "деепр", "прич", "инф"]
        vector = [0] * len(verbFeatures)
        fullVector = [1] * len(verbFeatures)
        for word in analysis:
            if vector == fullVector:
                break
            if 'analysis' in word and len(word['analysis']) > 0:
                POS = word['analysis'][0]['gr'].split(',')[0]
                if '=' in POS:
                    POS = POS.split('=')[0]
                if POS != 'V':
                    continue
                for index, verbFeature in enumerate(verbFeatures):
                    if(verbFeature in word['analysis'][0]['gr']):
                        vector[index] = 1

        # vector = [0] * len(self.speechActVerbsList)
        # for index, lemma in enumerate(self.speechActVerbsList):
        #     if lemma in lemmas:
        #         vector[index] = 1
        return vector
