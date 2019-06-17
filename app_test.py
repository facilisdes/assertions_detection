import csv
import hashlib
import os

from sklearn import model_selection

from classification.main import classifiers
from common import caching
from common import textPreps
from features.main import featuresExtractor

def getSplits(X, Y, doSplits = True, randomState = 123456):
    if doSplits:
        XTrain, XTest, YTrain, YTest= \
            model_selection.train_test_split(X, Y, test_size=0.2, random_state=randomState, stratify=Y)
    else:
        XTrain = inputFixedMessages
        XTest = []
        YTrain = inputClasses
        YTest = []

    return XTrain, XTest, YTrain, YTest

# общие параметры работы программы
MODE = [
    True,   # разделять ли выборку на обучающую и тестовую
    False,  # делать ли вместо обучения поиск гиперпараметров
    False,    # делать предсказание по всем моделям вместо лучшей
    False  # сбрасывать ли кеш
]
ID = "dataset 1 test"
if MODE[0] == False:
    ID = "dataset 2 test"

ID+= ' with params ' + ','.join(map(str,MODE))
caching.saveVar("prediction for test on " + ID, None)
caching.saveVar("featuresVectors for " + ID, None)
caching.saveVar("featuresVectors for test on " + ID, None)

if MODE[3] == True:
    caching.saveVar("featuresVectors for " + ID, None)
    caching.saveVar("featuresVectors for test on " + ID, None)
    caching.saveVar("prediction for test on " + ID, None)
    caching.saveVar('ngrams for ' + ID, None)

# сообщения обучающей выборки
curDir = os.path.dirname(__file__)
inputFile = os.path.join(curDir, "data/trainMessages.csv")

if not os.path.isfile(inputFile):
    raise Exception("Train messages file (%s) does not exists" % inputFile)
with open(inputFile, 'r', encoding='utf-8-sig') as f:
    inputMessagesList = list(csv.reader(f, delimiter=";"))

inputMessages = [message[0] for message in inputMessagesList]
# они же, но очищенные
inputFixedMessages = [textPreps.prepareText(inputMessage) for inputMessage in inputMessages]
# их классы
inputClasses = [message[1] for message in inputMessagesList]

trainMessages0, testMessages0, trainClasses0, testClasses0 = getSplits(inputFixedMessages, inputClasses, doSplits=False, randomState=None)
trainMessages1, testMessages1, trainClasses1, testClasses1 = getSplits(inputFixedMessages, inputClasses, randomState=None)
trainMessages2, testMessages2, trainClasses2, testClasses2 = getSplits(inputFixedMessages, inputClasses, randomState=None)
trainMessages3, testMessages3, trainClasses3, testClasses3 = getSplits(inputFixedMessages, inputClasses, randomState=None)
trainMessages4, testMessages4, trainClasses4, testClasses4 = getSplits(inputFixedMessages, inputClasses, randomState=None)
trainMessages5, testMessages5, trainClasses5, testClasses5 = getSplits(inputFixedMessages, inputClasses, randomState=None)
trainMessages6, testMessages6, trainClasses6, testClasses6 = getSplits(inputFixedMessages, inputClasses, randomState=None)

TM = [trainMessages1, trainMessages2, trainMessages3, trainMessages4, trainMessages5, trainMessages6]
tM = [testMessages1, testMessages2, testMessages3, testMessages4, testMessages5, testMessages6]
TC = [trainClasses1, trainClasses2, trainClasses3, trainClasses4, trainClasses5, trainClasses6]
tC = [testClasses1, testClasses2, testClasses3, testClasses4, testClasses5, testClasses6]
TFV = []

caching.saveVar(ID, None)
fe = featuresExtractor()
if fe.isTrainingRequired(ID):
    fe.trainNGrams(trainMessages0, trainClasses0, cache=True, textsHash = ID)

TFV = caching.readVar("featuresVectors for " + ID)
if TFV is None:
    TFV = []
    for trainMessages in TM:
        trainFeatures = []
        for message in trainMessages:
            featuresVector = fe.getFeaturesVector(message)
            trainFeatures.append(featuresVector)
        TFV.append(trainFeatures)
    caching.saveVar("featuresVectors for " + ID, TFV)

# анализируемые сообщения
messages = []
# значения их признаков
features = []
# их классы
classes = []

cll = classifiers()

FR = []
for i, cl in enumerate(TC):
    cll.trainModels(TFV[i], TC[i])

    testFeatures = []
    for message in tM[i]:
        featuresVector = fe.getFeaturesVector(message)
        testFeatures.append(featuresVector)

    prediction = cll.predictModels(testFeatures)

    classes = set(tC[i])
    classes = list(map(str, classes))
    f1Errors = {c: {'FP': 0, 'FN': 0, 'TP': 0, 'TN': 0} for c in classes}
    f1Scores = {}
    for model in prediction:
        modelPrediction = prediction[model]
        for j, classPrediction in enumerate(modelPrediction):
            classFact = tC[i][j]
            if classFact != classPrediction:
                # false positive для найденного класса и false negative для упущенного
                f1Errors[classPrediction]['FP'] += 1
                f1Errors[classFact]['FN'] += 1
            else:
                # true positive для найденного класса
                f1Errors[classPrediction]['TP'] += 1
            # true negative для остальных
            for c in classes:
                if c == classFact or c == classPrediction:
                    continue
                f1Errors[c]['TN'] += 1

        f1Scores[model] = {c: 2 * f1Errors[c]['TP'] / (2 * f1Errors[c]['TP'] + f1Errors[c]['FN'] + f1Errors[c]['FP'])
                           for c
                           in f1Errors}
        f1Scores[model]['sum'] = sum(f1Scores[model].values())
        f1Scores[model]['avg'] = f1Scores[model]['sum'] / len(classes)
    FR.append(f1Scores)

result = FR

print(result)

