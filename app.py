import csv
import hashlib
import os

from sklearn import model_selection

from classification.main import classifiers
from common import caching
from common import textPreps
from features.main import featuresExtractor

def getSplits(X, Y, doSplits = False, randomState = 123456):
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
    True  # сбрасывать ли кеш
]
ID = "dataset 1"
if MODE[0] == False:
    ID = "dataset 2"

ID+= ' with params ' + ','.join(map(str,MODE))

caching.saveVar("prediction for test on " + ID, None)

if MODE[3] == True:
    #caching.saveVar("featuresVectors for " + ID, None)
    #caching.saveVar("featuresVectors for test on " + ID, None)
    caching.saveVar("prediction for test on " + ID, None)
    #caching.saveVar('ngrams for ' + ID, None)

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

trainMessages, testMessages, trainClasses, testClasses = getSplits(inputFixedMessages, inputClasses, MODE[0])

fe = featuresExtractor()
if fe.isTrainingRequired('ngrams for ' + ID):
    fe.trainNGrams(trainMessages, trainClasses, cache=True, textsHash = 'ngrams for ' + ID)

trainFeatures = caching.readVar("featuresVectors for " + ID)
if trainFeatures is None:
    trainFeatures = []
    for message in trainMessages:
        featuresVector = fe.getFeaturesVector(message)
        trainFeatures.append(featuresVector)
    caching.saveVar("featuresVectors for " + ID, trainFeatures)

# анализируемые сообщения
messages = []
# значения их признаков
features = []
# их классы
classes = []

c = classifiers()

if MODE[1]:
    #svm = c.findBestParamsForSVM(trainFeatures, trainClasses)
    #NB = c.findBestParamsForNaiveBayes(trainFeatures, trainClasses)
    LogReg = c.findBestParamsForLogReg(trainFeatures, trainClasses)
    #DecTree = c.findBestParamsForDecTree(trainFeatures, trainClasses)
    result = {'svm': svm, 'NB': NB, 'LogReg': LogReg, 'DecTree': DecTree}
else:
    prediction = caching.readVar("prediction for test on " + ID)
    if prediction is None:
        if MODE[2]:
            c.trainModels(trainFeatures, trainClasses)
        else:
            c.trainModel(c.models.SVM, trainFeatures, trainClasses)

        testFeatures = caching.readVar("featuresVectors for test on " + ID)
        if testFeatures is None:
            testFeatures = []
            for message in testMessages:
                featuresVector = fe.getFeaturesVector(message)
                testFeatures.append(featuresVector)
            caching.saveVar("featuresVectors for test on " + ID, testFeatures)

        if MODE[2]:
            prediction = c.predictModels(testFeatures)
        else:
            prediction = c.predictModel(c.models.SVM, testFeatures)

        caching.saveVar("prediction for test on " + ID, prediction)

    if MODE[2]:
        classes = set(testClasses)
        classes = list(map(str, classes))
        f1Errors = {c: {'FP': 0, 'FN': 0, 'TP': 0, 'TN': 0} for c in classes}
        f1Scores = {}
        for model in prediction:
            modelPrediction = prediction[model]
            for i, classPrediction in enumerate(modelPrediction):
                classFact = testClasses[i]
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

            f1Scores[model] = {c: 2 * f1Errors[c]['TP'] / (2 * f1Errors[c]['TP'] + f1Errors[c]['FN'] + f1Errors[c]['FP']) for c
                    in f1Errors}
            f1Scores[model]['sum'] = sum(f1Scores[model].values())
            f1Scores[model]['avg'] = f1Scores[model]['sum'] / len(classes)
    else:
        classes = set(testClasses)
        classes = list(map(str, classes))
        f1Errors = {c: {'FP': 0, 'FN': 0, 'TP': 0, 'TN': 0} for c in classes}
        f1Scores = {}

        for i, classPrediction in enumerate(prediction):
            classFact = testClasses[i]
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

        f1Scores = {c: 2 * f1Errors[c]['TP'] / (2 * f1Errors[c]['TP'] + f1Errors[c]['FN'] + f1Errors[c]['FP'])
                           for c
                           in f1Errors}
        f1Scores['sum'] = sum(f1Scores.values())
        f1Scores['avg'] = f1Scores['sum'] / len(classes)

    result = [prediction, f1Scores]

print(result)

