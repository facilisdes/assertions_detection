import csv
import os

from sklearn import model_selection

from classification.main import classifiers
from clusterization.main import clusterizer
from common import caching
from common import textPreps
from features.main import featuresExtractor

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
import random
import copy


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


def humanReadableOutput(result):
    predictions = result['prediction']
    scores = result['scores']
    clusters = result['clusters']

    print("Этап 1. Вывод предсказаний для сообщений.")
    for i, prediction in enumerate(predictions):
        print('Текст: ' + testMessages[i])
        print('Предсказание/Факт: ' + prediction + "/" + testClasses[i])
        print("\n")

    print("F1-меры измерений по каждому классу и в целом:")
    for textClass in [1, 2, 3, 4]:
        textClass = str(textClass)
        print("Класс " + textClass + ", оценка " + str(round(scores[textClass], 2)))
    print("Среднее  " + str(round(scores['avg'], 2)))
    print("\n\n")
    print("*"*120)

    print("Этап 2. Вывод групп отзывов с выражением мнения.")
    iter = 1
    for clusterGroup in clusters:
        print("Группа " + str(iter))
        print("Тексты:")
        print(*clusters[clusterGroup]['messages'], sep='\n')
        print("Слова, описывающие группу:")
        print(*clusters[clusterGroup]['leadWords'], sep='\n')
        print("Слова мнения, опиcывающие группу:")
        print(*clusters[clusterGroup]['leadOpinionWords'], sep='\n')
        print("Рейтинг эмоциональности группы: " + str(clusters[clusterGroup]['opinionScore']))
        print("\n")
        iter+=1


# общие параметры работы программы
MODE = [
    True,   # разделять ли выборку на обучающую и тестовую
    False,  # делать ли вместо обучения поиск гиперпараметров
    False,    # делать предсказание по всем моделям вместо лучшей
    True,  # сбрасывать ли кеш
    True # группировать ли отзывы
]
ID = "dataset 1"
if MODE[0] == False:
    ID = "dataset 2"

ID += ' with params ' + ','.join(map(str,MODE[:4]))

caching.saveVar("prediction for test on " + ID, None)

if MODE[3] == True:
    q = 1
    #caching.saveVar("featuresVectors for " + ID, None)
    #caching.saveVar("featuresVectors for test on " + ID, None)
    #caching.saveVar("prediction for test on " + ID, None)
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
    svm = c.findBestParamsForSVM(trainFeatures, trainClasses)
    NB = c.findBestParamsForNaiveBayes(trainFeatures, trainClasses)
    LogReg = c.findBestParamsForLogReg(trainFeatures, trainClasses)
    DecTree = c.findBestParamsForDecTree(trainFeatures, trainClasses)
    result = {'svm': svm, 'NB': NB, 'LogReg': LogReg, 'DecTree': DecTree}
else:
    prediction = caching.readVar("prediction for test on " + ID)
    if prediction is None:
        testFeatures = caching.readVar("featuresVectors for test on " + ID)
        if testFeatures is None:
            testFeatures = []
            for message in testMessages:
                featuresVector = fe.getFeaturesVector(message)
                testFeatures.append(featuresVector)
            caching.saveVar("featuresVectors for test on " + ID, testFeatures)

        if MODE[2]:
            c.trainModels(trainFeatures, trainClasses)
        else:
            y_score = c.models.SVM.fit(trainFeatures, trainClasses).decision_function(trainFeatures)
            #c.trainModel(c.models.LogReg, trainFeatures, trainClasses)


        if MODE[2]:
            prediction = c.predictModels(testFeatures)
        else:
            prediction = c.predictModel(c.models.SVM, testFeatures)

        caching.saveVar("prediction for test on " + ID, prediction)

    if MODE[2]:
        classes = set(testClasses)
        classes = list(map(str, classes))
        testClassesCounts = {c: testClasses.count(c) for c in classes}

        f1Scores = {}
        for model in prediction:
            f1Errors = {c: {'FP': 0, 'FN': 0, 'TP': 0, 'TN': 0} for c in classes}
            modelPrediction = prediction[model]
            for i, classPrediction in enumerate(modelPrediction):
                classFact = testClasses[i]
                if classFact != classPrediction:
                    # false pos itive для найденного класса и false negative для упущенного
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

            for c in classes:
                classCount = testClassesCounts[c]
                f1Errors[c]['prec'] = (classCount-f1Errors[c]['FP']) / classCount
                f1Errors[c]['recall'] = (classCount-f1Errors[c]['FN']) / classCount

            f1Scores[model] = {c: 2 * f1Errors[c]['TP'] / (2 * f1Errors[c]['TP'] + f1Errors[c]['FN'] +
                                                           f1Errors[c]['FP']) for c in f1Errors}
            f1Scores[model]['sum'] = sum(f1Scores[model].values())
            f1Scores[model]['avg'] = f1Scores[model]['sum'] / len(classes)

            f1Scores[model]['prec'] = {p: f1Errors[p]['prec'] for p in f1Errors}
            f1Scores[model]['recall'] = {p: f1Errors[p]['recall'] for p in f1Errors}
    else:
        classes = set(testClasses)
        classes = list(map(str, classes))
        testClassesCounts = {c: testClasses.count(c) for c in classes}
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
        for c in classes:
            classCount = testClassesCounts[c]
            f1Errors[c]['prec'] = (classCount-f1Errors[c]['FP']) / classCount
            f1Errors[c]['recall'] = (classCount-f1Errors[c]['FN']) / classCount

        f1Scores = {c: 2 * f1Errors[c]['TP'] / (2 * f1Errors[c]['TP'] + f1Errors[c]['FN'] + f1Errors[c]['FP'])
                           for c
                           in f1Errors}
        f1Scores['sum'] = sum(f1Scores.values())
        f1Scores['avg'] = f1Scores['sum'] / len(classes)

        f1Scores['prec'] = {p: f1Errors[p]['prec'] for p in f1Errors}
        f1Scores['recall'] = {p: f1Errors[p]['recall'] for p in f1Errors}

    result = {'prediction': prediction, 'scores': f1Scores}

if MODE[4]:
    statementMessages = []
    for i, classIndex in enumerate(prediction):
        if classIndex == '3':
            statementMessages.append(testMessages[i])

    cl = clusterizer()
    clusters = cl.clusterize(statementMessages, groupByOpinionWords=True, cacheHash="clusterizing for " + ID,
                             resetCache=False)
    result['clusters'] = clusters

    humanReadableOutput(result)


else:
    print(result)
