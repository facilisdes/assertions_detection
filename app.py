import csv
import hashlib
import os

from sklearn import model_selection

from classification.main import classifiers
from common import caching
from common import textPreps
from features.main import featuresExtractor

# 1 для работы с разделением выборки на обучающую и тестовую, 2 для работы со всей выборкой в целом
MODE = [
    True, # разделять ли выборку на обучающую и тестовую
    False # делать ли вместо обучения поиск гиперпараметров
]
ID = "dataset 1"
if(MODE[0] == False):
    ID = "dataset 2"

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

if MODE[0]:
    trainMessages, testMessages, trainClasses, testClasses = model_selection.train_test_split(inputFixedMessages,
                                                                  inputClasses, test_size=0.2, stratify=inputClasses)
else:
    trainMessages = inputFixedMessages
    trainClasses = inputClasses
    testMessages = []
    testClasses = []

fe = featuresExtractor()
if fe.isTrainingRequired(ID):
    fe.trainNGrams(trainMessages, trainClasses, cache=True, textsHash = ID)

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
    aa = c.findBestParamsForSVM(trainFeatures, trainClasses)
    bb = c.findBestParamsForNaiveBayes(trainFeatures, trainClasses)
    cc = c.findBestParamsForLogReg(trainFeatures, trainClasses)
    dd = c.findBestParamsForDecTree(trainFeatures, trainClasses)
    result = {'svm': aa, 'NB': bb, 'LogReg': cc, 'DecTree': dd}
else:
    prediction = caching.readVar("prediction for test on " + ID)
    if prediction is None:
        c.trainModels(trainFeatures, trainClasses)
        testFeatures = caching.readVar("featuresVectors for test on " + ID)
        if testFeatures is None:
            testFeatures = []
            for message in testMessages:
                featuresVector = fe.getFeaturesVector(message)
                testFeatures.append(featuresVector)
            caching.saveVar("featuresVectors for test on " + ID, testFeatures)

        prediction = c.predictModels(testFeatures)

    caching.saveVar("prediction for test on " + ID, prediction)
    result = prediction

print(result)