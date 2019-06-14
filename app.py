import csv
import hashlib
import os

from sklearn import model_selection

from classification.main import classifiers
from common import caching
from common import textPreps
from features.main import featuresExtractor

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

#trainMessages, testMessages, trainClasses, testClasses = model_selection.train_test_split(inputFixedMessages,
#                                                                  inputClasses, test_size=0.2, stratify=inputClasses)

trainMessages = inputFixedMessages
trainClasses = inputClasses
#textsHash = 'dataset 1'
textsHash = 'dataset 2'
fe = featuresExtractor()
if fe.isTrainingRequired(textsHash):
    fe.trainNGrams(trainMessages, trainClasses, cache=True, textsHash = textsHash)

trainFeatures = caching.readVar("featuresVectors for " + textsHash)
if trainFeatures is None:
    trainFeatures = []
    for message in trainMessages:
        featuresVector = fe.getFeaturesVector(message)
        trainFeatures.append(featuresVector)
    caching.saveVar("featuresVectors for "+textsHash, trainFeatures)


# анализируемые сообщения
messages = []
# значения их признаков
features = []
# их классы
classes = []

c = classifiers()
"""
c.trainModels(trainFeatures, trainClasses)

for index, message in enumerate(messages):
    cleanMessage = textPreps.prepareText(message)
    featuresVector = fe.getFeaturesVector(cleanMessage)
    features.append(featuresVector)
"""
# messageClasses = c.predictModel(c.models.SVM, features)
# messageClasses = c.predictModels(features)
aa = c.findBestParamsForSVM(trainFeatures,trainClasses)
bb = c.findBestParamsForNaiveBayes(trainFeatures,trainClasses)
cc = c.findBestParamsForLogReg(trainFeatures,trainClasses)
dd = c.findBestParamsForDecTree(trainFeatures,trainClasses)

q=1