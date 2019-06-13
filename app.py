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

trainMessages, testMessages, trainClasses, testClasses = model_selection.train_test_split(inputFixedMessages,
                                                                  inputClasses, test_size=0.2, stratify=inputClasses)

textsHash = hashlib.md5(''.join(trainMessages).encode("utf-8")).hexdigest()
fe = featuresExtractor()
if fe.isTrainingRequired(textsHash):
    fe.trainNGrams(trainMessages, trainClasses)

trainFeatures = caching.readVar("featuresVectors for "+textsHash)
if trainFeatures is None:
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
c.trainModels(trainFeatures, trainClasses)

for index, message in enumerate(messages):
    cleanMessage = textPreps.prepareText(message)
    featuresVector = fe.getFeaturesVector(cleanMessage)
    features.append(featuresVector)

# messageClasses = c.predictModel(c.models.SVM, features)
messageClasses = c.predictModels(features)

q=1