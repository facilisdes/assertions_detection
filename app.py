from common import textPreps
from sklearn import svm
import os
import csv
import hashlib

from features.main import featuresExtractor
import common.twitter as tw

tw.search("beyonce")

# сообщения обучающей выборки
curDir = os.path.dirname(__file__)
trainFile = os.path.join(curDir, "data/trainMessages.csv")

if not os.path.isfile(trainFile):
    raise Exception("Train messages file (%s) does not exists" % trainFile)
with open(trainFile, 'r', encoding='utf-8-sig') as f:
    trainsList = list(csv.reader(f, delimiter=";"))

trainMessages = [train[0] for train in trainsList]
# они же, но очищенные
trainFixedMessages = [textPreps.prepareText(trainMessage) for trainMessage in trainMessages]
# значения их признаков
trainFeatures = []
# их классы
trainClasses = [train[1] for train in trainsList]
# анализируемые сообщения
messages = [
    """RT:Процесс обучения (с учителем): реализации алгоритма метода \"опорных векторов\" подается на вход обучающая вы
борка, полученная на этапе 0 (т.е. набор сообщений и меток из классов, полученных методом экспертных оценок), в 
результате он определяет оптимальную разделяющую гиперплоскость в пространстве определенных в 1.1 признаков. Реализация
используется из библиотеки scikit-learn.""",
    """Специфичные для сервиса Twitter символы @ (@, #, RT - упоминание, хэштег, ретвит). В работе [4] отмечено, что 
семантика данных символов зависит от их положения в сообщении; положение в начале сообщения более прогностическое, 
нежели в любом другом месте. В связи с этим выделено три бинарных признака для определения факта присутствия каждого 
символа в сообщении в целом и три бинарных признака для определения присутствия символа в начале сообщения;""",
    """Сокращения. На основании словаря из работы [10] и онлайн-словарей создан список из 26 сокращений, часто 
используемых в интернет-среде. На его основании определены 26 бинарных признаков для определения факта присутствия 
каждого сокращения в сообщении;""",
    """Синтаксические деревья. По аналогии с n-граммами из предыдущей группы признаков, здесь набор признаков строится 
автоматически на основании размеченных данных. Сперва строятся синтаксические деревья всех сообщений обучающей 
выборки, их них извлекаются все поддеревья не длиннее трёх слов и не глубже двух уровней. Затем отфильтровываются 
деревья, встречающиеся в менее чем 5% сообщений. Затем отфильтровываются деревья, содержащие в себе имена 
собственные.""",
    """После по аналогии с n-грамма ми рассчитывается нормализованная энтропия распределения дерева. Отфильтровываются 
деревья, для которых это значение выше 0,3. Полученный список деревьев составляет в среднем 30% всех изначально 
выявленных деревьев; деревья этого списка встречаются в целом в 38% сообщений.  Количество признаков для классификатора 
равняется количеству полученных деревьев, все признаки бинарные и отображают факт наличия дерева в сообщении;""",
    """Части речи. В сообщении на данном этапе определяется наличие прилагательных и междометий. Междометия #зачастую 
используются для выражения эмоций и факт их наличия в сообщении может говорить о том, что сообщение выражает мнение. 
Прилагательные в свою очередь аналогично зачастую используются для выражения мнения или рекомендаций. Выделено два 
бинарных признака, отображающих наличие данных частей речи в сообщении"""
        ]
# значения их признаков
features = []
# их классы
classes = []

lin_clf = svm.LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True, intercept_scaling=1,
                        loss='squared_hinge', max_iter=1000, multi_class='ovr', penalty='l2', random_state=None,
                        tol=0.0001, verbose=0)

textsHash = hashlib.md5(''.join(trainFixedMessages).encode("utf-8")).hexdigest()
fe = featuresExtractor()
if fe.isTrainingRequired(textsHash):
    fe.trainNGrams(trainFixedMessages, trainClasses)
for message in trainFixedMessages:
    trainFeatures.append(fe.getFeaturesVector(message))

lin_clf.fit(trainFeatures, trainClasses)

for index, message in enumerate(messages):
    cleanMessage = textPreps.prepareText(message)
    featuresVector = fe.getFeaturesVector(cleanMessage)
    features.append(featuresVector)
    dec = lin_clf.decision_function(featuresVector)
    messageClass = dec.shape[1]
    classes.append(messageClass)

