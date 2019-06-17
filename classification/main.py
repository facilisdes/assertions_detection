from sklearn.externals import joblib
from sklearn import naive_bayes
from sklearn import svm
from sklearn import linear_model
from sklearn import tree
from sklearn import model_selection
from sklearn import metrics
import os

class classifiers:
    class models:
        #https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
        # prefer dual=False when n_samples > n_features.
        SVM = svm.SVC(kernel='linear', gamma='auto', C=1, class_weight=None, tol=0.5, random_state=12345)

        # https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html
        NaiveBayes = naive_bayes.MultinomialNB(alpha=0)

        #https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
        LogReg = linear_model.LogisticRegression(penalty='l1', multi_class='ovr', solver='liblinear', class_weight='balanced', dual=False, C=0.6, tol=0.03, random_state=12345)

        #https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        DecTree = tree.DecisionTreeClassifier(max_depth=10, min_samples_split=0.1, min_samples_leaf=1, max_features=0.9, random_state=12345)


    def __init__(self):


        self.nbc = self.models.NaiveBayes

        self.svmc = self.models.SVM

        self.lrc = self.models.LogReg

        self.dtc = self.models.DecTree


    def trainModel(self, model, features, classes):
        model.fit(features, classes)

    def predictModel(self, model, features):
        result = model.predict(features)
        #acc = accuracy_score(fact, result)
        return result

    def trainSVM(self, features, classes):
        self.svmc.fit(features, classes)

    def predictSVM(self, features):
        result = self.svmc.predict(features)
        # acc = accuracy_score(fact, result)
        return result

    def trainLogReg(self, features, classes):
        self.lrc.fit(features, classes)

    def predictLogReg(self, features):
        result = self.lrc.predict(features)
        # acc = accuracy_score(fact, result)
        return result

    def trainDecisionTree(self, features, classes):
        self.dtc.fit(features, classes)

    def predictDecisionTree(self, features):
        result = self.dtc.predict(features)
        # acc = accuracy_score(fact, result)
        return result

    def trainNaiveBayes(self, features, classes):
        self.nbc.fit(features, classes)

    def predictNaiveBayes(self, features):
        result = self.nbc.predict(features)
        # acc = accuracy_score(fact, result)
        return result

    def trainModels(self, features, classes):
        self.trainSVM(features, classes)
        self.trainLogReg(features, classes)
        self.trainDecisionTree(features, classes)
        self.trainNaiveBayes(features, classes)

    def predictModels(self, features):
        result = dict()
        for modelName in [x for x in dir(self.models) if not x.startswith('__')]:
            model = getattr(self.models, modelName)
            result[modelName] = self.predictModel(model, features)

        return result

    def saveModel(self, model, key):
        joblib.dump(model, "/data/models/%s.pkl" % key)

    def saveModels(self, key):
        for modelName in [x for x in dir(self.models) if not x.startswith('__')]:
            model = getattr(self.models, modelName)
            self.saveModel(model, key + "_" + modelName)

    def loadModel(self, key):
        file = "/data/models/%s.pkl" % key
        if os.path.isfile(file):
            model = joblib.load(file)
            return model
        else:
            return None

    def loadModels(self, key):
        for modelName in [x for x in dir(self.models) if not x.startswith('__')]:
            m = self.loadModel(key + "_" + modelName)
            if m is not None:
                setattr(self.models, modelName, m)

    def findBestParamsForSVM(self, features, classes):
        # 148 variants
        parameters = {
            'kernel': ['poly', 'rbf', 'sigmoid', 'linear'],
            'gamma': ['auto', 'scale',],
            'C': [0, 0.5, 0.75, 1, 1.2],
            'class_weight': ["balanced", None],
            'tol': [0.4, 0.5, 0.6],
        }

        classifier = model_selection.GridSearchCV(self.models.SVM, parameters, cv=5, error_score=0.0)
        classifier.fit(features, classes)
        self.models.SVM = classifier

        bestParams = {
            'kernel': classifier.best_estimator_.kernel,
            'gamma': classifier.best_estimator_.gamma,
            'C': classifier.best_estimator_.C,
            'class_weight': classifier.best_estimator_.class_weight,
            'tol': classifier.best_estimator_.tol
        }

        return bestParams

    def findBestParamsForNaiveBayes(self, features, classes):
        # 11 variants
        parameters = {
            'alpha': [0, 0.25, 0.5, 0.75, 1],
        }

        classifier = model_selection.GridSearchCV(self.models.NaiveBayes, parameters, cv=5, error_score=0.0)
        classifier.fit(features, classes)
        self.models.NaiveBayes = classifier

        bestParams = {
            'alpha': classifier.best_estimator_.alpha,
        }

        return bestParams

    def findBestParamsForLogReg(self, features, classes):
        # 6 variants
        parameters = {
            'penalty': ['l1', 'l2'],
            'solver': ['newton-cg', 'lbfgs', 'liblinear'],
            'class_weight': [None, "balanced"],
            'multi_class': ["ovr", "multinomial"],
            'C': [0.5, 0.6, 0.7],
            'tol': [0.01, 0.03, 0.05],
        }

        classifier = model_selection.GridSearchCV(self.models.LogReg, parameters, cv=5, error_score=0.0)
        classifier.fit(features, classes)
        self.models.LogReg = classifier

        bestParams = {
            'solver': classifier.best_estimator_.solver,
            'penalty': classifier.best_estimator_.penalty,
            'class_weight': classifier.best_estimator_.class_weight,
            'multi_class': classifier.best_estimator_.multi_class,
            'C': classifier.best_estimator_.C,
            'tol': classifier.best_estimator_.tol,
        }

        return bestParams

    def findBestParamsForDecTree(self, features, classes):

        parameters = {
            'max_depth': [None] + list(range(2,15)),
            'min_samples_split': [float(i/10) for i in range(1, 9)] + list(range(1,4)),
            'min_samples_leaf': list(range(1,3)) + [i/100 for i in range(20, 41, 5)],
            'max_features': [None] + [i/10 for i in range(1, 10)],
        }

        classifier = model_selection.GridSearchCV(self.models.DecTree, parameters, cv=5, error_score=0.0)
        classifier.fit(features, classes)
        self.models.DecTree = classifier

        bestParams = {
            'max_depth': classifier.best_estimator_.max_depth,
            'min_samples_split': classifier.best_estimator_.min_samples_split,
            'min_samples_leaf': classifier.best_estimator_.min_samples_leaf,
            'max_features': classifier.best_estimator_.max_features,
        }

        return bestParams

    def evaluateModel(self, model, features, classes, train_size=0.7):
        XT, XF, YT, YF = model_selection.train_test_split(features, classes, train_size)

        kf2 = model_selection.KFold(n_splits=5, shuffle=True, random_state=12345)

        # https: // scikit - learn.org / stable / modules / cross_validation.html  # cross-validation
        # https://chrisalbon.com/machine_learning/model_evaluation/cross_validation_parameter_tuning_grid_search/

        # Разбивает так, что каждый элемент единожды попадает в тестовую выборку, по очереди
        kf1 = model_selection.KFold(n_splits=5, shuffle=False, random_state=12345)

        # Разбивает так, что каждый элемент единожды попадает в тестовую выборку, случайный порядок
        kf2 = model_selection.KFold(n_splits=5, shuffle=True, random_state=12345)

        # Разбивает так, что все тестовые выборки содержат примерно одинаковое количество эл-тов разных классов
        kf3 = model_selection.StratifiedKFold(n_splits=5, shuffle=False, random_state=12345)

        # Разбивает в случайном порядке, элементы могут повторяться
        kf4 = model_selection.ShuffleSplit(n_splits=10, random_state=12345)

        # Разбивает в случайном порядке, элементы могут повторяться, тестовые выборки содержат примерно одинаковое количество эл-тов разных классов
        kf5 = model_selection.StratifiedShuffleSplit(n_splits=10, random_state=12345)

        # делает N тестовых выборок, содержащих поочередно каждый элемент
        kf6 = model_selection.LeaveOneOut()

        self.trainModel(model, XT, YT)
        YP = self.predictModel(model, XF)

        acc = metrics.accuracy_score(YF, YP)
        prec = metrics.precision_score(YF, YP)
        rec = metrics.recall_score(YF, YP)
        f1 = metrics.f1_score(YF, YP)

        return f1, prec, rec, acc
