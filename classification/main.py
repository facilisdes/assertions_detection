from sklearn.externals import joblib
from sklearn import naive_bayes
from sklearn import svm
from sklearn import linear_model
from sklearn import tree

class classification:

    def __init__(self):

        # https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html
        self.nbc = naive_bayes.MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)

        #https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
        self.svmc = svm.LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=1.0, multi_class='ovr',
                                  fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0,
                                  random_state=None, max_iter=1000)

        #https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
        self.lrc = linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True,
                                                   intercept_scaling=1, class_weight=None, random_state=None,
                                                   solver='warn', max_iter=100, multi_class='warn', verbose=0,
                                                   warm_start=False, n_jobs=None, l1_ratio=None)

        #https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        self.dtc = tree.DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=2,
                                              min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None,
                                              random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0,
                                              min_impurity_split=None, class_weight=None, presort=False)


    def trainSVM(self, features, classes):
        self.svmc.fit(features, classes)

    def predictSVM(self, features):
        result = self.svmc.predict(features)
        return result

    def trainLogReg(self, features, classes):
        self.lrc.fit(features, classes)

    def predictLogReg(self, features):
        result = self.lrc.predict(features)
        return result

    def trainDecisionTree(self, features, classes):
        self.dtc.fit(features, classes)

    def predictDecisionTree(self, features):
        result = self.dtc.predict(features)
        return result

    def trainNaiveBayes(self, features, classes):
        self.nbc.fit(features, classes)

    def predictNaiveBayes(self, features):
        result = self.nbc.predict(features)
        return result