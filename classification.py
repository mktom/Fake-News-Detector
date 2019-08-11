import pickle
import pandas as pd
import numpy as np
import copy
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler
#_____________________
##from pandas.plotting import scatter_matrix
##import matplotlib.pyplot as plt
##from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
##from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
##from sklearn.preprocessing import normalize
##from sklearn.metrics.pairwise import cosine_similarity
#_____________________
f = open('feature.pkl', 'rb')
trainX_all = pickle.load(f)
f.close()
combined_dataframe = pd.read_hdf("prs_trn_2.h5")
#combined_dataframe = pd.read_hdf("prs_test_2.h5")
print(combined_dataframe, combined_dataframe.info())
combined_dataframe['first']=combined_dataframe['Stance'].apply({'unrelated':1,'discuss':0,'agree':0,'disagree':0}.get)
combined_dataframe['second'] = combined_dataframe['Stance'].apply({'unrelated':0,'discuss':1,'agree':2,'disagree':3}.get)
trainy_all = list((combined_dataframe.values[:,4]).astype('int64'))
stage2 = combined_dataframe[combined_dataframe['first']==0].index.tolist()
stage2_frame = combined_dataframe[combined_dataframe['first']==0]
trainX = []
for i in stage2:
    trainX.append(trainX_all[i])
trainY = list((stage2_frame.values[:,5]).astype('int64'))


# trainX_all and trainy_all is for binary classification
# trainX and trainY is for binary classification


#method1: 2 classification for all 4 class
import xgboost as xgb
def train_relatedness_classifier(trainX, trainY):
    xg_train = xgb.DMatrix(trainX, label=trainY)
    # setup parameters for xgboost
    param = {}
    # use softmax multi-class classification
    param['objective'] = 'binary:logistic'
    # scale weight of positive examples
    param['eta'] = 0.1
    param['max_depth'] = 6
    param['silent'] = 1
    param['nthread'] = 20

    num_round = 1000
    relatedness_classifier = xgb.train(param, xg_train, num_round)
    return relatedness_classifier
def method1():
    trainX = copy.deepcopy(trainX_all)
    trainY = list((combined_dataframe.values[:,4]).astype('int64'))
    relatedness_classifier = train_relatedness_classifier(trainX, trainY)

    relatedTrainX = trainX3
    related_frame = combined_dataframe[combined_dataframe['first']==0]
    related_frame["discussY"] = related_frame['Stance'].apply({'discuss':1,'agree':0,'disagree':0}.get)
    relatedTrainY = list((related_frame.values[:,6]).astype('int64'))
    discuss_classifier = train_relatedness_classifier(relatedTrainX, relatedTrainY)

    agree = combined_dataframe[combined_dataframe['second']>1].index.tolist()
    agreeTrainX = []
    for i in agree:
        agreeTrainX.append(trainX_all[i])
    agree_frame = combined_dataframe[combined_dataframe['second']>1]
    agree_frame["agreeY"] = agree_frame['Stance'].apply({'agree':1,'disagree':0}.get)
    agreeTrainY = (agree_frame.values[:,6]).astype('int64')
    agree_classifier = train_relatedness_classifier(agreeTrainX, agreeTrainY)
    return relatedness_classifier,discuss_classifier
##def prediction1():
##    xg_test = xgb.DMatrix(testX)
##    relatedness_pred = relatedness_classifier.predict(xg_test);
##    discuss_pred = discuss_classifier.predict(xg_test)
##    agree_pred = agree_classifier.predict(xg_test)
##
##    ret, scores = [], []
##    for (pred_relate, pred_discuss, pred_agree) in zip(relatedness_pred, discuss_pred, agree_pred):
##        scores.append((pred_relate, pred_discuss, pred_agree))
##        if pred_relate >= 0.5:
##            ret.append('unrelated')
##        elif pred_discuss >= 0.5:
##            ret.append('discuss')
##        elif pred_agree >= 0.5:
##            ret.append('agree')
##        else:
##            ret.append('disagree')
##    return ret,scores

#method2:  2 classification + 3 classification
def binary():
    lr= LogisticRegression(C = 1.0,penalty = 'l2',solver = 'lbfgs')
    metric_all = pd.DataFrame()
    metric = cross_val_score(lr, trainX_all,trainy_all, cv=10, scoring='f1',verbose = 1)
    metric.sort()
    metric_all['glm'] = metric[::-1]


    X_train1, X_test1, y_train1, y_test1 = train_test_split(trainX_all,trainy_all, test_size = 0.3, random_state = 0)
    svc = SVC(C=1.0, kernel='rbf', gamma='auto')
    metric = cross_val_score(svc, X_test1,y_test1, cv=10, scoring='f1',verbose = 1)
    #[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    #[Parallel(n_jobs=1)]: Done  10 out of  10 | elapsed:  1.8min finished
    metric.sort()
    metric_all['svm'] = metric[::-1]

    dt = DecisionTreeClassifier(criterion="gini",max_depth=11)
    X_train1, X_test1, y_train1, y_test1 = train_test_split(trainX_all,trainy_all, test_size = 0.3, random_state = 0)
    metric = cross_val_score(dt, X_train1,y_train1, cv=10, scoring='f1',verbose = 1)
    #[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    #[Parallel(n_jobs=1)]: Done  10 out of  10 | elapsed:  2.2min finished
    metric.sort()
    metric_all['tree'] = metric[::-1]

    RF = RandomForestClassifier(n_estimators=30, criterion='gini', random_state=0)
    metric = cross_val_score(RF,trainX_all,trainy_all, cv=10, scoring='f1',verbose = 1)
    #[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    #[Parallel(n_jobs=1)]: Done  10 out of  10 | elapsed:  4.6min finished
    metric.sort()
    metric_all['RF30'] = metric[::-1]

    RF = RandomForestClassifier(n_estimators=50, criterion='gini', random_state=0)
    metric = cross_val_score(RF,trainX_all,trainy_all, cv=10, scoring='f1',verbose = 1)
    #[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    #[Parallel(n_jobs=1)]: Done  10 out of  10 | elapsed:  7.7min finished
    metric.sort()
    metric_all['RF50'] = metric[::-1]

    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()
    metric = cross_val_score(gnb, trainX_all,trainy_all, cv=10, scoring='f1',verbose = 1)
    #[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    #[Parallel(n_jobs=1)]: Done  10 out of  10 | elapsed:   24.8s finished
    metric.sort()
    metric_all['gnb'] = metric[::-1]

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=None, priors=None)
    metric = cross_val_score(lda, trainX_all,trainy_all, cv=10, scoring='f1',verbose = 1)
    #[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    #[Parallel(n_jobs=1)]: Done  10 out of  10 | elapsed:   42.5s finished
    metric.sort()
    metric_all['lda'] = metric[::-1]

    from sklearn.ensemble import GradientBoostingClassifier
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)
    metric = cross_val_score(lda, trainX_all,trainy_all, cv=10, scoring='f1',verbose = 1)
    #[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    #[Parallel(n_jobs=1)]: Done  10 out of  10 | elapsed:   39.8s finished
    metric.sort()
    metric_all['gb'] = metric[::-1]
    print(metric_all)
    metric_mean = metric_all.mean()
    print(metric_mean.sort_values(ascending=False))
    return metric_all
#————————————————————————————————————————

metric_all2 = pd.DataFrame()
scoring = 'accuracy'
from sklearn.model_selection import KFold
kfold = KFold(n_splits=10, random_state=0)
lr= LogisticRegression(C = 1.0,penalty = 'l2',solver = 'lbfgs')
metric = cross_val_score(lr, trainX, trainY, cv=kfold, scoring=scoring)
metric.sort()
metric_all2['glm'] = metric[::-1]
dt = DecisionTreeClassifier(criterion="entropy",max_depth=12)
metric = cross_val_score(dt, trainX, trainY, cv=kfold, verbose = 1,scoring=scoring)
metric.sort()
metric_all2['dt'] = metric[::-1]


validation_size = 0.3
seed = 0
X_train1, X_test1, y_train1, y_test1 = train_test_split(trainX_all,trainy_all, \
                                                    test_size = validation_size, random_state = seed)

def bi_percep():
    ppn = Perceptron(eta0=0.1, random_state=0)
    ppn.fit(X_train, y_train)
    y_pred = ppn.predict(X_test)
    acc1 = accuracy_score(y_test, y_pred)
    print(acc1)
    return ppn
    #0.9806563500533618
#Logistic Regression


