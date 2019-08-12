import pickle
import pandas as pd
import numpy as np
import copy

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from score import report_score
from sklearn.linear_model import LogisticRegression
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB

#_____________________
##from pandas.plotting import scatter_matrix
##import matplotlib.pyplot as plt
##from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
##from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
##from sklearn.preprocessing import normalize
##from sklearn.metrics.pairwise import cosine_similarity
#_____________________

TRAIN_H5 = "prs_trn_2.h5"
F_H5 = "prs_comp_tst.h5"
F_PKL = 'test_feature_new.pkl'

f = open('feature_new.pkl', 'rb')
trainX_all = pickle.load(f)
f.close()
combined_dataframe = pd.read_hdf(TRAIN_H5)
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
# trainX and trainY is for triple classification


#method1: 2 classification for all 4 class
def method1():
    alltrainX = copy.deepcopy(trainX_all)
    alltrainY = list((combined_dataframe.values[:,4]).astype('int64'))
    xg1 = XGBClassifier(learning_rate=0.1, n_estimators=1000, max_depth=6,eta = 0.1,\
                             objective="binary:logistic", subsample=0.9, colsample_bytree=0.8)
    xg1.fit(np.array(alltrainX),np.array(alltrainY))
    #relatedness_classifier = train_relatedness_classifier(trainX, trainY)

    relatedTrainX = trainX
    related_frame = combined_dataframe[combined_dataframe['first']==0].copy()
    related_frame["discussY"] = related_frame['Stance'].apply({'discuss':1,'agree':0,'disagree':0}.get)
    relatedTrainY = list((related_frame.values[:,6]).astype('int64'))
    xg2 = XGBClassifier(learning_rate=0.1, n_estimators=1000, max_depth=6,eta = 0.1,\
                             objective="binary:logistic", subsample=0.9, colsample_bytree=0.8)
    xg2.fit(np.array(relatedTrainX), np.array(relatedTrainY))

    agree = combined_dataframe[combined_dataframe['second']>1].index.tolist()
    agreeTrainX = []
    for i in agree:
        agreeTrainX.append(trainX_all[i])
    agree_frame = combined_dataframe[combined_dataframe['second']>1].copy()
    agree_frame["agreeY"] = agree_frame['Stance'].apply({'agree':1,'disagree':0}.get)
    agreeTrainY = (agree_frame.values[:,6]).astype('int64')
    lr= LogisticRegression(C = 1.0,penalty = 'l2',solver = 'lbfgs')
    lr.fit(agreeTrainX, agreeTrainY)
    return xg1,xg2,lr
#method2:  2 classification + 3 classification
def binary():
    lr= LogisticRegression(C = 1.0,penalty = 'l2',solver = 'lbfgs')
    metric_all = pd.DataFrame()
    metric = cross_val_score(lr, trainX_all,trainy_all, cv=10, scoring='f1',verbose = 1)
    #metric.sort()
    metric_all['LogisticRegression'] = metric[::-1]


    X_train1, X_test1, y_train1, y_test1 = train_test_split(trainX_all,trainy_all, test_size = 0.3, random_state = 0)
    svc = SVC(C=1.0, kernel='rbf', gamma='auto')
    metric = cross_val_score(svc, X_test1,y_test1, cv=10, scoring='f1',verbose = 1)
    #[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    #[Parallel(n_jobs=1)]: Done  10 out of  10 | elapsed:  1.8min finished
    #metric.sort()
    metric_all['svm'] = metric[::-1]

    dt = DecisionTreeClassifier(criterion="gini",max_depth=11)
    X_train1, X_test1, y_train1, y_test1 = train_test_split(trainX_all,trainy_all, test_size = 0.3, random_state = 0)
    metric = cross_val_score(dt, X_train1,y_train1, cv=10, scoring='f1',verbose = 1)
    #[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    #[Parallel(n_jobs=1)]: Done  10 out of  10 | elapsed:  2.2min finished
    #metric.sort()
    metric_all['DecisionTree'] = metric[::-1]

    gnb = GaussianNB()
    metric = cross_val_score(gnb, trainX_all,trainy_all, cv=10, scoring='f1',verbose = 1)
    #[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    #[Parallel(n_jobs=1)]: Done  10 out of  10 | elapsed:   24.8s finished
    #metric.sort()
    metric_all['GaussianNB'] = metric[::-1]

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=None, priors=None)
    metric = cross_val_score(lda, trainX_all,trainy_all, cv=10, scoring='f1',verbose = 1)
    #[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    #[Parallel(n_jobs=1)]: Done  10 out of  10 | elapsed:   42.5s finished
    #metric.sort()
    metric_all['LDA'] = metric[::-1]

    from sklearn.ensemble import GradientBoostingClassifier
    gb = GradientBoostingClassifier(n_estimators=500,validation_fraction = 0.2,tol = 0.01,learning_rate=0.1,\
                                     min_samples_split=20,max_features='sqrt',subsample=0.8,random_state=10)
    metric = cross_val_score(lda, trainX_all,trainy_all, cv=10, scoring='f1',verbose = 1)
    #[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    #[Parallel(n_jobs=1)]: Done  10 out of  10 | elapsed:   39.8s finished
    #metric.sort()
    metric_all['GradientBoosting'] = metric[::-1]

    from xgboost.sklearn import XGBClassifier
    xg = XGBClassifier(learning_rate=0.1, n_estimators=1000, max_depth=6,eta = 0.1,\
                             objective="binary:logistic", subsample=0.9, colsample_bytree=0.8)
    metric = cross_val_score(xg,np.array(trainX_all),
                             np.array(trainy_all), cv=10, scoring='f1',verbose = 1)
    metric_all['xgboosting'] = metric[::-1]
    print(metric_all)
    metric_mean = metric_all.mean()
    print(metric_mean.sort_values(ascending=False))
    return metric_all
#————————————————————————————————————————
def triple():
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
    X_train1, X_test1, y_train1, y_test1 = train_test_split(trainX,trainY, test_size = 0.5, random_state = 0)
    metric = cross_val_score(svc, X_train1, y_train1, cv=kfold, verbose = 1,scoring=scoring)
    metric.sort()
    metric_all2['svm'] = metric[::-1]
    gnb = GaussianNB()
    metric = cross_val_score(gnb, trainX, trainY, cv=kfold, verbose = 1,scoring=scoring)
    metric.sort()
    metric_all2['gnb'] = metric[::-1]
    lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=None, priors=None)
    metric = cross_val_score(lda, trainX, trainY, cv=kfold, verbose = 1,scoring=scoring)
    metric.sort()
    metric_all2['LDA'] = metric[::-1]
    RF = RandomForestClassifier(n_estimators=100, criterion='gini', random_state=0)
    metric = cross_val_score(RF, trainX, trainY, cv=kfold, verbose = 1,scoring=scoring)
    metric.sort()
    metric_all2['RF100'] = metric[::-1]
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)
    metric = cross_val_score(RF, trainX, trainY, cv=kfold, verbose = 1,scoring=scoring)
    metric.sort()
    metric_all2['gb'] = metric[::-1]
    print(metric_all2)
    metric_mean = metric_all2.mean()
    print(metric_mean.sort_values(ascending=False))
    return metric_all2

def voting():
    from sklearn.ensemble import VotingClassifier
    dt = DecisionTreeClassifier(criterion="gini",max_depth=11)
    lr= LogisticRegression(C = 1.0,penalty = 'l2',solver = 'lbfgs')
    svc = SVC(C=1.0, kernel='rbf', gamma='auto')
    lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=None, priors=None)
    clf = VotingClassifier(estimators=[('lr', rf_optimized), ('dt', svc_optimized), ('svm', xgb_optimized)],
                        voting='hard')
    scores = cross_val_score(clf,trainX_all,trainy_all, cv=10, scoring='f1',verbose = 1)
    return scores


def bi_percep():
    ppn = Perceptron(eta0=0.1, random_state=0)
    ppn.fit(X_train, y_train)
    y_pred = ppn.predict(X_test)
    acc1 = accuracy_score(y_test, y_pred)
    print(acc1)
    return ppn
    #0.9806563500533618

def _cross_entropy_like_loss(model, input_data, targets, num_estimators):
    loss = np.zeros((num_estimators, 1))
    for index,pred in enumerate(model.staged_predict(input_data)):
        loss[index, :] = 1-accuracy_score(targets,pred)
        #print(f'ce ls {index}:{loss[index, :]}')
    return loss
def plot_err_iter():
    n_estimators = 1000
    X_train, X_val, Y_train, Y_val = train_test_split(trainX,trainY, test_size=0.3, random_state=10)
    clf = GradientBoostingClassifier(n_estimators=800,validation_fraction = 0.2,tol = 0.01,learning_rate=0.1,\
                                     min_samples_split=20,max_features='sqrt',subsample=0.8,random_state=10)
    clf.fit(X_train, Y_train)
    tr_loss_ce = _cross_entropy_like_loss(clf, X_train, Y_train, n_estimators)
    test_loss_ce = _cross_entropy_like_loss(clf, X_val, Y_val, n_estimators)
    plt.figure()
    plt.plot(np.arange(n_estimators) + 1, tr_loss_ce, '-r', label='training_loss_ce')
    plt.plot(np.arange(n_estimators) + 1, test_loss_ce, '-b', label='val_loss_ce')
    plt.ylabel('Error')
    plt.xlabel('Iterations')
    plt.legend(loc='upper right')
    
    plt.show()
    
def train_and_test():
    result = pd.DataFrame()
    f = open(F_PKL, 'rb')
    testX_all = pickle.load(f)
    f.close()
    
    from xgboost.sklearn import XGBClassifier
    xg = XGBClassifier(learning_rate=0.1, n_estimators=1000, max_depth=6,eta = 0.1,\
                             objective="binary:logistic", subsample=0.9, colsample_bytree=0.8)
    xg.fit(np.array(trainX_all), np.array(trainy_all))
    y_pred_binary = xg.predict(np.array(testX_all))
  
    y_pred_binary = list((np.array(y_pred_binary)-1)*(-1))
    result['binary'] = y_pred_binary
    stage2 = result[result['binary']==1].index.tolist()
    testX = []
    for i in stage2:
        testX.append(testX_all[i])
    gb = GradientBoostingClassifier(n_estimators=800,validation_fraction = 0.2,\
                                    tol = 0.01,learning_rate=0.1,min_samples_split=300,max_features='sqrt',subsample=0.8,random_state=10)
    gb.fit(trainX,trainY)
    y_pred = list(gb.predict(testX))
    Stance = {0:'unrelated',1:'discuss',2:'agree',3:'disagree'}
    pred = []
    for i in range(len(y_pred_binary)):
        if y_pred_binary[i] == 0:
            pred.append('unrelated')
        else:
            pred.append(Stance[y_pred.pop(0)])
    dataframe = pd.read_hdf(F_H5)
    actual = list(dataframe['Stance'])
    #actual = list(combined_dataframe['Stance'])
    report_score(actual,pred)
    return pred
pred = train_and_test()
