from __future__ import print_function
'''POI Script to generate create and dump classifier
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import pickle
import random
import math
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn import cross_validation
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data

def remove_outlier(df):

    df = df[df.person != 'TOTAL']
    df = df[df.person != 'LOCKHART EUGENE E']
    df = df[df.person != 'THE TRAVEL AGENCY IN THE PARK']
    return df

def feature_selection(df):
    #starting with all the features
    features_list = ['poi',
     'deferral_payments',
     'deferred_income',
     'director_fees',
     'email_address',
     'exercised_stock_options',
     'expenses',
     'from_messages',
     'from_poi_to_this_person',
     'from_this_person_to_poi',
     'loan_advances',
     'long_term_incentive',
     'other',
     'bonus',
     'restricted_stock',
     'restricted_stock_deferred',
     'salary',
     'shared_receipt_with_poi',
     'to_messages',
     'total_payments',
     'total_stock_value',
     'persons']
     # since we have read from pkl to a dataframe the NaNs have come as Strings, replacing them with numpy NaNs fo easy identification
    df.replace(to_replace='NaN',value=np.nan, inplace=True )

    nans = []
    #Collecting list of features
    features = df.axes[1].tolist()
    #if a feature has more than 75 NaNs, it wont make much impact on final model performance, hence collecting and dropping those
    for i,k in enumerate(df.isnull().sum()):
        if k > 75:
            nans.append(features[i])

    for c, s in df.iteritems():
        if(c in nans):
            df.drop(c, axis=1, inplace=True)
    #email address and person are primarily identifier features which won't be useful in model building and classidication and might cause issue as they are string values
    df.drop('email_address', axis=1, inplace=True)
    df.drop('person', axis=1, inplace=True)
    return df


def feature_creation(df):
    #creaating 3 set of new features
    df['eso/tsv'] = df['exercised_stock_options'] / df['total_stock_value']
    df['from_poi/to_msg'] = df['from_poi_to_this_person']/df['to_messages']
    df['to_poi/from_msg'] = df['from_this_person_to_poi']/df['from_messages']
    #droppping the features used for above feature creation
    to_drop = ['exercised_stock_options', 'total_stock_value', 'from_messages', 'from_poi_to_this_person','from_this_person_to_poi','to_messages']
    df.drop(to_drop, axis=1, inplace=True)
    return df


def score_classifier(clf,features,labels):
    score = cross_validation.cross_val_score(clf, features, labels)
    predicted = cross_val_predict(clf, features,labels, cv=10)
    precision = metrics.precision_score(labels, predicted)
    return score,precision
def main():
    # reading and converting the pkl to DataFrame
    with open ('final_project_dataset.pkl','rb') as f:
        enron_data = pickle.load(f)
    df = pd.DataFrame.from_records(list(enron_data.values()))
    persons = pd.Series(list(enron_data.keys()))
    df['person'] = persons
    # Step I removing outlier
    df = remove_outlier(df)
    #Step II Select the features to be used
    df = feature_selection(df)
    # Let's have a benchmark for before feature creation using random forest


    # Step III Feature Creation

    df = feature_creation(df)
    model_labels = df['poi'].copy(deep=True).astype(int).as_matrix()
    model_features = (df.drop('poi', axis=1)).fillna(0).copy(deep=True).as_matrix()
    shuffle = StratifiedShuffleSplit(model_labels, 4, test_size=0.3, random_state=0)
    # Step IV trying a varity of classifier
    #RandomForestClassifier
    clf_rfc = RandomForestClassifier(n_estimators=10,)
    scored_rfc,precision_rfc = score_classifier(clf_rfc,model_features,model_labels)
    print('Random Forest Classifier')
    print('Scores')
    print(scored_rfc)
    print('Precision')
    print(precision_rfc)

    #GaussianNaiveBayes
    clf_gnb = GaussianNB()
    scored_gnb,precision_gnb = score_classifier(clf_gnb,model_features,model_labels)
    print('GaussianNaiveBayes Classifier')
    print('Scores')
    print(scored_gnb)
    print('Precision')
    print(precision_gnb)
    #SVC
    clf_svc = SVC()
    scored_svc,precision_svc = score_classifier(clf_svc,model_features,model_labels)
    print('SVM Classifier')
    print('Scores')
    print(scored_svc)
    print('Precision')
    print(precision_svc)
    #AdaBoost
    clf_abc = AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,learning_rate=1.0, n_estimators=50, random_state=None)
    scored_abc,precision_abc = score_classifier(clf_abc,model_features,model_labels)
    print('AdaBoost Classifier')
    print('Scores')
    print(scored_abc)
    print('Precision')
    print(precision_abc)
    #Bagging
    clf_bag = BaggingClassifier(base_estimator=None, bootstrap=True,
         bootstrap_features=False, max_features=1.0, max_samples=1.0,
         n_estimators=10, n_jobs=1, oob_score=False, random_state=None,
         verbose=0, warm_start=False)
    scored_bag,precision_bag = score_classifier(clf_bag,model_features,model_labels)
    print('AdaBoost Classifier')
    print('Scores')
    print(scored_bag)
    print('Precision')
    print(precision_bag)


    # Step V Paramter Tuning.
    # Tuning Both RandomForest and AdaBoost Classifier using GridSearchCV

    cv = cross_validation.StratifiedShuffleSplit(model_labels, n_iter=10)

    num_iteration = [100,300,500,850,1000,2000,5000]
    learning = [0.01, 0.1, 0.6, 1.0, 1.5, 2.0]

    clf_adb = AdaBoostClassifier(DecisionTreeClassifier(max_depth=8))

    parameters_adaboost = {'n_estimators':num_iteration,'learning_rate':learning,'algorithm':['SAMME.R','SAMME']}
    adaclf = GridSearchCV(clf_adb, parameters_adaboost, cv = cv)
    adaclf.fit(model_features, model_labels)

    parameters = {'max_depth': [2,3,4,5,6],'min_samples_split':[2,3,4,5], 'n_estimators':[10,20,50], 'min_samples_leaf':[1,2,3,4], 'criterion':('gini', 'entropy')}
    clf_rfc = RandomForestClassifier()
    rfclf = GridSearchCV(clf_rfc, parameters, cv = cv)
    rfclf.fit(model_features, model_labels)


    list_cols = list(df.columns.values)
    list_cols.remove('poi')
    list_cols.insert(0, 'poi')#poi has to be first feature
    data = df[list_cols].fillna(0).to_dict(orient='records')
    enron_data_sub = {}
    counter = 0
    for item in data:
        enron_data_sub[counter] = item
        counter += 1
    ada_best_clf = adaclf.best_estimator_
    test_classifier(ada_best_clf, enron_data_sub, list_cols)

    rf_best_clf = rfclf.best_estimator_
    test_classifier(rf_best_clf, enron_data_sub, list_cols)

    dump_classifier_and_data(ada_best_clf, enron_data_sub, list_cols)

if __name__ == '__main__':
    main()
