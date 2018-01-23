#!/usr/bin/python

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

def scoring(estimator, features_test, labels_test):
     labels_pred = estimator.predict(features_test)
     p = metrics.precision_score(labels_test, labels_pred, average='micro')
     r = metrics.recall_score(labels_test, labels_pred, average='micro')
     if p > 0.3 and r > 0.3:
            return metrics.f1_score(labels_test, labels_pred, average='macro')
     return 0


def computeFraction( poi_messages, all_messages ):
    """ given a number messages to/from POI (numerator)
        and number of all messages to/from a person (denominator),
        return the fraction of messages to/from that person
        that are from/to a POI
   """

    if all_messages == 'NaNNaN': # occurred when created additive features (all emails)
        all_messages = 'NaN'
    if poi_messages == 'NaNNaN':
        poi_messages = 'NaN'
    if all_messages == 'NaN':
        return 0
    if poi_messages == 'NaN':
        return 0
    if all_messages == 0:
        return 0
    return 1.*poi_messages/all_messages
    return fraction
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

#features_list = ['poi','salary'] # You will need to use more features
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

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

df = pd.DataFrame.from_records(list(data_dict.values()))
df.replace(to_replace='NaN',value=np.nan, inplace=True )
nan_features = []
features = df.axes[1].tolist()

for index,i in enumerate(df.isnull().sum()):
    if(i >= 75):
        nan_features.append(features[index])


for feature in features_list:
    if(feature in nan_features):
        features_list.remove(feature)

# email and persons are useless feaures
features_list.remove('email_address')
features_list.remove('persons')

### Task 2: Remove outliers
data_dict.pop('TOTAL',0) #total metric, not a person

data_dict.pop('LOCKHART EUGENE E',0) # all features are NaN, dropping

data_dict.pop('THE TRAVEL AGENCY IN THE PARK',0) # again all feautes are NaNNaN


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

for iter in my_dataset:
    exercised_stock_options = my_dataset[iter]['exercised_stock_options']
    total_stock_value = my_dataset[iter]['total_stock_value']

    my_dataset[iter]['eso/tsv'] = computeFraction(exercised_stock_options,total_stock_value)
    features_list.append('eso/tsv')

    from_poi_to_this_person = my_dataset[iter]['from_poi_to_this_person']
    to_messages = my_dataset[iter]['to_messages']

    my_dataset[iter]['from_poi/to_msg'] = computeFraction(from_poi_to_this_person,to_messages)
    features_list.append('from_poi/to_msg')
    from_this_person_to_poi = my_dataset[iter]['from_this_person_to_poi']
    from_messages = my_dataset[iter]['from_messages']

    my_dataset[iter]['to_poi/from_msg'] = computeFraction(from_poi_to_this_person,to_messages)
    features_list.append('to_poi/from_msg')

features_list.remove('exercised_stock_options')
features_list.remove('total_stock_value')

features_list.remove('from_poi_to_this_person')
features_list.remove('to_messages')

features_list.remove('from_this_person_to_poi')
features_list.remove('from_messages')


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
clf_gnb = GaussianNB()
scored_gnb = cross_validation.cross_val_score(clf_gnb, features, labels)
predicted_gnb = cross_val_predict(clf_gnb, features,labels, cv=10)
precision_score_gnb = metrics.precision_score(labels, predicted_gnb)

clf_svc = SVC()
scored_svc = cross_validation.cross_val_score(clf_svc, features, labels)
predicted_svc = cross_val_predict(clf_svc, features,labels, cv=10)
precision_score_svc = metrics.precision_score(labels, predicted_svc)

clf_rfc = RandomForestClassifier(n_estimators=10,)
scored_rfc = cross_validation.cross_val_score(clf_rfc, features, labels)
predicted_rfc = cross_val_predict(clf_rfc, features,labels, cv=10)
precision_score_rfc = metrics.precision_score(labels, predicted_rfc)

clf_abc = AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,learning_rate=1.0, n_estimators=50, random_state=None)
scored_abc = cross_validation.cross_val_score(clf_abc, features, labels)
predicted_abc = cross_val_predict(clf_abc, features,labels, cv=10)
precision_score_abc = metrics.precision_score(labels, predicted_abc)

clf_bag = BaggingClassifier(base_estimator=None, bootstrap=True,
         bootstrap_features=False, max_features=1.0, max_samples=1.0,
         n_estimators=10, n_jobs=1, oob_score=False, random_state=None,
         verbose=0, warm_start=False)
scored_bag = cross_validation.cross_val_score(clf_bag, features, labels)
predicted_bag = cross_val_predict(clf_bag, features,labels, cv=10)
precision_score_bag = metrics.precision_score(labels, predicted_bag)


### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Tuning both randomFOrest and ADABoost

cv = cross_validation.StratifiedShuffleSplit(labels, n_iter=10)



num_iteration = [100,300,500,850,1000,2000,5000]
learning = [0.01, 0.1, 0.5, 1.0, 1.5, 2.0]

clf_adb = AdaBoostClassifier(DecisionTreeClassifier(max_depth=8))

parameters_adaboost = {'n_estimators':num_iteration,'learning_rate':learning,'algorithm':['SAMME.R','SAMME']}
adaclf = GridSearchCV(clf_adb, parameters_adaboost,scoring = scoring, cv = cv)
adaclf.fit(features, labels)
ada_best_clf = adaclf.best_estimator_
test_classifier(ada_best_clf, my_dataset, features_list)


parameters = {'max_depth': [2,3,4,5,6],'min_samples_split':[2,3,4,5], 'n_estimators':[10,20,50], 'min_samples_leaf':[1,2,3,4], 'criterion':('gini', 'entropy')}
clf_rfc = RandomForestClassifier()
rfclf = GridSearchCV(clf_rfc, parameters,scoring = scoring,cv = cv)
rfclf.fit(features, labels)
rf_best_clf = rfclf.best_estimator_
test_classifier(rf_best_clf, my_dataset, features_list)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.



dump_classifier_and_data(ada_best_clf, my_dataset, features_list)
