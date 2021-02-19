# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 08:19:37 2021

@author: adogan
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 13:51:44 2020

@author: adogan
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 17:31:59 2020

@author: adogan
"""

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.utils import resample
from missingpy import MissForest
#loading data
from sklearn.model_selection import train_test_split
data = pd.read_csv(r'C:\Users\adogan\OneDrive - Oklahoma A and M System\Deep Learning - based Point Cloud Analysis\Expert System\Visit2_3_4_working\visit5_FATCHD_work.csv')
#data = pd.read_csv(r'C:\Users\adoga\OneDrive - Oklahoma A and M System\Classified\Codes2\MI\MI_5DC5IC.csv')
Y = data.CLASS
X = data.drop(columns=[ "ID_C",'CLASS'], axis = 1)# 'ID_C',
#imputer = KNNImputer(n_neighbors=10)
imputer = MissForest()
list_col = list(X.columns[:])
for var in (list_col):
    if X[var].isnull().sum() == X.shape[0]:
        X = X.drop(columns = var)
X = pd.DataFrame(imputer.fit_transform(X))

#X = data.drop(columns=['CLASS'], axis = 1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.5,
                                                    random_state=0)

random_state = np.random.RandomState(0)
n_samples, n_features = X.shape
#X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

train_data = X_train.copy()
train_data['CLASS']=Y_train.copy()

'''
Up-sampling the minority class
'''

 #Separate majority and minority classes
df_majority = train_data[train_data.CLASS==0]
df_minority = train_data[train_data.CLASS==1]

 
# Upsample minority class
df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=10097,    # to match majority class
                                 random_state=123) # reproducible results
 
# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])

# Separate input features (X) and target variable (y)
Y_train = df_upsampled.CLASS
X_train = df_upsampled.drop('CLASS', axis=1)




random_state = 22


# #############################################################################
## Classification and ROC analysis
#y = Y_train.copy()
#X = X_train.copy()
#n_samples, n_features = X.shape
#X.index = pd.RangeIndex(start=0, stop=X.shape[0], step=1)
#y.index = pd.RangeIndex(start=0, stop=X.shape[0], step=1)

#from sklearn.model_selection import train_test_split
#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.3,
#                                                    random_state=0)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


rf = RandomForestClassifier(n_estimators=10000, criterion='entropy', max_depth=None, min_samples_split=200, min_samples_leaf=20, 
                            min_weight_fraction_leaf=0.0, max_features=5, max_leaf_nodes=None, 
                            min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, 
                            n_jobs=None, random_state=42, verbose=0, warm_start=True, class_weight=None, 
                            ccp_alpha=0.0, max_samples=12142)
#rf = RandomForestClassifier(n_estimators=50000, max_features=5,min_samples_split=20,min_samples_leaf=20,random_state=0) #n_estimators=10000,max_features=5,min_samples_split=200,min_samples_leaf=20,random_state=42,)#max_features=5, n_estimators=10000,min_samples_split=200,min_samples_leaf=20, random_state=0) #for Stroke, MI
rf.fit(X_train, Y_train)

nb =LogisticRegression(penalty='l2', tol=10, C=1.0, fit_intercept=False, intercept_scaling=0, class_weight=None, random_state=0, solver='lbfgs', max_iter=100000, multi_class='auto', verbose=1,n_jobs=None,l1_ratio=0.5)

#nb =LogisticRegression(solver='lbfgs', tol=1, C=1.0,  max_iter=100000,)
nb.fit(X_train, Y_train)

#mlp = MLPClassifier(hidden_layer_sizes=(32,20,12,8,6,4,2 ), activation='relu', solver='adam', alpha=0.0001, 
#                    batch_size='auto', learning_rate='constant', learning_rate_init=0.001, 
#                    power_t=0.5, max_iter=2000, shuffle=True, random_state=0, tol=0.0001, 
#                    verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, 
#                    early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, 
#                    epsilon=1e-08, n_iter_no_change=10, max_fun=15000)

mlp = MLPClassifier(hidden_layer_sizes=(32,20,12,8,6,4,2 ),max_iter=75000,random_state=42,) #hidden_layer_sizes=(32,20,12,8,6,4,2 ),batch_size=64,max_iter=10000,n_iter_no_change=1)
mlp.fit(X_train, Y_train)


svc = SVC( C=1000.0, kernel='rbf', degree=300, gamma='scale', coef0=0.9, shrinking=True, probability=True, tol=0.001, cache_size=200, 
          class_weight='balanced', verbose=False, max_iter=-1, decision_function_shape='ovo', break_ties=False, random_state=0)
#svc = SVC(C=1000.0,probability=True)
svc.fit(X_train,Y_train)


gb = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.25, max_features= 2, max_depth=2, random_state=0)
gb.fit(X_train, Y_train)

#for Total Outcome
kNN = KNeighborsClassifier(n_neighbors=705, weights='distance', algorithm='ball_tree', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None,)
kNN.fit(X_train,Y_train)


r_probs = [0 for _ in range(len(Y_test))]
rf1 = rf.predict(X_test)
rf_probs_2 = rf.predict_proba(X_test)
nb_probs_2 = nb.predict_proba(X_test)
mlp_probs_2 = mlp.predict_proba(X_test)
svc_probs_2 = svc.predict_proba(X_test)
gb_probs_2 = gb.predict_proba(X_test)
kNN_probs_2 = kNN.predict_proba(X_test)

rf_probs = rf_probs_2[:, 1]
nb_probs = nb_probs_2[:, 1]
mlp_probs = mlp_probs_2[:,1]
svc_probs = svc_probs_2[:, 1]
gb_probs = gb_probs_2[:,1]
kNN_probs = kNN_probs_2[:, 1]





from sklearn.metrics import confusion_matrix
from sklearn import preprocessing

from sklearn.metrics import roc_curve, roc_auc_score

rf_auc = roc_auc_score(Y_test,rf_probs)
nb_auc = roc_auc_score(Y_test, nb_probs)
mlp_auc = roc_auc_score(Y_test, mlp_probs)
svc_auc = roc_auc_score(Y_test, svc_probs)
gb_auc = roc_auc_score(Y_test, gb_probs)
kNN_auc = roc_auc_score(Y_test, kNN_probs)

#print('Random (chance) Prediction: AUROC = %.3f' % (r_auc))
print('Random Forest: AUROC = %.3f' % (rf_auc))
print('Naive Bayes: AUROC = %.3f' % (nb_auc))
print('MLP: AUROC = %.3f' % (mlp_auc))
print('SVC: AUROC = %.3f' % (svc_auc))
print('Gradient Boosting: AUROC = %.3f' % (gb_auc))
print('k Nearest Neighbor: AUROC = %.3f' % (kNN_auc))



r_fpr, r_tpr, _ = roc_curve(Y_test, r_probs)
rf_fpr, rf_tpr, _ = roc_curve(Y_test, rf_probs)
nb_fpr, nb_tpr, _ = roc_curve(Y_test, nb_probs)
mlp_fbr,mlp_tpr, _ = roc_curve(Y_test.astype(int), mlp_probs)
svc_fbr,svc_tpr, _ = roc_curve(Y_test.astype(int), svc_probs)
gb_fbr,gb_tpr, _ = roc_curve(Y_test.astype(int), gb_probs)
kNN_fbr,kNN_tpr, _ = roc_curve(Y_test.astype(int), kNN_probs)

import matplotlib.pyplot as plt
#
#plt.plot(r_fpr, r_tpr, linestyle='--') #, label='Random prediction (AUROC = %0.3f)' % r_auc)
#plt.plot(rf_fpr, rf_tpr, marker='.', label='Random Forest (AUROC = %0.3f)' % rf_auc)
#plt.plot(nb_fpr, nb_tpr, marker='.', label='Logistic Regression (AUROC = %0.3f)' % nb_auc)
#plt.plot(mlp_fbr, mlp_tpr, marker = '.', label = 'MLPClassifier (AUROC = %0.3f)' % mlp_auc)
#plt.plot(svc_fbr, svc_tpr, marker = '.', label = 'Support Vector Classification(AUROC = %0.3f)' % svc_auc)
#plt.plot(gb_fbr, gb_tpr, marker = '.', label = 'Gradient Boosting (AUROC = %0.3f)' % gb_auc)
#plt.plot(kNN_fbr, kNN_tpr, marker = '.', label = 'kNN (AUROC = %0.3f)' % kNN_auc)
#
## Title
#plt.title('ROC Plot - TotalOutcome  ', fontsize = 18) #(New Features) (Discretized)
## Axis labels
#plt.xlabel('False Positive Rate', fontsize = 14)
#plt.ylabel('True Positive Rate', fontsize = 14)
## Show legend
#plt.legend() # 
#plt.figure(figsize=(8, 8))
## Show plot
#plt.show()


#ROC Graph
plt.figure(figsize=(10 , 10))
plt.title('FatCHD' ,  fontweight = "bold" , fontsize=20)
plt.plot([0, 1], [0, 1], linestyle='--', lw=3, color=[0.5,0.5,0.5],alpha=.8)
plt.plot(rf_fpr, rf_tpr,color=[0.99,0.0,0.0], label='Random Forest (AUROC = %0.3f)' % rf_auc,lw=3, alpha=.8)
plt.plot(svc_fbr, svc_tpr,color=[0.0,0.5,0.0], label = 'Support Vector Classification(AUROC = %0.3f)' % svc_auc,lw=3, alpha=.8)
plt.plot(kNN_fbr, kNN_tpr,color=[0.0,0.0,0.7], label = 'kNN (AUROC = %0.3f)' % kNN_auc,lw=3, alpha=.8)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('1 - Spesificity' ,  fontweight = "bold" , fontsize=20)
plt.ylabel('Sensivity',fontweight = "bold" , fontsize=20)
#plt.tick_params(axis='both', which='major', fontweight = "bold", labelsize=20)
plt.legend( prop={'size':16} , loc = 'lower right')
plt.show()


# precision-recall curve and f1
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from matplotlib import pyplot

yhat = rf.predict(X_test)
lr_precision, lr_recall, _ = precision_recall_curve(Y_test, rf_probs)
lr_f1, lr_auc = f1_score(Y_test, yhat), auc(lr_recall, lr_precision)
yhatsvc = svc.predict(X_test)
lr_precisionsvc, lr_recallsvc, _svc = precision_recall_curve(Y_test, svc_probs)
lr_f1svc, lr_aucsvc = f1_score(Y_test, yhatsvc), auc(lr_recallsvc, lr_precisionsvc)
yhatkNN = kNN.predict(X_test)
lr_precisionkNN, lr_recallkNN, _kNN = precision_recall_curve(Y_test, kNN_probs)
lr_f1kNN, lr_auckNN = f1_score(Y_test, yhatkNN), auc(lr_recallkNN, lr_precisionkNN)
# summarize scores
print('Logistic: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
# plot the precision-recall curves
#no_skill = len(Y_test[Y_test==1]) / len(Y_test)
##pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--')#, label='No Skill')
plt.figure(figsize=(10 , 10))
plt.title('FatCHD' ,  fontweight = "bold" , fontsize=20)
pyplot.plot(lr_recall, lr_precision, color=[0.99,0.0,0.0], label=r'Mean RF (AUCPR = %0.3f)'% (lr_auc),
         lw=3)
pyplot.plot(lr_recallsvc, lr_precisionsvc, color=[0.0,0.5,0.0], label=r'Mean SVC (AUCPR = %0.3f)'% (lr_aucsvc),
         lw=3)
pyplot.plot(lr_recallkNN, lr_precisionkNN, color=[0.0,0.0,0.7], label=r'Mean kNN (AUCPR = %0.3f)'% (lr_auckNN),
         lw=3)
            #label='Random Forest (AUC={:.3f}'.format(lr_auc))
# axis labels
pyplot.xlabel('Recall',  fontweight = "bold" , fontsize=20)
pyplot.ylabel('Precision',  fontweight = "bold" , fontsize=20)
#plt.tick_params(axis='both', which='major', fontweight = "bold", labelsize=20)
# show the legend
pyplot.legend( prop={'size':16} , loc = 'upper right')
# show the plot
pyplot.show()



