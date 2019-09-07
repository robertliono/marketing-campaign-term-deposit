'''Exploratory data analysis'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Read data from local file
df = pd.read_csv(r'C:\Users\user\Desktop\bank\bank-full.csv', sep=';')
df.describe()
df.info()

mody_df = pd.get_dummies(df.y)
df = pd.concat([df, mody_df], axis='columns').drop(['y','no'], axis=1)

def corr_y(feature):
    '''This is to plot numerical data distribution and relationship between the input and y.'''
    plt.clf()
    dfx = df.groupby(feature)['yes'].agg(['mean','count'])
    plt.style.use('ggplot')
    plt.subplot(2,1,1)
    plt.scatter(x=dfx.index, y=dfx['mean']*100)
    plt.ylabel('% Having term deposit')
    plt.subplot(2,1,2)
    plt.scatter(x=dfx.index, y=dfx['count'])
    plt.ylabel('Data count')
    plt.xlabel(dfx.index.name)
    figure = plt.gcf() # get current figure
    figure.set_size_inches(10, 8)
    plt.savefig(r'C:\Users\user\Desktop\bank\{}_vs_y.png'.format(feature))

corr_y('age')
corr_y('balance')
corr_y('day')
corr_y('duration')
corr_y('campaign')
corr_y('pdays')
corr_y('previous')

def corr_y_cat(feature):
    '''This is to plot categorical data distribution and relationship between the input and y.'''
    plt.clf()
    dfx = df.groupby(feature)['yes'].agg(['mean','count'])
    plt.style.use('ggplot')
    plt.subplot(2,1,1)
    plt.bar(x=dfx.index, height=dfx['mean']*100)
    #plt.plot(x=dfx.index, y=dfx['mean']*100)
    plt.ylabel('% Having term deposit')
    plt.xticks(size=9)
    plt.subplot(2,1,2)
    plt.bar(x=dfx.index, height=dfx['count'])
    plt.ylabel('Data count')
    plt.xlabel(dfx.index.name)
    plt.xticks(size=9)
    figure = plt.gcf() # get current figure
    figure.set_size_inches(16, 12)
    plt.savefig(r'C:\Users\user\Desktop\bank\{}_vs_y.png'.format(feature))

corr_y_cat('job')
corr_y_cat('marital')
corr_y_cat('education')
corr_y_cat('default')
corr_y_cat('housing')
corr_y_cat('loan')
corr_y_cat('contact')
corr_y_cat('month')
corr_y_cat('poutcome')

#Convert pdays, duration, balance, previous into categorical data for plotting, since using scatter plot, there are too many data points.
df['pdays_cat'] = ['1) No' if x==-1 else '2) <50' if x<50 else '3) 50-99' if x<100 else '4) 100-149' if x<150 else '5) 150-199' if x<200 else '6) >=200' for x in df['pdays']]
corr_y_cat('pdays_cat')

df['duration_cat'] = df['duration']//100
corr_y_cat('duration_cat')

df['balance_cat'] = ['0) <0' if x<0 else '1) 0-1k' if x<=1000 else '2) 1k-2k' if x<=2000 else '3) 2k-3k' if x<=3000 else '4) 3k-4k' if x<=4000 else '5) 4k-5k' if x<=5000 else '6) >5k' for x in df['balance']]
corr_y_cat('balance_cat')

df['previous_cat'] = ['0' if x==0 else '1-10' if x<=10 else '11-20' if x<=20 else '21-30' if x<=30 else '31-40' if x<=40 else '41-50' if x<=50 else '50 or more' for x in df['previous']]
corr_y_cat('previous_cat')

##############################
'''Model development'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#Convert categorical data points to binary data points with 1 or 0.
df = pd.read_csv(r'C:\Users\user\Desktop\bank\bank-full.csv', sep=';')
mody_df = pd.get_dummies(df.y)
df = pd.concat([df, mody_df], axis='columns').drop(['y','no'], axis=1)
mod_df = pd.get_dummies(df).drop(['job_unknown','education_unknown','marital_single','default_no','housing_no','loan_no','contact_unknown','poutcome_unknown'], axis=1)
#For pdays, -1 means client hasn't been contacted before in previous campaign. For better model accuracy, convert this to 999 (means very long time has passed by since the last contact)
mod_df['pdays_mod'] = [999 if x==-1 else x for x in mod_df['pdays']]
mod_df = mod_df.drop('pdays', axis=1)
X = mod_df.drop(['yes'], axis=1)
y = mod_df['yes']

#Draw correlation matrix among all the X variables and y
mod_df.corr().to_csv(r'C:\Users\user\Desktop\bank\correlation.csv')

#Upsample the minority y (yes)
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)
Xy_train = pd.concat([X_train, y_train], axis=1)
positive = Xy_train[Xy_train['yes'] == 1]
negative = Xy_train[Xy_train['yes'] == 0]
positive_upsampled = resample(positive, replace=True, n_samples=len(negative), random_state=1)
upsampled = pd.concat([negative, positive_upsampled])
X_train = upsampled.drop('yes', axis=1)
y_train = upsampled['yes']

#Logistic Regression

#with penalty=l2
lr = LogisticRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
y_pred_prob = lr.predict_proba(X_test)[:,1]
accuracy_score(y_test, y_pred)
roc_auc_score(y_test, y_pred_prob)
f1_score(y_test, y_pred)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#with penalty=l1
lr = LogisticRegression(penalty='l1', C=1)
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
y_pred_prob = lr.predict_proba(X_test)[:,1]
accuracy_score(y_test, y_pred)
roc_auc_score(y_test, y_pred_prob)
f1_score(y_test, y_pred)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#Use standard scaler to scale all Xs and perform hyperparameter tuning with different C

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

C = [10, 1, .1, .001]

for c in C:
    lr = LogisticRegression(penalty='l1', C=c)
    lr.fit(X_train_std, y_train)
    y_pred = lr.predict(X_test_std)
    y_pred_prob = lr.predict_proba(X_test_std)[:,1]
    accuracy_score(y_test, y_pred)
    roc_auc_score(y_test, y_pred_prob)
    f1_score(y_test, y_pred)
    print('C:', c)
    print('Coefficient of each feature:', lr.coef_)

#Model with highest F1 score is C=1 (default C), so use that to plot the coefficient to determine important features
lr = LogisticRegression(penalty='l1', C=1)
lr.fit(X_train_std, y_train)
y_pred = lr.predict(X_test_std)
y_pred_prob = lr.predict_proba(X_test_std)[:,1]
coeff = lr.coef_
names = X_train.columns
_ = plt.plot(range(len(names)), coeff[0])
_ = plt.xticks(range(len(names)), names, size=6, rotation=90)
_ = plt.ylabel('Coefficients', size=8)
_ = plt.tight_layout()
plt.show()

#Bagging - decision tree
dt = DecisionTreeClassifier(max_depth=20, min_samples_leaf=0.01, random_state=1)
bc = BaggingClassifier(base_estimator=dt, n_estimators=300, n_jobs=-1)
bc.fit(X_train,y_train)
y_pred = bc.predict(X_test)
y_pred_prob = bc.predict_proba(X_test)[:,1]
accuracy_score(y_test, y_pred)
roc_auc_score(y_test, y_pred_prob)
f1_score(y_test, y_pred)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#Random forest
rf = RandomForestClassifier(n_estimators=300, max_depth=30, min_samples_leaf=0.0001, max_features=12, random_state=1)
rf.fit(X_train,y_train)
y_pred = rf.predict(X_test)
y_pred_prob = rf.predict_proba(X_test)[:,1]
accuracy_score(y_test, y_pred)
roc_auc_score(y_test, y_pred_prob)
f1_score(y_test, y_pred)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#Perform cross validation for random forest
F1_CV = cross_val_score(rf, X_train, y_train, cv= 5, scoring='f1', n_jobs = -1)
F1_CV.mean()
f1_score(y_test, y_pred)
f1_score(y_train, rf.predict(X_train))

#Hyperparameter tuning for random forest
from sklearn.model_selection import GridSearchCV
params_dt = {'max_depth': [25,30,35],'min_samples_leaf': [0.0005,0.0001],'max_features': [9,12,15]}
rf = RandomForestClassifier(n_estimators=300, random_state=1)
grid_rf = GridSearchCV(estimator=rf, param_grid=params_dt, scoring='roc_auc', cv=5, n_jobs=-1)
grid_rf.fit(X_train, y_train)
best_hyperparams = grid_rf.best_params_
print('Best hyerparameters:\n', best_hyperparams)

#Feature importance for random forest. Random forest setup is made to be more general to prevent capturing noise.
rf = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=0.0001, max_features=40, random_state=1)
rf.fit(X_train,y_train)
y_pred = rf.predict(X_test)
f1_score(y_test, y_pred)
importances_rf = pd.Series(rf.feature_importances_, index = X.columns)
sorted_importances_rf = importances_rf.sort_values()
plt.clf()
sorted_importances_rf.plot(kind='barh', color='lightgreen'); plt.show()
