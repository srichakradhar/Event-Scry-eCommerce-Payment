# -*- coding: utf-8 -*-
import os
import pandas as pd
import statsmodels.api as sm
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
import matplotlib.pyplot as plt 
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
import seaborn as sns

plt.rc("font", size=14)
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
#input data file
os.chdir(r'C:\Users\sridhar\Documents\TCS\hackathons\Event Scry\Final Development Data set')
df = pd.read_csv("Final_Development set_With corrected dates and errors_Dimension1.csv", sep=";")

print("Some statistics about the data:")
df.describe()
browsers = df['ga.browser.encoded'].unique()
browsers.sort()
oss = df['ga.operatingSystem.encoded'].unique()
oss.sort()
languages = df['ga.language.encoded'].unique()
languages.sort()
devices = df['ga.DeviceInfo.encoded'].unique()
devices.sort()
sessions = df['ga.sessionsWithEvent'].unique()
sessions.sort()

df['month'] = (df['ga.dateHourMinute'] / 1000000) % 100
df['month'] = df['month'].map(int)
df['date'] = (df['ga.dateHourMinute'] / 10000) % 100
df['date'] = df['date'].map(int)
df['hour'] = (df['ga.dateHourMinute'] / 100) % 100
df['hour'] = df['hour'].map(int)
df['minute'] = df['ga.dateHourMinute'] % 100
df['time'] = df['ga.dateHourMinute'] % 10000
df['time'] = df['time'].map(int)

#pd.crosstab(df.date, df.Success).plot()
#pd.crosstab(df.hour, df.Success).plot()
#pd.crosstab(df.minute, df.Success).plot()
#pd.crosstab(df.time, df.Success).plot()
#pd.crosstab(df['month'], df.Success).plot()
#pd.crosstab(df['ga.dateHourMinute'], df.Success).plot()
#pd.crosstab(df['ga.sessionDurationBucket'], df.Success).plot()
#pd.crosstab(df['ga.browser.encoded'], df.Success).plot()
#pd.crosstab(df['ga.operatingSystem.encoded'], df.Success).plot()
#pd.crosstab(df['ga.operatingSystemVersion.encoded'], df.Success).plot()
#pd.crosstab(df['ga.language.encoded'], df.Success).plot()
#pd.crosstab(df['ga.DeviceInfo.encoded'], df.Success).plot()
#pd.crosstab(df['ga.sessionsWithEvent'], df.Success).plot()

df = df.drop("Unique.code", axis=1)
df = df.drop(['ga.dateHourMinute'], axis=1)

df['ga.browser.encoded'] = df['ga.browser.encoded'].replace('[A-Z]', '', regex=True).map(int)
df['ga.operatingSystem.encoded'] = df['ga.operatingSystem.encoded'].replace('[A-Z]', '', regex=True).map(int)
df['ga.operatingSystemVersion.encoded'] = df['ga.operatingSystemVersion.encoded'].replace('[A-Z]', '', regex=True).map(int)
df['ga.language.encoded'] = df['ga.language.encoded'].replace('[A-Z]', '', regex=True).map(int)
df['ga.DeviceInfo.encoded'] = df['ga.DeviceInfo.encoded'].replace('[A-Z]', '', regex=True).map(int)

df['ga.DeviceInfo.encoded'] = df['ga.DeviceInfo.encoded'] / df['ga.DeviceInfo.encoded'].max()
df['ga.sessionDurationBucket'] = df['ga.sessionDurationBucket'] / df['ga.sessionDurationBucket'].max()
df['ga.browser.encoded'] = df['ga.browser.encoded'] / df['ga.browser.encoded'].max()
df['ga.operatingSystem.encoded'] = df['ga.operatingSystem.encoded'] / df['ga.operatingSystem.encoded'].max()
df['ga.operatingSystemVersion.encoded'] = df['ga.operatingSystemVersion.encoded'] / df['ga.operatingSystemVersion.encoded'].max()
df['ga.language.encoded'] = df['ga.language.encoded'] / df['ga.language.encoded'].max()
df['ga.DeviceInfo.encoded'] = df['ga.DeviceInfo.encoded'] / df['ga.DeviceInfo.encoded'].max()
df['ga.sessionsWithEvent'] = df['ga.sessionsWithEvent'] / df['ga.sessionsWithEvent'].max()
df['month'] = df['month'] / df['month'].max()
df['date'] = df['date'] / df['date'].max()
df['time'] = df['time'] / df['time'].max()
df['hour'] = df['hour'] / df['hour'].max()
df['minute'] = df['minute'] / df['minute'].max()

#x = df.values #returns a numpy array
#min_max_scaler = preprocessing.MinMaxScaler()
#x_scaled = min_max_scaler.fit_transform(x)
#df = pd.DataFrame(x_scaled)

#df.to_csv('Dataset.csv')

df = df.drop("minute", axis=1)
df = df.drop("time", axis=1)

data_final_vars=df.columns.values.tolist()
y=['Success']
X=[i for i in data_final_vars if i not in y]

logreg = LogisticRegression()
rfe = RFE(logreg, 18)
rfe = rfe.fit(df[X], df[y] )
print(rfe.support_)
print(rfe.ranking_)

logit_model=sm.Logit(df[y],df[X])
result=logit_model.fit()
print(result.summary())

X_train, X_test, y_train, y_test = train_test_split(df[X], df[y], test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

kfold = model_selection.KFold(n_splits=10, random_state=7)
modelCV = LogisticRegression()
scoring = 'accuracy'
results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
print("10-fold cross validation average accuracy: %.3f" % (results.mean()))

confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)