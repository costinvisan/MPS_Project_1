import io
import sys
import pandas as pd
from google.colab import files
import os, sys
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


df = pd.read_csv('mps.dataset.csv')

df.drop('instituția sursă', axis=1, inplace=True)
df.drop('dată debut simptome declarate', axis=1, inplace=True)
df.drop('dată internare', axis=1, inplace=True)
df.drop('data rezultat testare', axis=1, inplace=True)
df.drop('diagnostic și semne de internare', axis=1, inplace=True)

df = df.fillna('0')  

import category_encoders as ce
import numpy as np
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn import metrics, preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import re

for idx in range(df['sex'].shape[0]):
  if df['sex'][idx] == 'masculin':
    df['sex'][idx] = 'MASCULIN'
  if df['sex'][idx][0] == 'F':
    df['sex'][idx] = 'FEMININ'

for string in ['simptome declarate', 'simptome raportate la internare']:
  for idx in range(df[string].shape[0]):
    var = 0
    
    if re.search('tuse', df[string][idx], re.IGNORECASE):
      if re.search('febra', df[string][idx], re.IGNORECASE):
        df[string][idx] = 'tuse + febra'
      elif re.search('dispne', df[string][idx], re.IGNORECASE):
        df[string][idx] = 'tuse + dispnee'
      else:
        df[string][idx] = 'tuse'
      var = 1


    if re.search('febra', df[string][idx], re.IGNORECASE):
      if re.search('tuse', df[string][idx], re.IGNORECASE):
        df[string][idx] = 'tuse + febra'
      elif re.search('dispne', df[string][idx], re.IGNORECASE):
        df[string][idx] = 'febra + dispnee'
      else:
        df[string][idx] = 'febra'
      var = 1

    if re.search('dispnee', df[string][idx], re.IGNORECASE):
      if re.search('febra', df[string][idx], re.IGNORECASE):
        df[string][idx] = 'dispnee + febra'
      elif re.search('tuse', df[string][idx], re.IGNORECASE):
        df[string][idx] = 'tuse + dispnee'
      else:
        df[string][idx] = 'dispnee'
      var = 1

    
    if re.search('asimptom', df[string][idx], re.IGNORECASE):
      df[string][idx] = 'asimptomatic'
      var = 1
    
    if re.search('nu', df[string][idx], re.IGNORECASE):
      df[string][idx] = 'asimptomatic'
      var = 1

    if re.search('absent', df[string][idx], re.IGNORECASE):
      df[string][idx] = 'asimptomatic'
      var = 1

    if re.search('fara acuze', df[string][idx], re.IGNORECASE):
      df[string][idx] = 'asimptomatic'
      var = 1

    if re.search('cefale', df[string][idx], re.IGNORECASE):
      df[string][idx] = 'cefalee'
      var = 1

    if re.search('asteni', df[string][idx], re.IGNORECASE):
      df[string][idx] = 'astenie'
      var = 1

    if re.search('durer', df[string][idx], re.IGNORECASE):
      df[string][idx] = 'dureri'
      var = 1
    
    if var == 0:
      df[string][idx] = 'i'

for idx in range(df['mijloace de transport folosite'].shape[0]):
  var = 0
  
  if re.search('nu', df['mijloace de transport folosite'][idx], re.IGNORECASE) or re.search('nea', df['mijloace de transport folosite'][idx], re.IGNORECASE):
      df['mijloace de transport folosite'][idx] = 'nu'
      var = 1

  if re.search('da', df['mijloace de transport folosite'][idx], re.IGNORECASE):
    df['mijloace de transport folosite'][idx] = 'da'
    var = 1

  if var == 0:
      df['mijloace de transport folosite'][idx] = '0' 


for idx in range(df['istoric de călătorie'].shape[0]):
  var = 0
  
  if re.search('nu', df['istoric de călătorie'][idx], re.IGNORECASE) or re.search('nea', df['istoric de călătorie'][idx], re.IGNORECASE):
      df['istoric de călătorie'][idx] = 'nu'
      var = 1

  if re.search('da', df['istoric de călătorie'][idx], re.IGNORECASE):
    df['istoric de călătorie'][idx] = 'da'
    var = 1
  
  if var == 0:
      df['istoric de călătorie'][idx] = '0' 

aux = 'confirmare contact cu o persoană infectată'

for idx in range(df[aux].shape[0]):
  var = 0
  
  if re.search('tie', df[aux][idx], re.IGNORECASE):
    df[aux][idx] = 'nu stie'
    var = 1
  else:
    if re.search('nu', df[aux][idx], re.IGNORECASE) or re.search('nea', df[aux][idx], re.IGNORECASE):
      df[aux][idx] = 'nu'
      var = 1

    if re.search('da', df[aux][idx], re.IGNORECASE):
      df[aux][idx] = 'da'
      var = 1
  
  if var == 0:
      df[aux][idx] = '0'

aux = 'rezultat testare'

for idx in range(df[aux].shape[0]):
  if re.search('neg', df[aux][idx], re.IGNORECASE):
    df[aux][idx] = 'negativ'
    var = 1
  elif re.search('poz', df[aux][idx], re.IGNORECASE):
      df[aux][idx] = 'pozitiv'
      var = 1
  else:
    df.drop(idx, axis=0, inplace=True)

encoder = ce.OrdinalEncoder(cols=["simptome declarate", "simptome raportate la internare", "istoric de călătorie", "mijloace de transport folosite", "confirmare contact cu o persoană infectată", "rezultat testare"])
encoder.fit(df, verbose=1)
df = encoder.transform(df).iloc[:,0:100]

df = pd.get_dummies(df, columns=["sex"])

df.drop('sex_0', axis=1, inplace=True)

features=df.loc[:,df.columns!='rezultat testare'].values[:,1:]
labels=df.loc[:,'rezultat testare'].values

scaler=MinMaxScaler((-1,1))
x=scaler.fit_transform(features)
y=labels

#Split the dataset
x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.2, random_state=7)

model=XGBClassifier()
model.fit(x_train,y_train)

y_pred=model.predict(x_test)
print(accuracy_score(y_test, y_pred)*100)
