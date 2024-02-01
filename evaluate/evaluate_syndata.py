import pandas as pd
import numpy as np
from numpy import linalg as LA
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample
from sklearn import preprocessing
import json
from utility import balancedata, normalizedata, ACC, LOSS


dataset_name=['adult','churn', 'compas', 'dutch', 'law', 'heart']
y_attribute=['income>50K','Churn Value', 'two_year_recid','occupation','pass_bar','output']
Epsilon = [1/4, 2/4, 3/4, 1, 5/4, 6/4, 7/4, 2]

performance={}
performance['acc']={}
performance['roc']={}
performance['loss']={}
for data in range(len(dataset_name)):
  performance['acc'][dataset_name[data]]={}
  performance['roc'][dataset_name[data]]={}
  performance['loss'][dataset_name[data]]={}

  for epsilon in Epsilon:
    # performance['acc'][dataset_name[data]][epsilon]={}
    # performance['roc'][dataset_name[data]][epsilon]={}
    # performance['loss'][dataset_name[data]][epsilon]={}


    accuracy_10=[]
    roc_10=[]
    loss_10=[]
    for set in range(10):
      df=pd.read_csv ('preprocessd_data/%s.csv'%(dataset_name[data]))
      # df_syn=pd.read_csv('real_data_recount/syn_data_l1/set_%s_syn_%s_eps=%s.csv'%(set, dataset_name[data], epsilon))
      df_syn=pd.read_csv('output/syn_data/set_%s_syn_%s_eps=%s.csv'%(set, dataset_name[data], epsilon))


      #original data
      # X_train=df.drop('%s'%(y_attribute[data]),axis=1)
      # Y_train=df['%s'%(y_attribute[data])]
      # X_train_syn=df_syn.drop('%s'%(y_attribute[data]),axis=1)
      # Y_train_syn=df_syn['%s'%(y_attribute[data])]

      #preprocess data  
      # df=balancedata(df, '%s'%(y_attribute[data]), method='upsampled')
      X_train, Y_train=normalizedata(df, '%s'%(y_attribute[data]), method=None)  
      # df_syn=balancedata(df_syn, '%s'%(y_attribute[data]), method='upsampled')
      X_train_syn, Y_train_syn=normalizedata(df_syn, '%s'%(y_attribute[data]), method=None) 

      for i in range(10):
        x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.20)
        # log_model = SGDClassifier(loss='log_loss', max_iter=10000)
        x_train_syn, x_test_syn, y_train_syn, y_test_syn = train_test_split(X_train_syn, Y_train_syn, test_size=0.20)
        # log_model = SGDClassifier(loss='log_loss', max_iter=10000)
        log_model = LogisticRegression(max_iter=10000, solver='lbfgs')
    		# log_model = LogisticRegression(max_iter=10000, solver='liblinear')
    		# log_model_syn = LogisticRegression(max_iter=10000, solver='liblinear')

        log_model.fit(x_train_syn,y_train_syn)

    		
        accuracy_real, roc_real, _= ACC(log_model, x_test, y_test)
        Loss= LOSS(log_model, x_test, y_test)
        accuracy_10.append(accuracy_real)
        roc_10.append(roc_real)
        loss_10.append(Loss)

    performance['acc'][dataset_name[data]][epsilon]=accuracy_10
    performance['roc'][dataset_name[data]][epsilon]=roc_10
    performance['loss'][dataset_name[data]][epsilon]=loss_10


file_path = "output/syndata_evaluation.json"  
# Save the dictionary to a JSON file
with open(file_path, "w") as json_file:
  json.dump(performance, json_file, indent=4)

#can load the result by following codes:
# with open(file_path, "r") as json_file:
#     performance = json.load(json_file)
# performance['acc']['adult']
# performance['roc']['adult']
# performance['loss']['adult']

