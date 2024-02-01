import pandas as pd
import numpy as np
from numpy import linalg as LA
from scipy import sparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample
import json
from sklearn.metrics import accuracy_score
from utility import balancedata, normalizedata, ACC, LOSS




dataset_name=['adult','churn', 'compas', 'dutch', 'heart', 'law']
y_attribute=['income>50K','Churn Value', 'two_year_recid','occupation','output','pass_bar']


performance={}
performance['acc']={}
performance['roc']={}
performance['loss']={}
for data in range(len(dataset_name)):

	df=pd.read_csv ('real_data/%s.csv'%(dataset_name[data]))

	#original data
	# X_train=df.drop('%s'%(y_attribute[data]),axis=1)
	# Y_train=df['%s'%(y_attribute[data])]

	#preprocess data	
	# df=balancedata(df, '%s'%(y_attribute[data]), method='upsampled')
	X_train, Y_train=normalizedata(df, '%s'%(y_attribute[data]), method=None)	

	accuracy_10=[]
	roc_10=[]
	loss_10=[]
	for i in range(10):
		x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.20)
		log_model = LogisticRegression(max_iter=10000, solver='lbfgs')

		log_model.fit(x_train,y_train)


		accuracy_real, roc_real, report_real= ACC(log_model, x_test, y_test)
		Loss= LOSS(log_model, x_test, y_test)
		accuracy_10.append(accuracy_real)
		roc_10.append(roc_real)
		loss_10.append(Loss)
	performance['acc'][dataset_name[data]]=accuracy_10
	performance['roc'][dataset_name[data]]=roc_10
	performance['loss'][dataset_name[data]]=loss_10


file_path = "output/baseline_evaluation.json"  # You can change the filename and path as needed

# Save the dictionary to a JSON file
with open(file_path, "w") as json_file:
  json.dump(performance, json_file, indent=4)

#can load the result by following codes:
# with open(file_path, "r") as json_file:
#     performance = json.load(json_file)
# performance['acc']['adult']
# performance['roc']['adult']
# performance['loss']['adult']

