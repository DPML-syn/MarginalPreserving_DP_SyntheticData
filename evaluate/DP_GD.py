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
import math
import matplotlib.pyplot as plt
import gc
from utility import balancedata, normalizedata, ACC, LOSS

# training and testing data using type of np.array
class DP_GD:
  def __init__(self, learning_rate=0.1, T=50, C=1, sigma=1, decay=0.1, DP=False, lambda_val=0):
    self.learning_rate = learning_rate
    self.T = T
    self.weights = None
    self.costs=None
    self.sigma=sigma
    self.C=C   #norm clipping
    self.DP=DP
    self.decay=decay
    self.lambda_val=lambda_val

  def _sigmoid(self, z):
    return 1 / (1 + np.exp(-z))

  def _calculate_gradient(self, X, Y):     #X and Y is all training data

    n, m=X.shape
    z = np.dot(X, self.weights)
    y_pred = self._sigmoid(z)

    regularization_term = 2 * self.lambda_val * self.weights      #l2 regulization, can be delete if not needs.

    # grad = np.dot(x.T, y_pred - y)
    grad_sum=np.zeros(m)
    
    for i in range(n):
      grad = np.dot(X[i].T, y_pred[i] - Y[i])
      grad=grad*np.min([1, self.C/(np.linalg.norm(grad)+1e-16)])     #clipping sample gradient
      grad_sum+=grad
    grad_avg=grad_sum/n
    if self.DP==True:  
      grad_avg=grad_avg+np.random.normal(loc=0, scale=self.sigma, size=m)        
    return grad_avg

  def loss(self, X, y, w):
    n=y.size
    y_0_index=np.where(y==0)
    y_1_index=np.where(y==1)
    z=np.dot(X, w)     #z value for test dataset to define the training loss.
    y_pred = self._sigmoid(z)
    Loss=-1/n*(np.inner(y[y_1_index], np.log(y_pred[y_1_index].flatten()+1e-16))
    +np.inner(1-y[y_0_index], np.log(1-y_pred[y_0_index].flatten()+1e-16)))
    return Loss

  def fit(self, X, Y):
    # indices=partition_indice(X, self.batch_size, self.n_batch)
    # m=self.batch_size*self.n_batch
    self.costs=[]
    X = np.insert(X, 0, 1, axis=1)  # Add bias term
    self.weights = np.zeros(X.shape[1])  # Initialize weight
    n=len(X)
    
    cost_prev=10^10
    # learning_rate=self.learning_rate
    for t in range(self.T):

      grad=self._calculate_gradient(X, Y)
      learning_rate = self.learning_rate / (1 + self.decay * t)     #decay learning rate
      self.weights-=learning_rate * grad
      # cost=self.loss(X, Y, self.weights)
      cost=self.loss(X, Y, self.weights)+ self.lambda_val * np.sum(self.weights ** 2)    #loss with regulization


      if (t % 10 == 0):
        self.costs.append(cost)
        # print("Cost after %i iteration is %f" %(t, cost))
      # if (t % 50 == 0):
      #   learning_rate=learning_rate/2


  def predict(self, X):
    X = np.insert(X, 0, 1, axis=1)  # Add bias term
    z = np.dot(X, self.weights)
    y_pred = self._sigmoid(z)

    return np.round(y_pred)

  def predict_proba(self, X):
    X = np.insert(X, 0, 1, axis=1)  # Add bias term
    z = np.dot(X, self.weights)
    y_pred = self._sigmoid(z)
    return y_pred

dataset_name_set=['adult','churn', 'compas', 'dutch', 'law', 'heart']
y_attribute_set=['income>50K','Churn Value', 'two_year_recid','occupation','pass_bar','output']
iteration_set=[500, 500, 500, 500, 300, 1000]
learning_rate_set=[5, 1, 5, 5, 1, 5]
decay_set=[0.1, 0.5, 0.1, 0.1, 0.1, 0.1]

performance={}
performance['acc']={}
performance['roc']={}
performance['loss']={}

for data in range(len(dataset_name_set)):
# for data in [5]:  #run for single dataset
  # print(data)
  dataset_name=dataset_name_set[data]
  y_attribute=y_attribute_set[data]
  decay_rate=decay_set[data]

  df=pd.read_csv ('preprocessd_data/%s.csv'%(dataset_name))

  # df=balancedata(df, y_attribute, method='upsampled')      #upsample for data heart
  X_train, Y_train=normalizedata(df, y_attribute, method=None)

  Epsilon = [1]
  # Epsilon = [1/4, 2/4, 3/4, 1, 5/4, 6/4, 7/4, 2]

  performance['acc'][dataset_name]={}
  performance['roc'][dataset_name]={}
  performance['loss'][dataset_name]={}

  for epsilon in Epsilon: 

  	acc_10=[]     #save the accuracy for one epsilon for 3 times
  	roc_10=[]
  	loss_10=[]

  	for i in range(10):

  		print('%s_eps=%0.3f, trail_%i'%(dataset_name, epsilon, i))
  		x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.20)
  		# total = df.shape[0]
  		total = len(x_train)
  		C=1
  		delta=1/(total**2)
  		L=1
  		
  		iteration=iteration_set[data]
  		learning_rate=learning_rate_set[data]

  		Sigma=2*math.sqrt(2)*L*math.sqrt(iteration*np.log(1/delta))/(total*epsilon)


  		model = DP_GD(learning_rate=learning_rate, T=iteration, C=C, sigma=Sigma, decay=decay_rate, DP=True)

  		model.fit(x_train.to_numpy(), y_train.to_numpy())
  		pred_class=model.predict(x_test.to_numpy())
  		report=classification_report(y_test,pred_class, output_dict=True)
  		accuracy=report['accuracy']
  		pred_prob=model.predict_proba(x_test.to_numpy())
  		roc_auc = roc_auc_score(y_test, pred_prob)
  		cost=model.costs[-1]

  		acc_10.append(accuracy)
  		roc_10.append(roc_auc)
  		loss_10.append(cost)


  		#clear model before next run for different eps	
  		del model
  		# Run garbage collection
  		gc.collect()

  		print('%s_eps=%0.3f, trail_%i'%(dataset_name, epsilon, i))
  		print('learning rate: %i, iteration: %i'%(learning_rate, iteration))
  		print('accuracy is %0.3f'%(accuracy))
  		print('roc_auc score is %0.3f'%(roc_auc))
  		print('-------------------------------')

  		# plt.plot(model.costs)
  		# plt.ylabel('cost')
  		# plt.xlabel('iterations (per 10)')
  		# plt.title('Cost reduction over time')
  		# plt.show()

  	performance['acc'][dataset_name][epsilon]=acc_10
  	performance['roc'][dataset_name][epsilon]=roc_10
  	performance['loss'][dataset_name][epsilon]=loss_10



		
file_path = "output/dpgd.json" 
# Save the dictionary to a JSON file
with open(file_path, "w") as json_file:
	json.dump(performance, json_file, indent=4)	








