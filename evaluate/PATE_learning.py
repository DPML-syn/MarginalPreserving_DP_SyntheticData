#utitity
import pandas as pd
from sympy import symbols, solve
import math
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
import json

# to partition the training data into small subset
def partition_indice(n, batch_size, n_batch):
  rand_indice=np.random.permutation(n)
  indice=[]
  start=0
  for i in range(n_batch):
    end=start+batch_size
    indice.append(rand_indice[start:end])
    start=end
  return indice

def Gamma(T, delta, epsilon):
  x = symbols('x')
  expr =4*T*(x**2)+2*x*math.sqrt(2*T*np.log(1/delta))-epsilon
  gamma = solve(expr)
  return gamma[1]

# Aggregate teacher predictions
def aggregate_predictions(predictions, sigma):
  aggregated_preds = []
  for preds in zip(*predictions):
    vote_counts = np.bincount(preds)  # Count votes

    # Add noise to the vote counts
    # noise = np.random.normal(0, sigma, len(vote_counts))
    noise = np.random.laplace(0, 1/gamma, len(vote_counts))
    noisy_vote_counts = vote_counts + noise

    aggregated_pred = np.argmax(noisy_vote_counts)  # Select the most voted class
    aggregated_preds.append(aggregated_pred)
  return aggregated_preds

# Evaluate the aggregated predictions
def evaluate_predictions(predictions, y_true):
    accuracy = accuracy_score(y_true, predictions)
    return accuracy


#evaluate the PATE for diff dataset with diff teacher# with diff eps in 3 trails.
# dataset_name=['adult','churn', 'compas', 'dutch', 'heart', 'law']
# y_attribute=['income>50K','Churn Value', 'two_year_recid','occupation','output','pass_bar']
# teacher_n_list=[[100, 300, 500],[100, 150, 200], [100, 200, 300], [100, 450, 800], [5, 10, 15], [100, 150, 200]]
# # epsilon_set=[1/4, 2/4, 3/4, 1, 5/4, 6/4, 7/4, 2]


epsilon_set=[1]   
dataset_name=['heart']
y_attribute=['output']
teacher_n_list=[[5, 10, 15]]
T=100   #the query number, all the training size for student model
# T=20 #for heart data only
delta=1e-5


acc_overall={}
for data in range(len(dataset_name)):
    print(dataset_name[data])
    #query equal the same as in PATE paper https://arxiv.org/pdf/1610.05755.pdf
    acc_overall[dataset_name[data]]={}
    
  # i_count=0
    for n_teachers in teacher_n_list[data]:
      print(n_teachers)
      acc_overall[dataset_name[data]][n_teachers]={}
      # acc_overall[dataset_name[data]][eps]={}
      for eps in epsilon_set:
        print(eps)
        acc_overall[dataset_name[data]][n_teachers][eps]={}
        accuracys=[]
        for j in range(10):

          df=pd.read_csv ('real_data_recount/real_data/%s.csv'%(dataset_name[data]))
          X_train=df.drop('%s'%(y_attribute[data]),axis=1)
          Y_train=df['%s'%(y_attribute[data])]

          x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.20)

          x_teacher, x_student, y_teacher, y_student_true = train_test_split(X_train, Y_train, test_size=T)   

          total = df.shape[0]
       

        # Generate teacher models
           
          teachers = []
          n=len(y_teacher)
          subset_size=n//n_teachers
          indices= partition_indice(n, subset_size, n_teachers)

          for k in range(n_teachers):
            # teacher = LogisticRegression(max_iter=10000, solver='liblinear', intercept_scaling=1)
            teacher = LogisticRegression(max_iter=10000, solver='lbfgs')
            # teacher = SGDClassifier(loss='log_loss', max_iter=1000)
            teacher.fit(x_teacher.to_numpy()[indices[k]], y_teacher.to_numpy()[indices[k]])
            # teacher.fit(x_train, y_train)
            teachers.append(teacher)



            # Generate predictions from the teachers
            teacher_predictions = []
            for teacher in teachers:
                teacher_pred = teacher.predict(x_student.to_numpy())
                teacher_predictions.append(teacher_pred)

          # i_count+=1
          gamma=Gamma(T, delta, eps)
          # Aggregate the teacher predictions
          aggregated_predictions = aggregate_predictions(teacher_predictions, gamma)

          student = LogisticRegression(max_iter=10000, solver='lbfgs')
          student.fit(x_student.to_numpy(), aggregated_predictions)
          student_pred=student.predict(x_test)

          # Evaluate the aggregated predictions
          accuracy = evaluate_predictions(student_pred, y_test)
          accuracys.append(accuracy)
          # print("Accuracy %s_data with %i teachers, eps=%0.3f is :%0.3f"%(dataset_name[data], n_teachers, i, accuracy))
        print(accuracys)
        acc_overall[dataset_name[data]][n_teachers][eps]=accuracys

#save all accuracys to a json file
file_path = "real_data_recount/evaluate/pate_student_heart.json"  # You can change the filename and path as needed



# Save the dictionary to a JSON file
with open(file_path, "w") as json_file:
    json.dump(acc_overall, json_file, indent=4)

#can load the result by following codes:
# with open(file_path, "r") as json_file:
#     pate_acc = json.load(json_file)
# pate_acc['adult']['100']['1']


