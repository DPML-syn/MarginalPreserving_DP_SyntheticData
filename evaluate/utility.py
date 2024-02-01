def balancedata(df, target, method='downsampled'):
  
  a=df[target].value_counts()[0]
  b=df[target].value_counts()[1]
  if a>b:
    df_majority = df[df[target]==0]
    df_minority = df[df[target]==1]
    if method == 'downsampled':
      df_majority_downsampled = resample(df_majority,
                                replace=False,     # sample with replacement
                                n_samples=b,    # to match majority class
                                random_state=42)
      df_balanced = pd.concat([df_majority_downsampled, df_minority])
    elif method=='upsampled':
      df_minority_upsampled = resample(df_minority,
                                replace=True,     # sample with replacement
                                n_samples=a,    # to match majority class
                                random_state=42)
      df_balanced = pd.concat([df_majority, df_minority_upsampled])

  else:
    df_majority = df[df[target]==1]
    df_minority = df[df[target]==0]
    if method == 'downsampled':
      df_majority_downsampled = resample(df_majority,
                                replace=False,     # sample with replacement
                                n_samples=a,    # to match majority class
                                random_state=42)
      df_balanced = pd.concat([df_majority_downsampled, df_minority])
    elif method=='upsampled':
      df_minority_upsampled = resample(df_minority,
                                replace=True,     # sample with replacement
                                n_samples=b,    # to match majority class
                                random_state=42)
      df_balanced = pd.concat([df_majority, df_minority_upsampled])
  return df_balanced

def normalizedata(df, target, method='l'):
  X_train=df.drop(target,axis=1)
  Y_train=df[target]
  X_label = X_train.columns
  if method==None:
    return X_train, Y_train
  else:
    # 1. divid by the cardinality:
    if method=='l':
      L=np.max(X_train, axis=0)
      X_train_norm = X_train/L
    #2. normalized in l1 norm:
    elif method=='l1':
      X_train_norm=preprocessing.normalize(X_train, norm='l1', axis=0)
    #3. normalized in l2 norm:
    elif method=='l2':
      X_train_norm=preprocessing.normalize(X_train, norm='l2', axis=0)
    #4. Normalize the training data with standard devidation :
    elif method=='std':
      std_scale = preprocessing.StandardScaler().fit(X_train)
      X_train_norm=std_scale.transform(X_train)
    X_train = pd.DataFrame(X_train_norm, columns=X_label)
    return X_train, Y_train


#accuracy
def ACC(model, x, y):
  pred_class = model.predict(x)
  pred_prob = model.predict_proba(x)
  report=classification_report(y,pred_class, output_dict=True)
  accuracy=report['accuracy']
  roc=roc_auc_score(y, pred_prob[:,1])
  return accuracy, roc, report

def LOSS(model, x, y):
  w=model.coef_    #(1, 9)
  b=model.intercept_[0]
  n=y.size
  y_0_index=np.where(y==0)
  y_1_index=np.where(y==1)
  z=np.matmul(x.to_numpy() , np.transpose(w))+b     #z value for test dataset to define the training loss.
  y_hat=1/(1 + np.exp(-z))
  Loss=-1/n*(np.inner(y.to_numpy()[y_1_index], np.log(y_hat[y_1_index].flatten()+1e-16))
  +np.inner(1-y.to_numpy()[y_0_index], np.log(1-y_hat[y_0_index].flatten()+1e-16)))
  return Loss