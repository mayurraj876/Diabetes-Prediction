#!/usr/bin/env python
# coding: utf-8

# # Prediction of Diabetes based on given attribute using PIMA Diabetes dataset

# In[ ]:


import numpy as np   
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
#models
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from keras.models import Sequential
from keras.layers import Dense
#Evaluation
from sklearn.model_selection import train_test_split,cross_val_score,cross_validate,cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve,auc
#for warning 
from warnings import filterwarnings
filterwarnings("ignore")


# ## Function definations 

# In[55]:


def violin_plot(nrow=4,ncol=2): 
    """
    funtion to plot violin plot for all attributes
    
    input : optional input for number of column and rows for subplot by default value are 2,4 respectively
    
    output : violin plot for all attribute of dataframe 
    
    return : none
    """
    fig = plt.figure(figsize=(14,25))
    fig.tight_layout(pad=3.0)
    nrow,ncol,index=4,2,1    
    for attribute in attributes:
        plt.subplot(nrow, ncol, index)
        plt.title(attribute)
        sns.violinplot(x="Outcome", y=attribute, data=data)
        index+=1
    plt.show()
    
##############################################################################################################
        
def plot_roc(fpr,tpr,auc_model,name_of_algo):
    """
    This function pots the ROC curve with help of False positive rate
    and True positive rate and auc object
    
    input : false positive rate, ture positive rate,auc of model,name_of_algo
    
    output : ROC plot 
    
    return : None
    """
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='RF (area = {:.3f})'.format(auc_model))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve of '+name_of_algo)
    plt.legend(loc='best')
    plt.show()
 
#############################################################################################################

# calculation of median for each attribute for both possible outcome  
def median_target(attribute):
    """
    This function replaces the Nan of given attribute with 
    median when grouped by Outcome into the global variable 
    data("dataframe")
    
    input : attribute 
    
    output : column of that attibute is modified 
    
    return : none
    """
    temp = data[data[attribute].notnull()]# assigning non null value to temp 
    temp = temp[[attribute, 'Outcome']].groupby(['Outcome'])[[attribute]].mean().reset_index() #calculate mean for a attribute with either 0 or      1 outcome 
    mean_op_0=temp[attribute][0]
    mean_op_1=temp[attribute][1]
    data.loc[(data['Outcome'] == 0 ) & (data[attribute].isnull()), attribute] = mean_op_0 #assigning mean to null values 
    data.loc[(data['Outcome'] == 1 ) & (data[attribute].isnull()), attribute] = mean_op_1
    
##############################################################################################################

def median_target_all():
    # calling meadian_target for each attribute
    for attribute in attributes:
            median_target(attribute) 

##############################################################################################################
        
def outliers_removal():
    """
    This function removes outlier of the global variable data(dataframe)
    using IQR method 
    """
    #loop for replacing outlier of all attribute with Nan value 
    for attribute in attributes:
        q1 = data[attribute].quantile(0.25)
        q3 = data[attribute].quantile(0.75)
        iqr = q3 - q1
        fence_low = q1 - 1.5 * iqr 
        fence_high = q3 + 1.5 * iqr
        data.loc[(data[attribute] < fence_low) | (data[attribute] > fence_high),attribute]=np.nan

##############################################################################################################
        
def z_score(df):
    """Function for apply z score standardization
    
       Input: dataframe to be standardized
       
       output :standardized dataframe 
    """
    df_std = df.copy()
    for column in df_std.columns:
        df_std[column] = (df_std[column] - df_std[column].mean()) / df_std[column].std()     
    return df_std

##############################################################################################################

def plot_confusion_matrix(conf_mat):
    df_cm = pd.DataFrame(conf_mat)
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True)


# In[56]:


# loading of PIMA dataset 
data = pd.read_csv("diabetes.csv")
# assigning independent variable to attributes 
attributes = data.drop("Outcome",axis=1).columns


# ### Attributes
#     1. Pregnancies: Number of times pregnant
#     2. Glucose : Plasma glucose concentration a 2 hours in an oral glucose tolerance test
#     3. BloodPressure : Diastolic blood pressure (mm Hg)
#     4. SkinThickness : Triceps skin fold thickness (mm)
#     5. Insulin : 2-Hour serum insulin (mu U/ml)
#     6. BMI : Body mass index (weight in kg/(height in m)^2)
#     7. DiabetesPedigreeFunction : It provided some data on diabetes mellitus history in relatives and the genetic           relationship of those relatives to the patient.
#     8. Age : Age (years)
#     9. Outcome : Class variable (0 or 1) 268 of 768 are 1, the others are 0

# ## EDA

# In[57]:


data.head()


# In[58]:


data.info();


# In[59]:


data.describe()
print(data.describe())


# In[60]:


ax=data["Outcome"].value_counts().plot(kind="bar",color=["blue","red"])
ax.set_xticklabels(['Diabetes','No Diabetes'],rotation=0);


# In[61]:


violin_plot()


# In[62]:


# Pairwise plot of all attributes 
sns.set(style="ticks", color_codes=True)
sns.pairplot(data,hue='Outcome',palette='gnuplot');


# ## Data processing 

# In[63]:


# replacing missing value with nan value
nan_replacement_att=["Glucose",  "BloodPressure","SkinThickness","Insulin","BMI"]
data[nan_replacement_att]=data[nan_replacement_att].replace(0,np.nan)

median_target_all()  # median_target_all replaces nan value with median of that attribute grouped by outcome 


# In[64]:


outliers_removal() # replacing outliers with Nan 

median_target_all()


# In[65]:


print(data.isna().sum())


# In[66]:


fig = plt.figure(figsize=(14,15))
fig.tight_layout(pad=3.0)
nrow,ncol,index=4,2,1    
for attribute in attributes:
    plt.subplot(nrow, ncol, index)
    plt.title(attribute)
    plt.hist(data[attribute][data.Outcome==0],alpha=0.5,label="Outcome=0")
    plt.hist(data[attribute][data.Outcome==1],alpha=0.5,label="Outcome=1")
    plt.legend(loc="best")
    index+=1
plt.show()


# In[67]:



sns.set(style="ticks", color_codes=True)
sns.pairplot(data,hue='Outcome',palette='gnuplot');


# In[68]:


violin_plot()


# In[1]:


# standardization of dataset
data_std=z_score(data)
data_stdta_std.describe()


# In[70]:


# It shows the correlation(positive,neagative) between different columns(only integer value columns) 
corr_matrix = data_std.corr()
fig,ax = plt.subplots(figsize=(15,10))
ax = sns.heatmap(corr_matrix,annot=True,linewidth=0.5,fmt=".2f",cmap="YlOrBr")


# ###### Distribution of data set 

# In[71]:


y = data["Outcome"]
X=data_std.drop("Outcome",axis=1)
X_train,X_test,y_train,y_test =  train_test_split(X,y,test_size=0.2)


# ## Models 

# ### Code using PCA for reducing dimensionality 
# ``` python 
# from sklearn.decomposition import PCA
# from sklearn.pipeline import Pipeline
# for i in range(1,9):
# 
#     list_of_algo=[LogisticRegression(),GaussianNB(),SVC(probability=True),KNeighborsClassifier(),
#                   RandomForestClassifier(),AdaBoostClassifier(),XGBClassifier()]
# 
#     name_of_algo=["LogisticRegression","GaussianNB","SVM","KNeighborsClassifier",
#                   "RandomForestClassifier","AdaBoostClassifier","XGBClassifier"]
#     for i,algorithm in enumerate(list_of_algo):
#         steps = [('pca', PCA(n_components=i)), ('m', algorithm)]
#         model = Pipeline(steps=steps)
# 
#         #### Evaluate model
#         model_score = cross_val_score(model, X, y, scoring='roc_auc',cv=10)
#         print("*"*120)
#         print('Accuracy of {} : {} '.format(name_of_algo[i],(model_score.mean()*100)))
#     print(" ")
#     print("#"*120)
#     print("")
# ```
# 

# In[72]:


from sklearn.model_selection import train_test_split,cross_val_score,cross_validate,cross_val_predict
list_of_algo=[LogisticRegression(),GaussianNB(),SVC(probability=True),KNeighborsClassifier(),
              RandomForestClassifier(),AdaBoostClassifier(),XGBClassifier()]

name_of_algo=["LogisticRegression","GaussianNB","SVM","KNeighborsClassifier",
              "RandomForestClassifier","AdaBoostClassifier","XGBClassifier"]

score = {"accuracy": "accuracy",
         "prec": "precision","recall" : "recall",
         "f1" : "f1","roc_auc" : "roc_auc"}
final_table=pd.DataFrame()
for i,algorithm in enumerate(list_of_algo):
    model=algorithm
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    model_score=cross_validate(model,X,y,cv=10,scoring=score)
    y_pred_cross = cross_val_predict(model,X,y,cv=10)
    
    ## Evalution of model 
    
    conf_mat = confusion_matrix(y, y_pred_cross)
    df_cm = pd.DataFrame(conf_mat)
    sensitivity = conf_mat[0,0]/(conf_mat[0,0]+conf_mat[0,1]) #sensitivity = tp/
    specificity = conf_mat[1,1]/(conf_mat[1,0]+conf_mat[1,1])
    # Roc  
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    auc_model = auc(fpr, tpr)
    print("*"*120)
    print()
    try :
        print(model.coef_)
    except:
        try:
            print(model.feature_importances_)
        except :
            print("null")
    print()
    print('AUC of {} : {:.3f} '.format(name_of_algo[i],(auc_model)))
    print('AVG AUC of {} : {:.3f} + {:.3f} '.format(name_of_algo[i],(model_score["test_roc_auc"].mean()*100),
                                                    model_score["test_roc_auc"].std()))
    print('Specificity of {} : {:.3f} '.format(name_of_algo[i],specificity))
    print('Sensitivity of {} : {:.3f} '.format(name_of_algo[i],sensitivity))
    
   

    plot_roc(fpr,tpr,auc_model,name_of_algo[i])
    sns.heatmap(df_cm, annot=True,fmt="d")
    plt.title(name_of_algo[i])
    plt.show()    


# In[73]:



##Build an model(Neural model )
model = Sequential()
model.add(Dense(8, input_dim=8, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_fit=model.fit(X_train, y_train, epochs=200, batch_size=8)

_, nn_acc = model.evaluate(X_test, y_test)

y_pred = model.predict(X_test).ravel()
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
auc_nn = auc(fpr, tpr)
plot_roc(fpr,tpr,auc_nn,"Neural network")


# ### Grid search for Random forest classifier
# ```python    
# from sklearn.model_selection import GridSearchCV
# print(RandomForestClassifier())
# n_estimators = [100, 200, 250, 300, 350]
# max_depth = [1, 3, 4, 5, 8, 10]
# min_samples_split = [10, 15, 20, 25, 30, 100]
# min_samples_leaf = [1, 2, 4, 5, 7] 
# max_features = ['auto', 'sqrt']
# criterion=['gini']
# bootstrap = [True, False]
# rfr=RandomForestClassifier()
# hyperF = dict(n_estimators = n_estimators, max_depth = max_depth,criterion=criterion,
#               max_features = max_features,  min_samples_split = min_samples_split, 
#               min_samples_leaf = min_samples_leaf,bootstrap = bootstrap)
# 
# gridF = GridSearchCV(rfr, hyperF,scoring='accuracy', cv = 3, verbose = 1, 
#                       n_jobs = -1)
# bestF = gridF.fit(X_train, y_train)
# ```
# 
# ### Grid search for XGB Classifer 
# ```python
# from sklearn.model_selection import GridSearchCV
# model = XGBClassifier()
# param_grid = {
#     'n_estimators': [100,200,300,],
#     'colsample_bytree': [0.5,0.6,0.7],
#     'max_depth': [3,5,8],
#     'reg_alpha': [1.1, 1.2, 1.3],
#     'reg_lambda': [1.1, 1.2, 1.3],
#     'subsample': [0.8, 0.9,1,1.1],
#     'gamma':[1.4,1.5,1.6,]
# }
# gs = GridSearchCV(
#         estimator=model,
#         param_grid=param_grid, 
#         cv=10, 
#         n_jobs=-1, 
#         scoring="roc_auc",
#         verbose=2
#     )
# gsf=gs.fit(X_train,y_train)
# print(gsf.best_params_)
# ```

# ## Finalizing optimal model for web application 

# In[74]:


y = data["Outcome"]
X=data_std.drop(["Outcome"],axis=1)
X_train,X_test,y_train,y_test =  train_test_split(X,y,test_size=0.2)
model_opt = XGBClassifier(colsample_bytree = 0.5,max_depth = 8,n_estimators=100,
                          reg_alpha=1.1,reg_lambda=1.1, subsample=1,gamma=1.5)

model_opt.fit(X_train,y_train)
model_score=cross_val_score(model_opt,X,y,cv=10,scoring="roc_auc")

print(model_opt.score(X_test,y_test))
print("score {:.4f} + {:.4f}".format(model_score.mean(),model_score.std()))


# ## Storing trained model in a file 

# In[75]:


import pickle
# Save trained model to file
pickle.dump(model_opt, open("Diabetes.pkl", "wb"))
loaded_model = pickle.load(open("Diabetes.pkl", "rb"))
loaded_model.predict(X_test)
loaded_model.score(X_test,y_test)


# In[ ]:




