#!/usr/bin/env python
# coding: utf-8

# # Prediction of Diabetes based on given attribute using PIMA Diabetes dataset

# In[2]:


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
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve,auc
#for warning 
from warnings import filterwarnings
filterwarnings("ignore")


# ## Function definations 

# In[65]:


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
    temp = temp[[attribute, 'Outcome']].groupby(['Outcome'])[[attribute]].median().reset_index() #calculate mean for a attribute with either 0 or      1 outcome 
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
    """
    Function for apply z score standardization
    
    Input: dataframe to be standardized
       
    output :standardized dataframe 
    """
    df_std = df.copy()
    for column in df_std.columns:
        df_std[column] = (df_std[column] - df_std[column].mean()) / df_std[column].std()     
    return df_std

##############################################################################################################

def plot_confusion_matrix(df_cm,name_of_algo):
    """
    Function for plot confusion matrix as heatmap with tittle name of algorith 
    
    Input : confusion matrix converted into dataframe, Name of algorithm  
    
    output : Plot heatmap of confusion matrix 
    
    return : None 
    """
    sns.heatmap(df_cm, annot=True,fmt="d")
    plt.title(name_of_algo)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.show()  

##############################################################################################################

def model_evalution(model,name_of_algo,X,y,score,final_Result):
    """
    Function calculate different performance using cross validate method 
    
    Input  : model = object of algorithm , string type name of algorithm , X,y 
             score = dictionary containing performance metrics , final_Result = empty Dict for storing result 
    return : final_result dictionary containg all performance result,
             df_cm a dataframe containing confusion matrix
    """
    model_score=cross_validate(model,X,y,cv=10,scoring=score)
    y_pred_cross = cross_val_predict(model,X,y,cv=10)
    conf_mat = confusion_matrix(y, y_pred_cross)
    df_cm = pd.DataFrame(conf_mat)
    sensitivity = conf_mat[0,0]/(conf_mat[0,0]+conf_mat[0,1]) # TPR, REC,sensitivity = TP / (TP + FN)
    specificity = conf_mat[1,1]/(conf_mat[1,0]+conf_mat[1,1])# specificity = TN / (TN + FP)
    precision =conf_mat[0,0]/(conf_mat[0,0]+conf_mat[1,1])# PREC, PPV =TP / (TP + FP)
    f1_score= 2*sensitivity*precision/(precision+sensitivity) # 2 * PREC * REC / (PREC + REC)
    avg_auc="{:.3f} +- {:.3f}".format((model_score["test_roc_auc"].mean()*100),(model_score["test_roc_auc"].std()))
    avg_auc="{:.3f} +- {:.3f}".format((model_score["test_roc_auc"].mean()*100),(model_score["test_roc_auc"].std()))
    avg_accuracy="{:.3f} +- {:.3f}".format((model_score["test_accuracy"].mean()*100),(model_score["test_accuracy"].std()))
    final_Result["Specificity"].append(specificity)
    final_Result["Sensitivity/Recall"].append(sensitivity)
    final_Result["Precision"].append(precision)
    final_Result["F1 Score"].append(f1_score)
    final_Result["Accuracy"].append(avg_accuracy)
    final_Result["AUC(ROC)"].append(avg_auc)
    final_Result["Model"].append(name_of_algo)
    return final_Result,df_cm

##############################################################################################################

def grid_search(model,parameter,score,name_model,X_train,y_train,cv=10):
        gridsearch = GridSearchCV(model, parameter,scoring=score, cv = cv, verbose = 2, 
                      n_jobs = -1)
        print(name_model)
        bestfit=gridsearch.fit(X_train,y_train)
        print(bestfit.best_params_)
    


# In[4]:


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

# In[5]:


data.head()


# In[6]:


data.info();


# In[7]:



data.describe()


# In[8]:


ax=data["Outcome"].value_counts().plot(kind="bar",color=["blue","red"])
ax.set_xticklabels(['Diabetes','No Diabetes'],rotation=0);


# In[9]:


violin_plot()


# In[10]:


# Pairwise plot of all attributes 
sns.set(style="ticks", color_codes=True)
sns.pairplot(data,hue='Outcome',palette='gnuplot');


# ## Data processing 

# In[11]:


# replacing missing value with nan value
 nan_replacement_att=["Glucose",  "BloodPressure","SkinThickness","Insulin","BMI"]
data[nan_replacement_att]=data[nan_replacement_att].replace(0,np.nan)

median_target_all()  # median_target_all replaces nan value with median of that attribute grouped by outcome 


# In[12]:


outliers_removal() # replacing outliers with Nan 

median_target_all()


# In[13]:


print(data.isna().sum())


# In[14]:


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


# In[15]:



#sns.set(style="ticks", color_codes=True)
#sns.pairplot(data,hue='Outcome',palette='gnuplot');


# In[16]:


violin_plot()


# In[17]:


# standardization of dataset
data_std=z_score(data)
data_std.describe()


# In[18]:


# It shows the correlation(positive,neagative) between different columns(only integer value columns) 
corr_matrix = data_std.corr()
fig,ax = plt.subplots(figsize=(15,10))
ax = sns.heatmap(corr_matrix,annot=True,linewidth=0.5,fmt=".2f",cmap="YlOrBr")


# ###### Distribution of data set 

# In[19]:


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

# In[20]:


from collections import defaultdict
list_of_algo=[LogisticRegression(),GaussianNB(),SVC(probability=True),KNeighborsClassifier(),
              RandomForestClassifier(),AdaBoostClassifier(),XGBClassifier()]

name_of_algo=["LogisticRegression","GaussianNB","SVM","KNeighborsClassifier",
              "RandomForestClassifier","AdaBoostClassifier","XGBClassifier"]

score = {"accuracy": "accuracy",
         "prec": "precision","recall" : "recall",
         "f1" : "f1","roc_auc" : "roc_auc"}
final_Result= defaultdict(list)
for i,algorithm in enumerate(list_of_algo):
    model=algorithm
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    
    ## Evalution of model 
    final_Result,df_cm = model_evalution(model,name_of_algo[i],X,y,score,final_Result)
    
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
    plot_roc(fpr,tpr,auc_model,name_of_algo[i])
    plot_confusion_matrix(df_cm,name_of_algo[i])
    """
    sns.heatmap(df_cm, annot=True,fmt="d")
    plt.title(name_of_algo[i])
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.show()
    """


# In[21]:


pd.DataFrame.from_dict(final_Result)


# In[22]:



##Build an model(Neural model )
model = Sequential()
model.add(Dense(8, input_dim=8, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_fit=model.fit(X_train, y_train, epochs=200, batch_size=8);

_, nn_acc = model.evaluate(X_test, y_test)

y_pred = model.predict(X_test).ravel()
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
auc_nn = auc(fpr, tpr)
plot_roc(fpr,tpr,auc_nn,"Neural network")


# In[72]:


param_adaboost = {
 'n_estimators': [50*x for x in range(1,10)],
 'learning_rate' : [0.0001, 0.001, 0.01, 0.1, 1.0],
  }
param_grid = {
    'n_estimators': list(range(50,250,50)),
    'colsample_bytree': [0.1*x for x in range(1,10)],
    'max_depth': [x for x in range(5,11)],
    'reg_alpha': [0.1*x for x in range(7,13)],
    'reg_lambda': [0.1*x for x in range(7,13)],
    'subsample': [0.1*x for x in range(7,13)],
    'gamma':[0.1*x for x in range(7,13)]
}
param_rf={
    'n_estimators' : [50*x for x in range(1,10)],
    'max_depth' : [x for x in range(1,15,2)],
    'min_samples_split' : [5*x for x in range(1,20,2)],
    'min_samples_leaf' : [x for x in range(1,10,2)],
    'max_features' : ['auto', 'sqrt'],
    'criterion' : ['gini'],
    'bootstrap' : [True, False]
}
param_knn ={
    'leaf_size' : [3*x for x in range(1,20)],
    'n_neighbors' : [x for x in range(1,20,2)],
    'weights' : ['uniform', 'distance']
}
param_SVM = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf', 'poly', 'sigmoid']}
para_list=[param_xgb,param_rf,param_adaboost,param_knn,param_SVM]
gridSearch_alg=[XGBClassifier(),RandomForestClassifier(),AdaBoostClassifier(),
                KNeighborsClassifier(),SVC(probability=True)]
gridSearch_alg_name=["XGBClassifier","RandomForestClassifier","AdaBoostClassifier",
                "KNeighborsClassifier","SVC"]
scores = ["accuracy","f1","roc_auc"]
score="roc_auc"


# ## XGBClassifier
# 
# grid_search(gridSearch_alg[0], para_list[0],score,gridSearch_alg_name[0],X,y,3)
# 
# Fitting 3 folds for each of 279936 candidates, totalling 839808 fits
# 
# {'colsample_bytree': 0.30000000000000004, 'gamma': 0.9, 'max_depth': 7, 'n_estimators': 100, 
#  'reg_alpha': 1.2000000000000002, 'reg_lambda': 0.8, 'subsample': 0.8}
#         

# # Grid search code

# ## RandomForestClassifier
# 
# grid_search(gridSearch_alg[1], para_list[1],score,gridSearch_alg_name[1],X,y,3)
# 
# Fitting 3 folds for each of 12600 candidates, totalling 37800 fits
# 
# {'bootstrap': True, 'criterion': 'gini', 'max_depth': 11, 'max_features': 'auto', 'min_samples_leaf': 3, 'min_samples_split': 5, 'n_estimators': 100}
# 

# ## AdaBoostClassifier
# 
# grid_search(gridSearch_alg[2], para_list[2],score,gridSearch_alg_name[2],X,y)
# 
# Fitting 10 folds for each of 45 candidates, totalling 450 fits
# 
# {'learning_rate': 0.1, 'n_estimators': 450}

# ## KNeighborsClassifier
# 
# grid_search(gridSearch_alg[3], para_list[3],score,gridSearch_alg_name[3],X_train,y_train)
# 
# Fitting 10 folds for each of 380 candidates, totalling 3800 fits
# 
# {'leaf_size': 3, 'n_neighbors': 17, 'weights': 'distance'}

# ## SVM 
# 
# grid_search(gridSearch_alg[4], para_list[4],score,gridSearch_alg_name[4],X_train,y_train)
# 
# Fitting 10 folds for each of 75 candidates, totalling 750 fits
# 
# {'C': 0.1, 'gamma': 1, 'kernel': 'rbf'}

# ## Finalizing optimal model for web application 

# In[78]:


y = data["Outcome"]
X=data_std.drop(["Outcome"],axis=1)
X_train,X_test,y_train,y_test =  train_test_split(X,y,test_size=0.2)
score = {"accuracy": "accuracy",
         "prec": "precision","recall" : "recall",
         "f1" : "f1","roc_auc" : "roc_auc"}

opt_alg_name=["XGBClassifier","RandomForestClassifier","AdaBoostClassifier",
                "KNeighborsClassifier"]

opt_algo =   [XGBClassifier(colsample_bytree = 0.3,max_depth = 7,n_estimators=100,
                        reg_alpha=1.2,reg_lambda=0.8, subsample=0.8,gamma=0.9),
            RandomForestClassifier(bootstrap = True, criterion = 'gini',max_depth = 11,
                        max_features= 'auto', min_samples_leaf = 3, min_samples_split = 5,
                        n_estimators = 100),
            AdaBoostClassifier(learning_rate = 0.1, n_estimators = 450),
            KNeighborsClassifier(leaf_size = 3, n_neighbors = 17, weights = 'distance')]
opt_final_Result= defaultdict(list)
for i,algorithm in enumerate(opt_algo):
    model=algorithm
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    
    ## Evalution of model 
    opt_final_Result,df_cm = model_evalution(model,opt_alg_name[i],X,y,score,opt_final_Result)
    
    # Roc  
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    auc_model = auc(fpr, tpr)
    
    plot_roc(fpr,tpr,auc_model,opt_alg_name[i])
    plot_confusion_matrix(df_cm,opt_alg_name[i])
"""model_opt.fit(X_train,y_train)

xgboost_opt_result= defaultdict(list)
xgboost_opt_result,df_cm = model_evalution(model_opt,"XGBoost",X,y,score,xgboost_opt_result)
"""


# In[79]:


pd.DataFrame.from_dict(opt_final_Result)


# ## Storing trained model in a file 

# In[ ]:


import pickle
# Save trained model to file
pickle.dump(model_opt, open("Diabetes.pkl", "wb"))
loaded_model = pickle.load(open("Diabetes.pkl", "rb"))
loaded_model.predict(X_test)
loaded_model.score(X_test,y_test)


# In[ ]:





# In[ ]:




