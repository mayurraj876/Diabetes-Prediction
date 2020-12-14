#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np   
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
#models
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from keras.models import Sequential
from keras.layers import Dense
#Evaluation
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import confusion_matrix, classification_report,plot_confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score


# # Data processing 

# In[10]:


data = pd.read_csv("diabetes.csv")
attributes = data.drop("Outcome",axis=1).columns
def violin_plot(nrow=4,ncol=2): 
    """
    funtion to plot violin plot for all attributes
    input : optional input for number of column and rows for subplot by default value are 2,4 respectively
    output : violin plot for all attribute of dataframe 
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
    
violin_plot()
        


# In[11]:


#replacing missing value with nan value
data[["Glucose",  "BloodPressure","SkinThickness","Insulin","BMI"]]=data[["Glucose",  "BloodPressure","SkinThickness","Insulin","BMI"]].replace(0,np.nan)

# calculation of median for each attribute for both possible 
def median_target(attribute):   
        temp = data[data[attribute].notnull()]# assigning non null value to temp 
        temp = temp[[attribute, 'Outcome']].groupby(['Outcome'])[[attribute]].mean().reset_index() #calculate mean for a attribute with either 0 or 1 outcome 
        mean_op_0=temp[attribute][0]
        mean_op_1=temp[attribute][1]
        data.loc[(data['Outcome'] == 0 ) & (data[attribute].isnull()), attribute] = mean_op_0 #assigning mean to null values 
        data.loc[(data['Outcome'] == 1 ) & (data[attribute].isnull()), attribute] = mean_op_1

#loop for replacing outlier of all attribute with Nan value 
for attribute in attributes:
    q1 = data[attribute].quantile(0.25)
    q3 = data[attribute].quantile(0.75)
    iqr = q3 - q1
    fence_low = q1 - 1.5 * iqr
    fence_high = q3 + 1.5 * iqr
    data.loc[(data[attribute] < fence_low) | (data[attribute] > fence_high),attribute]=np.nan
#calling meadian_target for each attribute
for attribute in attributes:
        median_target(attribute) 


# In[12]:


print(data.info())


# In[13]:


fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(12, 10))
fig.tight_layout(pad=3.0)
k=0
for i in range(0,4):
    for j in range(0,2):
        ax[i,j].set_title(attributes[k])
        ax[i,j].hist(data[attributes[k]][data.Outcome==1],color="green")
        k+=1

        
    


# In[14]:


violin_plot()


# In[15]:


# standardization of dataset
def z_score(df):
    """Function for apply z score standardization
       Input: dataframe to be standardized
       output :standardized dataframe 
    """
    df_std = df.copy()
    for column in df_std.columns:
        df_std[column] = (df_std[column] - df_std[column].mean()) / df_std[column].std()     
    return df_std
data_std=z_score(data)


# In[16]:


# It shows the correlation(positive,neagative) between different columns(only integer value columns) 
corr_matrix = data_std.corr()
fig,ax = plt.subplots(figsize=(15,10))
ax = sns.heatmap(corr_matrix,annot=True,linewidth=0.5,fmt=".2f",cmap="RdYlBu")


# ###### Distribution of data set 

# In[17]:


y = data["Outcome"]
X=data_std.drop("Outcome",axis=1)
X_train,X_test,y_train,y_test =  train_test_split(X,y,test_size=0.2)


# 
# # Models 

# In[18]:


"""from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
list_of_algo=[SVC(),AdaBoostClassifier(), 
              RandomForestClassifier(),
              LogisticRegression(),KNeighborsClassifier()]
name_of_algo=["SVM","AdaBoostClassifier", 
              "RandomForestClassifier",
              "LogisticRegression","KNeighborsClassifier"]
for i,algorithm in enumerate(list_of_algo):
    steps = [('pca', PCA(n_components=6)), ('m', algorithm)]
    model = Pipeline(steps=steps)
# evaluate model
    model_score = cross_val_score(model, X, y, scoring='accuracy', n_jobs=-1, error_score='raise')
    print("*"*120)
    print('Accuracy of {} : {} '.format(name_of_algo[i],(model_score.mean()*100)))"""


# In[25]:


def plot_auc(fpr,tpr,auc):
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='RF (area = {:.3f})'.format(auc))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()


# In[31]:


from sklearn.metrics import roc_curve,auc
list_of_algo=[SVC(),AdaBoostClassifier(), 
              RandomForestClassifier(),
              LogisticRegression(),KNeighborsClassifier()]
name_of_algo=["SVM","AdaBoostClassifier", 
              "RandomForestClassifier",
              "LogisticRegression","KNeighborsClassifier"]
for i,algorithm in enumerate(list_of_algo):
    model=algorithm
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    model_score=cross_val_score(model,X,y,cv=15)
    
    y_pred_prob = model.predict_
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    auc_model = auc(fpr, tpr)
    print("*"*120)
    print('Accuracy of {} : {} '.format(name_of_algo[i],(model_score.mean()*100)))
    print('Precision of {} : {} '.format(name_of_algo[i],(precision_score(y_test,y_pred)*100)))
    print('Recall of {} : {} '.format(name_of_algo[i],(recall_score(y_test,y_pred)*100)))
    plot_auc(fpr,tpr,auc_model)
    plot_confusion_matrix(model, X_test, y_test,values_format="d")
    plt.title(name_of_algo[i])
    plt.show()    


# In[27]:


"""from sklearn.model_selection import GridSearchCV
print(RandomForestClassifier())
n_estimators = [100, 200, 250, 300, 350]
max_depth = [1, 3, 4, 5, 8, 10]
min_samples_split = [10, 15, 20, 25, 30, 100]
min_samples_leaf = [1, 2, 4, 5, 7] 
max_features = ['auto', 'sqrt']
bootstrap = [True, False]
rfr=RandomForestClassifier()
hyperF = dict(n_estimators = n_estimators, max_depth = max_depth,
              max_features = max_features,  min_samples_split = min_samples_split, 
              min_samples_leaf = min_samples_leaf,bootstrap = bootstrap)

gridF = GridSearchCV(rfr, hyperF,scoring='accuracy', cv = 5, verbose = 1, 
                      n_jobs = -1)
bestF = gridF.fit(X_train, y_train)"""


# In[36]:


forestOpt = RandomForestClassifier(n_estimators = 100,criterion='gini')

modelOpt = forestOpt.fit(X_train, y_train)
model_score=cross_val_score(forestOpt,X,y,cv=4)
y_pred = modelOpt.predict(X_test)
print(modelOpt.score(X_test,y_test))
print("score",model_score.mean())


# In[20]:



##Build an model(Neural model )
model = Sequential()
model.add(Dense(8, input_dim=8, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_fit=model.fit(X_train, y_train, epochs=200, batch_size=4)
_, nn_acc = model.evaluate(X_test, y_test)

