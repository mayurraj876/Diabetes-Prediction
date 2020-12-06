#!/usr/bin/env python
# coding: utf-8

# In[134]:


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
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score


# # Data processing 

# In[135]:


data = pd.read_csv("diabetes.csv")
attributes = data.drop("Outcome",axis=1).columns
for attribute in attributes:
    plt.figure()
    sns.violinplot(x="Outcome", y=attribute, data=data, palette="muted", split=True)


# In[136]:


#replacing missing value with nan value
data[["Glucose",  "BloodPressure","SkinThickness","Insulin","BMI"]]=data[["Glucose",  "BloodPressure","SkinThickness","Insulin","BMI"]].replace(0,np.nan)

#loop for replacing outlier of all attribute with Nan value 
for attribute in attributes:
    q1 = data[attribute].quantile(0.25)
    q3 = data[attribute].quantile(0.75)
    iqr = q3 - q1
    fence_low = q1 - 1.5 * iqr
    fence_high = q3 + 1.5 * iqr
    data.loc[(data[attribute] < fence_low) | (data[attribute] > fence_high),attribute]=np.nan


# In[137]:


print(data.info())


# In[138]:


# calculation of median for each attribute for both possible 
def median_target(attribute):   
        temp = data[data[attribute].notnull()]# assigning non null value to temp 
        temp = temp[[attribute, 'Outcome']].groupby(['Outcome'])[[attribute]].median().reset_index() #calculate mean for a attribute with either 0 or 1 outcome 
        mean_op_0=temp[attribute][0]
        mean_op_1=temp[attribute][1]
        data.loc[(data['Outcome'] == 0 ) & (data[attribute].isnull()), attribute] = mean_op_0 #assigning mean to null values 
        data.loc[(data['Outcome'] == 1 ) & (data[attribute].isnull()), attribute] = mean_op_1
#calling meadian_target for each attribute
for attribute in attributes:
        median_target(attribute) 
print(data.info())


# In[139]:



fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(12, 10))
fig.tight_layout(pad=3.0)
ax[0,0].set_title('Glucose')
ax[0,0].hist(data.Glucose[data.Outcome==1]);
ax[0,1].set_title('Pregnancies')
ax[0,1].hist(data.Pregnancies[data.Outcome==1]);
ax[1,0].set_title('Age')
ax[1,0].hist(data.Age[data.Outcome==1]);
ax[1,1].set_title('Blood Pressure')
ax[1,1].hist(data.BloodPressure[data.Outcome==1]);
ax[2,0].set_title('Skin Thickness')
ax[2,0].hist(data.SkinThickness[data.Outcome==1]);
ax[2,1].set_title('Insulin')
ax[2,1].hist(data.Insulin[data.Outcome==1]);
ax[3,0].set_title('BMI')
ax[3,0].hist(data.BMI[data.Outcome==1]);
ax[3,1].set_title('Diabetes Pedigree Function')
ax[3,1].hist(data.DiabetesPedigreeFunction[data.Outcome==1]);


# In[140]:



for attribute in attributes:
    plt.figure()    
    sns.violinplot(x="Outcome", y=attribute, data=data, palette="muted", split=True)


# In[141]:


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


# In[142]:


# It shows the correlation(positive,neagative) between different columns(only integer value columns) 
corr_matrix = data_std.corr()
fig,ax = plt.subplots(figsize=(15,10))
ax = sns.heatmap(corr_matrix,annot=True,linewidth=0.5,fmt=".2f",cmap="RdYlBu")


# ###### Distribution of data set 

# In[143]:


y = data["Outcome"]
X=data_std.drop("Outcome",axis=1)
X_train,X_test,y_train,y_test =  train_test_split(X,y,test_size=0.2)


# 
# # Models 

# In[144]:


list_of_algo=[SVC(),AdaBoostClassifier(), 
              RandomForestClassifier(),
              LogisticRegression(),KNeighborsClassifier()]
name_of_algo=["SVM","AdaBoostClassifier", 
              "RandomForestClassifier",
              "LogisticRegression","KNeighborsClassifier"]
i=0
for algorithm in list_of_algo:
    model=algorithm
    model.fit(X_train,y_train)
    model_score=cross_val_score(model,X,y,cv=15)
    print("*"*120)
    print('Accuracy of {} : {} '.format(name_of_algo[i],(model_score.mean()*100)))
    i+=1
    
    


# In[145]:



##Build an model(Neural model )
model = Sequential()
model.add(Dense(8, input_dim=8, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=200, batch_size=10)
_, nn_acc = model.evaluate(X_test, y_test)


# In[146]:


print('Accuracy of Neural network: %.2f' % (nn_acc*100))
print('Accuracy of logistic regression : %.2f' % (log_score.mean()*100))
print('Accuracy of Knn: %.2f' % (knn_score.mean()*100))
print('Accuracy of Random Forest Classifier : %.2f' % (clfm_score.mean()*100))
print('Accuracy of SVM : %.2f' % (svm_score.mean()*100))

