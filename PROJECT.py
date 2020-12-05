#!/usr/bin/env python
# coding: utf-8

# In[236]:


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


# # data processing 

# In[237]:


data = pd.read_csv("diabetes.csv")
print(data.info())
attributes = data.drop("Outcome",axis=1).columns
for attribute in attributes:
    plt.figure()
    data.boxplot([attribute])


# In[238]:


data[["Glucose",  "BloodPressure","SkinThickness","Insulin","BMI"]]=data[["Glucose",  "BloodPressure","SkinThickness","Insulin","BMI"]].replace(0,np.nan)
for attribute in attributes:
    q1 = data[attribute].quantile(0.25)
    q3 = data[attribute].quantile(0.75)
    iqr = q3 - q1
    fence_low = q1 - 1.5 * iqr
    fence_high = q3 + 1.5 * iqr
    data.loc[(data[attribute] < fence_low) | (data[attribute] > fence_high),attribute]=np.nan


# In[239]:


print(data.info())


# In[240]:



def median_target(var):   
    temp = data[data[var].notnull()]
    temp = temp[[var, 'Outcome']].groupby(['Outcome'])[[var]].median().reset_index()
    return temp
for attribute in attributes:
    print(median_target(attribute))


# In[241]:


data.loc[(data['Outcome'] == 0 ) & (data['Pregnancies'].isnull()), 'Pregnancies'] = 2.0
data.loc[(data['Outcome'] == 1 ) & (data['Pregnancies'].isnull()), 'Pregnancies'] = 4.0
data.loc[(data['Outcome'] == 0 ) & (data['DiabetesPedigreeFunction'].isnull()), 'DiabetesPedigreeFunction'] = 0.325
data.loc[(data['Outcome'] == 1 ) & (data['DiabetesPedigreeFunction'].isnull()), 'DiabetesPedigreeFunction'] = 0.422
data.loc[(data['Outcome'] == 0 ) & (data['Age'].isnull()), 'Age'] = 26.0
data.loc[(data['Outcome'] == 1 ) & (data['Age'].isnull()), 'Age'] = 36.0
data.loc[(data['Outcome'] == 0 ) & (data['BMI'].isnull()), 'BMI'] = 30.1
data.loc[(data['Outcome'] == 1 ) & (data['BMI'].isnull()), 'BMI'] = 34.3
data.loc[(data['Outcome'] == 0 ) & (data['BloodPressure'].isnull()), 'BloodPressure'] = 70
data.loc[(data['Outcome'] == 1 ) & (data['BloodPressure'].isnull()), 'BloodPressure'] = 74.5
data.loc[(data['Outcome'] == 0 ) & (data['SkinThickness'].isnull()), 'SkinThickness'] = 27
data.loc[(data['Outcome'] == 1 ) & (data['SkinThickness'].isnull()), 'SkinThickness'] = 32
data.loc[(data['Outcome'] == 0 ) & (data['Glucose'].isnull()), 'Glucose'] = 107
data.loc[(data['Outcome'] == 1 ) & (data['Glucose'].isnull()), 'Glucose'] = 140
data.loc[(data['Outcome'] == 0 ) & (data['Insulin'].isnull()), 'Insulin'] = 102.5
data.loc[(data['Outcome'] == 1 ) & (data['Insulin'].isnull()), 'Insulin'] = 169.5


# In[242]:



print(data.info())


# In[243]:


for attribute in attributes:
    plt.figure()
    data.boxplot([attribute])


# In[244]:



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


# In[245]:



for attribute in attributes:
    plt.figure()    
    sns.violinplot(x="Outcome", y=attribute, data=data, palette="muted", split=True)


# In[246]:



def z_score(df):
    """Function for apply z score standardization
       Input: dataframe to be standardized
       output :standardized dataframe 
    """
    df_std = df.copy()
    for column in df_std.columns:
        print(column)
        df_std[column] = (df_std[column] - df_std[column].mean()) / df_std[column].std()
        
    return df_std
    
data_std=z_score(data)


# In[247]:


## It shows the correlation(positive,neagative) between different columns(only integer value columns) 
corr_matrix = data_std.corr()
fig,ax = plt.subplots(figsize=(15,10))
ax = sns.heatmap(corr_matrix,annot=True,linewidth=0.5,fmt=".2f",cmap="RdYlBu")


# ###### Distribution of data set 

# In[248]:





y = data["Outcome"]
X=data_std.drop("Outcome",axis=1)

X_train,X_test,y_train,y_test =  train_test_split(X,y,test_size=0.2)


# 
# # Models 

# In[249]:



log_reg = LogisticRegression(random_state=0)
log_reg.fit(X_train,y_train);
## Evaluating the model
log_score=cross_val_score(log_reg, X, y, cv=8)
print("log_reg =",log_score.mean() )

## Build an model (KNN)
knn = KNeighborsClassifier()
knn.fit(X_train,y_train);
## Evaluating the model
knn_score=cross_val_score(knn, X, y, cv=8)
print("knn",knn_score.mean())


## Build an model (Random forest classifier)
clfm= RandomForestClassifier()
clfm.fit(X_train,y_train);
## Evaluating the model
clfm_score=cross_val_score(clfm, X, y, cv=8)
print("clfm",clfm_score.mean())#

# Build an model (Support Vector Machine)
svm = SVC()
svm.fit(X_train,y_train)
## Evaluating the model
svm_score=cross_val_score(svm, X, y, cv=8)
print("svm",svm_score.mean())

ABC = AdaBoostClassifier(n_estimators=100, random_state=0)
ABC.fit(X, y)
ABC_score=cross_val_score(ABC, X, y, cv=8)
print("ABC",ABC_score.mean())


# In[250]:



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


# In[251]:


print('Accuracy of Neural network: %.2f' % (nn_acc*100))
print('Accuracy of logistic regression : %.2f' % (log_score.mean()*100))
print('Accuracy of Knn: %.2f' % (knn_score.mean()*100))
print('Accuracy of Random Forest Classifier : %.2f' % (clfm_score.mean()*100))
print('Accuracy of SVM : %.2f' % (svm_score.mean()*100))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




