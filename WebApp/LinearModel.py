#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


data = pd.read_csv('Admission_Predict_Ver1.1.csv')
data.head()


# In[3]:


data.drop(['Serial No.'], axis=1 , inplace= True)


# In[4]:


new_cols = ['GRE', 'TOEFL','Univ_rating','SOP','LOR','CGPA','Research','COA']
data.columns=new_cols
data.columns


# In[5]:


new_data = data.loc[:, ['GRE', 'TOEFL', 'CGPA', 'Research', 'COA']]
new_data.head()


# In[6]:


X = new_data[["GRE", "TOEFL", "CGPA", "Research"]]


# In[7]:


y = new_data[["COA"]]


# In[8]:


from sklearn.model_selection import train_test_split


# In[9]:


Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.2,random_state=0)


# In[10]:


from sklearn.linear_model import LinearRegression


# In[11]:


reg= LinearRegression()
reg.fit(Xtrain, ytrain)


# In[12]:


ypred=reg.predict(Xtest)


# In[13]:


from sklearn.metrics import mean_squared_error, r2_score


# In[14]:


np.sqrt(mean_squared_error(y_true=ytest, y_pred=ypred))


# In[15]:


r2_score(y_true=ytest, y_pred=ypred)


# In[16]:


import pickle


# In[17]:


pickle.dump(reg,open("model.pkl","wb"))


# In[18]:


model=pickle.load(open("model.pkl","rb"))

