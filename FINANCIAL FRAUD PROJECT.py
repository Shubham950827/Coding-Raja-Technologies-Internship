#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[2]:


credit_data=pd.read_csv("creditcard.csv");


# In[3]:


credit_data.head()


# In[4]:


credit_data.shape


# In[5]:


credit_data.tail()


# In[6]:


credit_data.info()


# In[7]:


# checking for missing values
credit_data.isnull().sum()


# In[8]:


credit_data['Class'].value_counts()


# In[9]:


#data cleaning
#seperating data variables  analysis
correct_data=credit_data[credit_data.Class==0]
fraud_data=credit_data[credit_data.Class==1]


# In[10]:


print(correct_data.shape)
print(fraud_data.shape)


# In[11]:


correct_data.Amount.describe()


# In[12]:


fraud_data.Amount.describe()


# In[15]:


legit_sample = correct_data.sample(n=492)


# In[17]:


new_sample=pd.concat([legit_sample,fraud_data],axis=0)


# In[19]:


new_sample.head()


# In[20]:


new_sample.tail()


# In[21]:


new_sample.describe()


# In[22]:


new_sample['Class'].value_counts()


# In[23]:


credit_data.groupby('Class').mean()


# In[24]:


new_sample.groupby('Class').mean()


# In[27]:


Y=new_sample.drop(columns='Class', axis=1)
X=new_sample['Class']


# In[29]:


print(Y.head())


# In[31]:


print(X)


# In[33]:


#splitting the training and target set
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=X,random_state=2)


# In[37]:


print(Y.shape,Y_train.shape,Y_test.shape)


# In[38]:


#Model Training
model=LogisticRegression()


# In[40]:


#training the data using training data
model.fit(Y_train,X_train)


# In[41]:


Y_train_accur=model.predict(Y_train)
training_accur=accuracy_score(Y_train_accur,X_train)


# In[44]:


print("Accuracy of model",training_accur)


# In[43]:


Y_test_accur=model.predict(Y_test)
testing_accur=accuracy_score(Y_test_accur,X_test)


# In[45]:


print("Accuracy of model",testing_accur)


# In[ ]:




