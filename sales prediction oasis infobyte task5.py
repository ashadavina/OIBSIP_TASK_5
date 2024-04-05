#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df=pd.read_csv("D:\oasis infobyte\Advertising.csv")


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


df=df.drop(columns=["Unnamed: 0"])


# In[7]:


df


# In[8]:


x=df.iloc[:,0:-1]


# In[9]:


x


# In[10]:


y=df.iloc[:,-1]


# In[11]:


y


# In[12]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2, random_state=4)


# In[13]:


x_train


# In[14]:


x_test


# In[15]:


y_train


# In[16]:


y_test


# In[17]:


x_train=x_train.astype(int)
y_train=y_train.astype(int)
x_test=x_test.astype(int)
y_test=y_test.astype(int)


# In[18]:


from sklearn.preprocessing import StandardScaler
Sc=StandardScaler()
x_train_scaled=Sc.fit_transform(x_train)
x_test_scaled=Sc.fit_transform(x_test)


# In[19]:


from sklearn.linear_model import LinearRegression


# In[20]:


lr=LinearRegression()


# In[21]:


lr.fit(x_train_scaled, y_train)


# In[22]:


y_pred=lr.predict(x_test_scaled)


# In[23]:


from sklearn.metrics import r2_score
r2_score(y_test, y_pred)


# In[24]:


import matplotlib.pyplot as plt
plt.scatter(y_test, y_pred, c='g')

