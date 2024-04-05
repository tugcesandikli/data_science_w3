#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import pandas as pd
import os


# In[10]:


titanic_train = pd.read_csv("/Users/tugcesandikli/Downloads/titanic/train.csv")


# In[11]:


titanic_train.shape   # Check dimensions


# In[12]:


titanic_train.head(5)   # Check the first 5 rows


# In[13]:


categorical = titanic_train.dtypes[titanic_train.dtypes == "object"].index
print(categorical)

titanic_train[categorical].describe()


# In[14]:


titanic_train["Ticket"][0:15]   # Check the first 15 tickets


# In[15]:


titanic_train["Ticket"].describe()


# In[17]:


del titanic_train["Ticket"]   # Remove Ticket


# In[18]:


new_Pclass = pd.Categorical(titanic_train["Pclass"],
                           ordered=True)

new_Pclass = new_Pclass.rename_categories(["Class1","Class2","Class3"])

new_Pclass.describe()


# In[19]:


titanic_train["Pclass"] = new_Pclass


# In[20]:


titanic_train["Cabin"].unique()   # Check unique cabins


# In[21]:


char_cabin = titanic_train["Cabin"].astype(str) # Convert data to str

new_Cabin = np.array([cabin[0] for cabin in char_cabin]) # Take first letter

new_Cabin = pd.Categorical(new_Cabin)

new_Cabin.describe()


# In[22]:


titanic_train["Cabin"] = new_Cabin


# In[23]:


dummy_vector = pd.Series([1,None,3,None,7,8])

dummy_vector.isnull()


# In[24]:


titanic_train["Age"].describe()


# In[25]:


missing = np.where(titanic_train["Age"].isnull() == True)
missing


# In[26]:


len(missing[0])


# In[27]:


titanic_train.hist(column='Age',   # Column to plot
                   figsize=(9,6),  # Plot size
                   bins=20)        # Number of histogram bins


# In[28]:


new_age_var = np.where(titanic_train["Age"].isnull(),   # Logical check
                       28,                              # Value if check is true
                       titanic_train["Age"])            # Value if check false

titanic_train["Age"] = new_age_var

titanic_train["Age"].describe()


# In[29]:


titanic_train.hist(column='Age',   # Column to plot
                   figsize=(9,6),  # Plot size
                   bins=20)        # Number of histogram bins


# In[ ]:




