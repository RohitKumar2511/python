#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
import os
import matplotlib.pyplot as plt


# In[2]:


import seaborn as sn


# In[4]:


os.getcwd()


# In[14]:


#os.chdir(r'C:\Users\91852\Documents\data science study material\datasets\archive')
os.chdir(r'C:\\Users\\rohit\\Desktop\\Jupiter_BaBa')


# In[15]:


df = pd.read_csv('loan_data_2007_2014.csv')


# In[18]:


df.head(10)


# In[17]:


df.info()


# In[19]:


df.describe()


# In[20]:


df.describe().iloc[:, 6:8]


# In[21]:


df.isnull().sum()


# In[23]:


ndict = dict(df.neighbourhood_group.value_counts())
ndict


# In[24]:


valist = list(ndict.values() )
valist


# In[25]:


count = 0
for i in ndict.keys():
  
  print(f' {i} have {str(valist[count]) } hosts. '  )
  count = count+1


# In[27]:


print(f' All unique neighbourhood groups are:  {df.neighbourhood_group.unique() }')


# In[28]:


print('Total no. of unique Neighborhoods is' ,len(df.neighbourhood.unique()))
print(f'All Unique neighborhoods are \n {df.neighbourhood.unique()}')


# In[30]:


nei_d = dict(df.neighbourhood.value_counts() )
cols = ['Neighborhood', 'Count']
countdf = pd.DataFrame(columns=cols)
countdf.Neighborhood = nei_d.keys()
countdf.Count = nei_d.values()
#countdf = countdf.pivot()


# In[31]:


print( 'Top 10 Neighborhood\n', countdf.head(10) )


# In[32]:


df.columns


# In[ ]:




