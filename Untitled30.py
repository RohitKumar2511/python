#!/usr/bin/env python
# coding: utf-8

# In[1]:


#This is analysis on the titanic train data 
#the model performance is calculated before testing on the original test data
import os
import pandas as pd 
import numpy as np


# In[2]:


os.getcwd()


# In[3]:


os.chdir(r'C:\Users\91852\Downloads\q1 data')


# In[4]:


titanic_train_df= pd.read_csv('train.csv')


# In[5]:


titanic_train_df.head()


# In[6]:



titanic_train_df.info()


# In[7]:


titanic_train_df.Survived.value_counts()


# In[8]:


titanic_train_df.columns


# In[9]:


#data preprocessing

titanic_train_df.drop(['Cabin','PassengerId','Name','Ticket'],axis=1, inplace=True)


# In[10]:


X_features = list( titanic_train_df.columns )
X_features.remove( 'Survived' )
X_features


# In[77]:


print(len( titanic_train_df.columns ))


# In[12]:


titanic_train_df.isnull().sum()


# In[13]:


titanic_train_df['Age'] = titanic_train_df['Age'].replace(np.nan, titanic_train_df['Age'].mean())


# In[14]:


titanic_train_df.head()


# In[15]:


titanic_train_df.isnull().sum()


# In[16]:


print(len( titanic_train_df.columns ))


# In[17]:


titanic_df_complete = pd.get_dummies(titanic_train_df[X_features], drop_first = True )


# In[18]:


print(len( titanic_df_complete.columns ))


# In[19]:


titanic_df_complete.head()


# In[20]:


Y = titanic_train_df.Survived
Y


# In[21]:


X = titanic_df_complete
X


# In[22]:


#spliting the original titanic train data into test and train 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 42 )


# In[23]:


#building a logistic regression model
import statsmodels.api as sm
logit = sm.Logit( y_train, sm.add_constant( X_train ) )
lg = logit.fit()
lg.summary()


# In[24]:


#function to get the get significant variables
def get_significant_vars( lm ):
    var_p_vals_df = pd.DataFrame( lm.pvalues )
    var_p_vals_df['vars'] = var_p_vals_df.index
    var_p_vals_df.columns = ['pvals', 'vars']
    return list( var_p_vals_df[var_p_vals_df.pvals <= 0.05]['vars'] )
significant_vars = get_significant_vars( lg )
significant_vars


# In[25]:


#measuring the performance of the model
from sklearn import metrics
def get_predictions( y_test, model ):
    y_pred_df = pd.DataFrame( { 'actual': y_test, "predicted_prob": lg.predict( sm.add_constant( X_test ) ) } )
    return y_pred_df


# In[26]:


y_pred_df = get_predictions( y_test, lg )
y_pred_df


# In[27]:


y_pred_df[0:10]


# In[28]:


y_pred_df['predicted'] = y_pred_df.predicted_prob.map( lambda x: 1 if x > 0.5 else 0)
y_pred_df[0:10]


# In[34]:


import matplotlib.pylab as plt
import seaborn as sn
get_ipython().run_line_magic('matplotlib', 'inline')


# In[69]:


#Function to build and show confusion matrix
def draw_cm( actual, predicted ):
    cm = metrics.confusion_matrix( actual, predicted, [1,0] ) 
    sn.heatmap(cm, annot=True, xticklabels = ["Dead", "Not Dead"] , yticklabels =["Dead", "Not Dead"] )


# In[70]:


plt.ylabel('True label')


# In[71]:


plt.xlabel('Predicted label')


# In[72]:


plt.show()


# In[73]:


draw_cm( y_pred_df.actual, y_pred_df.predicted )


# In[74]:


plt.ylabel('True label')


# In[75]:


plt.xlabel('Predicted label')


# In[76]:


plt.show()


# In[68]:


draw_cm( y_pred_df.actual, y_pred_df.predicted )


# In[ ]:





# In[ ]:




