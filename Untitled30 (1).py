#!/usr/bin/env python
# coding: utf-8

# In[2]:


#This is analysis on the titanic train data 
#the model performance is calculated before testing on the original test data
import os
import pandas as pd 
import numpy as np


# In[67]:


os.getcwd()


# In[68]:


os.chdir(r'C:\Users\91852\Downloads\q1 data')


# In[69]:


titanic_train_df= pd.read_csv('train.csv')


# In[70]:


titanic_train_df.head()


# In[71]:



titanic_train_df.info()


# In[72]:


titanic_train_df.Survived.value_counts()


# In[73]:


titanic_train_df.columns


# In[74]:


titanic_train_df.drop(['Cabin','PassengerId','Name','Ticket'],axis=1, inplace=True)


# In[10]:


X_features = list( titanic_train_df.columns )
X_features.remove( 'Survived' )
X_features


# In[75]:


print(len( titanic_train_df.columns ))


# In[76]:


titanic_train_df.isnull().sum()


# In[77]:


titanic_train_df['Age'] = titanic_train_df['Age'].replace(np.nan, titanic_train_df['Age'].mean())


# In[78]:


titanic_train_df.head()


# In[79]:


titanic_train_df.isnull().sum()


# In[80]:


print(len( titanic_train_df.columns ))


# In[81]:


titanic_df_complete = pd.get_dummies(titanic_train_df[X_features], drop_first = True )


# In[82]:


print(len( titanic_df_complete.columns ))


# In[19]:


titanic_df_complete.head()


# In[83]:


Y = titanic_train_df.Survived
Y


# In[84]:


X = titanic_df_complete
X


# In[85]:


#spliting the original titanic train data into test and train 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 42 )


# In[86]:


#building a logistic regression model
import statsmodels.api as sm
logit = sm.Logit( y_train, sm.add_constant( X_train ) )
lg = logit.fit()
lg.summary()


# In[87]:


#function to get the get significant variables
def get_significant_vars( lm ):
    var_p_vals_df = pd.DataFrame( lm.pvalues )
    var_p_vals_df['vars'] = var_p_vals_df.index
    var_p_vals_df.columns = ['pvals', 'vars']
    return list( var_p_vals_df[var_p_vals_df.pvals <= 0.05]['vars'] )
significant_vars = get_significant_vars( lg )
significant_vars


# In[88]:


#measuring the performance of the model
from sklearn import metrics
def get_predictions( y_test, model ):
    y_pred_df = pd.DataFrame( { 'actual': y_test, "predicted_prob": lg.predict( sm.add_constant( X_test ) ) } )
    return y_pred_df


# In[89]:


y_pred_df = get_predictions( y_test, lg )
y_pred_df


# In[90]:


y_pred_df[0:10]


# In[91]:


y_pred_df['predicted'] = y_pred_df.predicted_prob.map( lambda x: 1 if x > 0.5 else 0)
y_pred_df[0:10]


# In[92]:


import matplotlib.pylab as plt
import seaborn as sn
get_ipython().run_line_magic('matplotlib', 'inline')


# In[93]:


#Function to build and show confusion matrix
def draw_cm( actual, predicted ):
    cm = metrics.confusion_matrix( actual, predicted, [1,0] ) 
    sn.heatmap(cm, xticklabels = ["Dead", "Not Dead"] , yticklabels =["Dead", "Not Dead"] )


# In[31]:


plt.ylabel('True label')


# In[94]:


plt.xlabel('Predicted label')


# In[95]:


plt.show()


# In[96]:


draw_cm( y_pred_df.actual, y_pred_df.predicted )


# In[97]:


plt.ylabel('True label')


# In[98]:


plt.xlabel('Predicted label')


# In[99]:


plt.show()


# In[100]:


draw_cm( y_pred_df.actual, y_pred_df.predicted )


# In[101]:


print( 'Total Accuracy : ',np.round( metrics.accuracy_score( y_test, y_pred_df.predicted ), 2 ) )
print( 'Precision : ',np.round( metrics.precision_score( y_test, y_pred_df.predicted ), 2 ) )
print( 'Recall : ',np.round( metrics.recall_score( y_test, y_pred_df.predicted ), 2 ) )
cm1 = metrics.confusion_matrix( y_pred_df.actual, y_pred_df.predicted, [1,0] )

sensitivity = cm1[0,0]/(cm1[0,0]+cm1[0,1])
print('Sensitivity : ', round( sensitivity, 2) )
specificity = cm1[1,1]/(cm1[1,0]+cm1[1,1])
print('Specificity : ', round( specificity, 2 ) )


# In[102]:


auc_score = metrics.roc_auc_score( y_pred_df.actual, y_pred_df.predicted_prob )
print(round( float( auc_score ), 2 ))


# In[103]:


def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs)
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(6, 4))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('passenger characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return fpr, tpr, thresholds


# In[104]:


fpr, tpr, thresholds = draw_roc( y_pred_df.actual, y_pred_df.predicted_prob)


# In[105]:


thresholds[0:10]


# In[106]:


print(fpr[0:10])


# In[107]:


print(tpr[0:10])


# In[108]:


tpr_fpr = pd.DataFrame( { 'tpr': tpr, 'fpr': fpr, 'thresholds': thresholds } )
tpr_fpr[0:10]


# In[109]:


tpr_fpr['diff'] = tpr_fpr.tpr - tpr_fpr.fpr
tpr_fpr[0:10]


# In[110]:


tpr_fpr.sort_values( 'diff', ascending = False )[0:10]


# In[ ]:





# In[ ]:





# In[ ]:




