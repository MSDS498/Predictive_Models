#!/usr/bin/env python
# coding: utf-8

# In[1]:


# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 19:37:10 2019

@author: RSilvestre
"""

# prepare for Python version 3x features and functions
from __future__ import division, print_function

# import packages for analysis
import os
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import urllib.request
from pathlib import Path
import urllib.parse
import random
import statsmodels.api as sm # logistic regression
import statsmodels.formula.api as smf # R-like model specification
import patsy # translate model specification into design matrices


# In[2]:


# seed value for random number generators to obtain reproducible results
RANDOM_SEED = 1


# In[3]:


#Read in final cleaned and de-duped data
final_data = pd.read_csv('Merged_dataset_w_LatLong_rs_durationV4.csv') # bring in Olist data


# In[4]:


# check data shape
print(final_data.shape)


# In[5]:


# examine data structure
final_data.head()


# In[6]:


# examine summary statistics of numeric variables
print('\nSummary Statistics of Numeric Variables ---------------')
final_data.describe()


# In[7]:


#check data types
final_data.dtypes


# In[8]:


# frequency of satisfaction
print('\nAre Satisfied ---------------')


# In[9]:


final_data['rev_score_sat'].value_counts()


# In[10]:


# define rev score. Frequency of those satisfied
rev_score_sat = final_data['rev_score_sat']

print('\n---------- Percentage of satisfied buyers ---------------')
(rev_score_sat == 'yes').sum() / len(rev_score_sat) * 100


# In[12]:


# create a list w/numeric variables
list1 = ['review_score','payment_value','price','fulfill_duration','freight_value','product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm','product_description_lenght','product_name_lenght','product_photos_qty']


# In[13]:


#check distribution of numeric variables
final_data[list1].hist(bins=50, figsize=(15,15))


# In[14]:


# define model variables
rev_score_sat_binary = final_data['rev_score_sat_binary']
fulfill_duration = final_data['fulfill_duration']
ord_status_delivered = final_data['ord_status_delivered']
ord_status_shipped = final_data['ord_status_shipped']
ord_status_canceled = final_data['ord_status_canceled']


# In[15]:


################Logistic Regression model 1 w/4 explanatory variables. Dependent variable is Binary Satisfaction rating of 1=yes and 0=mo ########################
# gather four explanatory variables and response into a numpy array
# here we use .T to obtain the transpose for the structure we want
model_data2 = np.array([np.array(fulfill_duration), np.array(ord_status_delivered), np.array(ord_status_shipped), np.array(ord_status_canceled), np.array(rev_score_sat_binary)]).T

# check model data shape
model_data2.shape

# cross-validation scoring code adapted from Scikit Learn documentation
from sklearn.metrics import roc_auc_score

# specify the set of classifiers being evaluated
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression

classifier_names = ["Naive_Bayes", "Logistic_Regression"]

classifiers = [BernoulliNB(alpha=1.0, binarize=0.5, 
                           class_prior = [0.5, 0.5], fit_prior=False), 
               LogisticRegression(solver = 'lbfgs')]

# dimensions of the additive model X input and y response
print('\nData dimensions:', model_data2.shape)

# --------------------------------------------------------
# specify the k-fold cross-validation design
from sklearn.model_selection import KFold

# ten-fold cross-validation employed here
N_FOLDS = 10

# set up numpy array for storing results
cv_results = np.zeros((N_FOLDS, len(classifier_names)))

kf = KFold(n_splits = N_FOLDS, shuffle=False, random_state = RANDOM_SEED)

# check the splitting process by looking at fold observation counts
index_for_fold = 0  # fold count initialized

for train_index, test_index in kf.split(model_data2):
    print('\nFold index:', index_for_fold,
          '------------------------------------------')
#   note that 0:model_data.shape[1]-1 slices for explanatory variables
#   and model_data.shape[1]-1 is the index for the response variable    
    X_train = model_data2[train_index, 0:model_data2.shape[1]-1]
    X_test = model_data2[test_index, 0:model_data2.shape[1]-1]
    y_train = model_data2[train_index, model_data2.shape[1]-1]
    y_test = model_data2[test_index, model_data2.shape[1]-1]   

    print('\nShape of input data for this fold:',
          '\nData Set: (Observations, Variables)')
    print('X_train:', X_train.shape)
    print('X_test:',X_test.shape)
    print('y_train:', y_train.shape)
    print('y_test:',y_test.shape)

    index_for_method = 0  # initialize
    for name, clf in zip(classifier_names, classifiers):
        print('\nClassifier evaluation for:', name)
        print('  Scikit Learn method:', clf)
        clf.fit(X_train, y_train)  # fit on the train set for this fold
        # evaluate on the test set for this fold
        y_test_predict = clf.predict_proba(X_test)
        fold_method_result = roc_auc_score(y_test, y_test_predict[:,1]) 
        print('Area under ROC curve:', fold_method_result)
        cv_results[index_for_fold, index_for_method] = fold_method_result
        index_for_method += 1
  
    index_for_fold += 1

cv_results_df = pd.DataFrame(cv_results)
cv_results_df.columns = classifier_names

print('\n----------------------------------------------')
print('Average results from ', N_FOLDS, '-fold cross-validation\n',
      '\nMethod                 Area under ROC Curve', sep = '')     
print(cv_results_df.mean())


# In[ ]:




