#!/usr/bin/env python
# coding: utf-8

# # Linear and Ridge Regression Models

# In[1]:


# seed value for random number generators to obtain reproducible results
RANDOM_SEED = 1


# In[2]:


# Expect fitted values to be close to zero
SET_FIT_INTERCEPT = True


# In[3]:


# import base packages into the namespace for this program
import numpy as np
import pandas as pd
import os


# In[4]:


# modeling routines from Scikit Learn packages
import sklearn.linear_model 
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score 
from sklearn.model_selection import train_test_split
from math import sqrt  # for root mean-squared error calculation
import matplotlib # import matplotlib
import matplotlib.pyplot as plt  # static plotting
import seaborn as sns  # pretty plotting, including heat map
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler # for standardization
from sklearn.model_selection import KFold, GridSearchCV, train_test_split  # cross-validation / feature tuning
from sklearn.model_selection import cross_val_predict # Cross-validation
from sklearn.metrics import confusion_matrix # confusion matrix


# In[5]:


#check directory
os.getcwd()


# In[6]:


#change directory
#os.chdir('/Users/rsilvestre/Documents/MAIN 2017_2018/NU PA2/MSDS 498')
os.chdir('C:/Users/ashle/Documents/Personal Data/Northwestern/2019-04  fall MSDS498_Sec56 Capstone/Git_repo/Predictive_models')


# In[6]:


# read data for the Boston Housing Study
# creating data frame restdata
ecommerce_input = pd.read_csv('Final_cleaned102419.csv')  


# In[19]:


# correlation heat map setup for seaborn
def corr_chart(df_corr, file_nm='plot-corr-map.pdf'):
    corr=df_corr.corr()
    #screen top half to get a triangle
    top = np.zeros_like(corr, dtype=np.bool)
    top[np.triu_indices_from(top)] = True
    fig=plt.figure()
    fig, ax = plt.subplots(figsize=(12,12))
    sns.heatmap(corr, mask=top, cmap='coolwarm', 
        center = 0, square=True, 
        linewidths=.5, cbar_kws={'shrink':.5}, 
        annot = True, annot_kws={'size': 9}, fmt = '.3f')           
    plt.xticks(rotation=45) # rotate variable labels on columns (x axis)
    plt.yticks(rotation=0) # use horizontal variable labels on rows (y axis)
    plt.title('Correlation Heat Map')   
    plt.savefig(file_nm, 
        bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
        orientation='portrait', papertype=None, format=None, 
        transparent=True, pad_inches=0.25, frameon=None)  
    
    return plt


# In[8]:


# check the pandas DataFrame object boston_input
print('\neCommerce DataFrame (first and last five rows):')
print(ecommerce_input.head())
print(ecommerce_input.tail())


# In[54]:


#print('\nGeneral description of the boston_input DataFrame:')
ecommerce_input.info()
for col in ecommerce_input.columns:
    print(col)

# In[10]:


#drop categorical variables
ecommerce_input = ecommerce_input.drop(['Unnamed: 0'],axis='columns')


# In[11]:


model_data = ecommerce_input.loc[:, ['fulfill_duration','Cust_st_SP','lat_customer','freight_value','Cust_st_BA','Cust_st_PA','long_customer','Cust_st_CE','order_estimated_delivery_mo']]


# In[12]:


print(model_data.describe())


# In[13]:


# examine overall distributions of the variables
model_data.hist( bins = 50, figsize = (30, 20)); plt.show()


# In[14]:


# suppress warning messages
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


# In[15]:


# examine distribution of target variable
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.distplot(model_data['fulfill_duration'], bins=30)
print('\nHistogram of target variable ---------------')
plt.show()


# In[21]:


# examine intercorrelations among software preference variables
# with correlation matrix/heat map
corrplt = corr_chart(df_corr = model_data, file_nm='corr_ecommerce102419.pdf') 
# plt.savefig(, 
#     bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
#     orientation='portrait', papertype=None, format=None, 
#     transparent=True, pad_inches=0.25, frameon=None)
corrplt.show()
corrplt.close()


# In[22]:


#plot scatterplots
plt.figure(figsize=(20, 5))


# In[23]:


#identify features for the model
features = ['Cust_st_SP', 'lat_customer', 'freight_value', 'Cust_st_BA','Cust_st_PA','long_customer','Cust_st_CE','order_estimated_delivery_mo']
target = model_data['fulfill_duration']

for i, col in enumerate(features):
    plt.subplot(1, len(features) , i+1)
    x = model_data[col]
    y = target
    plt.scatter(x, y, marker='o')
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel('Delivery time')


# In[24]:


# create data frame copy for modeling
df_model_data_Scaled = model_data.copy()


# In[25]:


# set up preliminary data for data for fitting the models 
# the first column is the median housing value response
# the remaining columns are the explanatory variables
df_model_data_scaled = np.array([model_data.fulfill_duration,    model_data.Cust_st_SP,    model_data.lat_customer,    model_data.freight_value,    model_data.Cust_st_BA,    model_data.Cust_st_PA,    model_data.long_customer,    model_data.Cust_st_CE,    model_data.order_estimated_delivery_mo]).T


# In[26]:


# dimensions of the polynomial model X input and y response
# preliminary data before standardization
print('\nData dimensions:', df_model_data_scaled.shape)


# In[27]:


# standard scores for the columns... along axis 0
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
print(scaler.fit(df_model_data_scaled))


# In[28]:


# show standardization constants being employed
print(scaler.mean_)
print(scaler.scale_)


# In[29]:


# the model data will be standardized form of preliminary model data
model_data2 = scaler.fit_transform(df_model_data_scaled)


# In[30]:


# check model data
model_data2


# In[31]:


# dimensions of the polynomial model X input and y response
# all in standardized units of measure
print('\nDimensions for model_data:', model_data2.shape)


# In[32]:


X = model_data.drop('fulfill_duration', axis = 1)
y = model_data['fulfill_duration']


# In[33]:


model_data2 = scaler.fit_transform(X)


# # Model Development - 2) Linear Regression

# In[34]:


# split the data to Train and Test data
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(model_data2, y, test_size=.30, random_state=0)


# In[35]:


# Get model performance criteria
def get_performance(model_name, model, X_test, y_test, y_pred):
  mse = mean_squared_error(y_test, y_pred)
  rmse = np.sqrt(mse)
  score = model.score(X_test, y_test)
  
  return mse, rmse, score


# In[36]:


# Fit the model
def fit_pred( model, X_train, y_train, X_test):
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)
  plt.scatter(y_test, y_pred); plt.show()
  return y_pred


# In[37]:


# Show scores
def display_scores(perf_sum):  
  display('{} Performance Summary'.format(perf_sum['Regressor']))
  display('Score: {:.4}'.format(perf_sum['Score']))
  display('Mean-Squared Error : {:.4}'.format(perf_sum['MSE']))
  display('Root Mean-Squared Error : {:.4}'.format(perf_sum['RMSE']))


# In[38]:


perf_cols = ['Regressor', 'Score', 'MSE', 'RMSE']
perf_summary = pd.DataFrame(columns=perf_cols)


# In[39]:


lin_reg = LinearRegression(fit_intercept = SET_FIT_INTERCEPT)


# In[40]:


lin_pred = fit_pred(lin_reg, X_train, y_train, X_test)


# In[41]:


lm_mse, lm_rmse, lm_score = get_performance('Linear', lin_reg, X_test, y_test, lin_pred)


# In[42]:


# Show model performance
lm_perf = pd.DataFrame(['Linear', lm_mse, lm_rmse, lm_score]).T
lm_perf.columns = perf_cols

lm_perf


# In[43]:


perf_summary = perf_summary.append(lm_perf)


# # Ridge Regression

# In[44]:


# Ridge 
ridge_reg = Ridge(alpha = 1, solver = 'cholesky',
                fit_intercept = SET_FIT_INTERCEPT,
                normalize = False,
                random_state = RANDOM_SEED)


# In[45]:


ridge_pred = fit_pred(ridge_reg, X_train, y_train, X_test)


# In[46]:


rr_mse, rr_rmse, rr_score = get_performance('Ridge', ridge_reg, X_test, y_test, ridge_pred)


# In[47]:


rr_perf = pd.DataFrame(['Ridge', rr_mse, rr_rmse, rr_score]).T
rr_perf.columns = perf_cols

rr_perf


# In[48]:


perf_summary = perf_summary.append(rr_perf)


# In[49]:


perf_summary.set_index("Regressor", drop=True, inplace=True)


# In[50]:


# compare performance scores between two models
print('\nPeformance Summary Between Models ---------------')
perf_summary.sort_values(by=['RMSE'])


# In[ ]:




