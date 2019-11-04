# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 15:34:49 2019

@author: ashley
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta, datetime
from math import pi, sin, cos, acos
import seaborn as sns  # pretty plotting, including heat map

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from inspect import signature



#set some Pandas options
pd.set_option('display.max_columns', 65) 
pd.set_option('display.max_rows', 20) 
pd.set_option('display.width', 160)

wkg_dir = 'C:/Users/ashle/Documents/Personal Data/Northwestern/2019-04  fall MSDS498_Sec56 Capstone/Git_repo/Predictive_models/'


orders_df = pd.read_csv('../DataSet/Order_level_dataset.csv')
orders_df.head()
orders_df.info()

#convert fields to more specific datatypes (esp. date fields)
orders_df.order_estimated_delivery_date = pd.to_datetime(orders_df.order_estimated_delivery_date)  
orders_df.order_purchase_timestamp = pd.to_datetime(orders_df.order_purchase_timestamp)  
orders_df.order_approved_at = pd.to_datetime(orders_df.order_approved_at)  
orders_df.order_delivered_carrier_date = pd.to_datetime(orders_df.order_delivered_carrier_date)  
orders_df.order_delivered_customer_date = pd.to_datetime(orders_df.order_delivered_customer_date)  
orders_df.ship_limit_initial = pd.to_datetime(orders_df.ship_limit_initial)  
orders_df.ship_limit_final = pd.to_datetime(orders_df.ship_limit_final)  
orders_df.earliest_review_dt = pd.to_datetime(orders_df.earliest_review_dt)  
orders_df.latest_review_dt = pd.to_datetime(orders_df.latest_review_dt)  




# =================
#   Prep dataset
# =================
#our predicted variable!
orders_df['fulfill_duration'] = (orders_df.order_delivered_customer_date - orders_df.order_purchase_timestamp).dt.total_seconds()/86400



# Create calc'd variables that are suspected to be valuable for prediction
# ------------------------------------------------------------------------
orders_df['late_delivery_flag'] = (orders_df.order_delivered_customer_date >= orders_df.order_estimated_delivery_date + timedelta(days=1)).astype(int)
#sum(orders_df.late_delivery_flag)  #6535

# estimated delivery time (in days)
orders_df['est_delivery_time_days'] = (orders_df.order_estimated_delivery_date - orders_df.order_purchase_timestamp).dt.total_seconds()/86400
orders_df['est_delivery_time_days'] = np.ceil(orders_df['est_delivery_time_days']).astype(int)

#approval time
orders_df['approval_time_days'] = (orders_df.order_approved_at - orders_df.order_purchase_timestamp).dt.total_seconds()/86400



    
# distance/geography related aspects
# ----------------------------------
def distance_calcn_rough(start_lat, start_long, end_lat, end_long):
    #check for missing values and quit if any are found
    if ( np.isnan(start_lat) | np.isnan(start_long) | np.isnan(end_lat) | np.isnan(end_long) ):
        return None

    #convert from radians to radians
    start_lat = start_lat * pi / 180.0
    start_long = start_long * pi / 180.0
    end_lat = end_lat * pi / 180.0
    end_long = end_long * pi / 180.0
  
    cosine_val = sin(start_lat)*sin(end_lat) + cos(start_lat)*cos(end_lat)*cos(start_long - end_long)
  
    if (cosine_val > 1) | (cosine_val < -1): 
        cosine_val = round(cosine_val, 6)      #round it off so we aren't losing cases due to machine precision issues (esp. when cosine_val ~1 b/c they are at the exact same location)

    if (cosine_val > 1) | (cosine_val < -1):
        rtrn_val = None
    else: 
        rtrn_val = 6371.01*acos(cosine_val)

    return rtrn_val


orders_df['distance_km'] = orders_df.apply(lambda rww:  distance_calcn_rough(rww['lat_seller'], rww['long_seller'], rww['lat_customer'], rww['long_customer']), axis=1)
#sum(pd.isnull(orders_df.distance_km))  #1267 - not as many nulls as I thought

orders_df['states_same_or_diff'] = orders_df.customer_state.astype(str) == orders_df.seller_state.astype(str)
orders_df['states_same_or_diff'] = orders_df.states_same_or_diff.astype(int)

orders_df['state_pair'] = orders_df.customer_state.astype(str) + '-' + orders_df.seller_state.astype(str)
#orders_df.info()




#look for impacts by month, yr-and-mo, ....  of the purchase
orders_df['purchase_mo'] = orders_df.order_purchase_timestamp.dt.month_name()
orders_df['purchase_yr_and_mo'] = orders_df.order_purchase_timestamp.dt.year + (orders_df.order_purchase_timestamp.dt.month-1)/12
orders_df['purchase_day_of_wk'] = orders_df.order_purchase_timestamp.dt.weekday_name


#look for impacts by month, yr-and-mo, ....  of the estimated delivery date
orders_df['est_delivery_mo'] = orders_df.order_estimated_delivery_date.dt.month_name()
orders_df['est_delivery_yr_and_mo'] = orders_df.order_estimated_delivery_date.dt.year + (orders_df.order_estimated_delivery_date.dt.month-1)/12
orders_df['est_delivery_day_of_wk'] = orders_df.order_estimated_delivery_date.dt.weekday_name

#these return integers, and I don't want the model to think these are ordinal values as I don't think that's the case
#orders_df['est_delivery_mo'] = orders_df.order_estimated_delivery_date.dt.month
#orders_df['est_delivery_yr_and_mo'] = orders_df.order_estimated_delivery_date.dt.year*100 + orders_df.order_estimated_delivery_date.dt.month
#orders_df['est_delivery_day_of_wk'] = orders_df.order_estimated_delivery_date.dt.weekday



#  hopefully we don't need these for good predictions, as they aren't known until late in the game.  Ideally make a good prediction up front at time of purchase
# --------------------------------------------------------------------------------------------------------------------------------------------------------------
# shipping limit missed
orders_df['shipping_limit_missed'] = (orders_df.order_delivered_carrier_date > orders_df.ship_limit_final).astype(int)
orders_df['shipping_limit_miss_amt'] = (orders_df.order_delivered_carrier_date - orders_df.ship_limit_final).dt.total_seconds()/86400


# days remaining?
def days_remaining(start_dt1, start_dt2, end_dt):
    '''Calc time delta between end_dt and min(start_dt1, start_dt2)
    All three inputs are expected to be datetimes.'''
    
    if start_dt1 <= start_dt2:
        start_dt = start_dt1
    else:
        start_dt = start_dt2
        
    days_remng = (end_dt - start_dt).total_seconds()/86400
    return days_remng

    
orders_df['days_remaining'] = orders_df.apply(lambda rww: days_remaining(rww['order_delivered_carrier_date'], rww['ship_limit_final'], rww['order_estimated_delivery_date']), axis=1)
#orders_df[['order_delivered_carrier_date', 'ship_limit_final', 'order_estimated_delivery_date', 'days_remaining']]


# end of "hopefully we don't need these" vars








# outlier and missing handling
# ----------------------------
# drop rows with missing values as some techniques are sensitive to this (logistic regression)
orders_df2 = orders_df.loc[ ( pd.notnull(orders_df.order_purchase_timestamp) ) &
                            ( pd.notnull(orders_df.order_delivered_customer_date) ) &
                            ( pd.notnull(orders_df.distance_km) )  &
                            ( pd.notnull(orders_df.order_approved_at) ) &
                            ( pd.notnull(orders_df.order_delivered_carrier_date) ) &
                            ( pd.notnull(orders_df.ttl_pd) )  ]
orders_df2.info()    #95,981   
    #96,476 w/ just requiring the 1st 2 fields (essential to getting fulfill duration) so the extra cols with pruning for nulls make very little difference

    #distance_km is the main one, but this only drops another ~500 rows, seems certain we'll need this as a predictor

for col in orders_df2.columns:
    nbr_nulls = sum(pd.isnull(orders_df2[col]))
    if nbr_nulls > 0:
        print(f'{col} has {nbr_nulls} nulls still.')

#product_ctgry_mfu has 1344 nulls still.     but we should be able to impute a value to this if necessary (leave all one hot encodings zero??) so just move on
        




# ============================
# Split into test and train
# ============================
#need to start doing EDA to look for important predictors, volumes, etc.
#  before doing that, split the data into training and test so that we are not data snooping into the test set.
#  need to do the EDA on just the training or it's cheating to consider the test set an unknown dataset

predicted_col = ['fulfill_duration']

RANDOM_SEED=42

orders_train, orders_test, y_train, y_test = train_test_split(orders_df2, orders_df2[predicted_col], test_size=0.25, random_state=RANDOM_SEED)


    


# =============================================================
# EDA for important trends, insights, and explanatory variables
# =============================================================

#orders_train.info()
#orders_train.select_dtypes(exclude=['datetime', 'object']).columns
#
#'fulfill_duration', 
# 'est_delivery_time_days', 'distance_km', 'approval_time_days', 
#
#'ttl_pd', 'ttl_price', 'ttl_freight', 'pmt_mthds_used', 'installments_used_ttl', 'payment_types_used', 
#'nbr_items', 'nbr_sellers', 'nbr_products', 'nbr_photos', 
#'ttl_wt', 'ttl_length', 'ttl_height', 'ttl_width
#   
#     'purchase_yr_and_mo', 'est_delivery_yr_and_mo', 
# 'lat_customer', 'long_customer', 'lat_seller', 'long_seller', 
#
#       'shipping_limit_miss_amt', 'days_remaining'],
#
#'states_same_or_diff',     'late_delivery_flag', 'shipping_limit_missed',
#numerical, but don't use in correlation analysis as I don't think the "order" of zips has meaning so correlation has no meaning:  customer_zip_code_prefix', 'seller_zip_code_prefix', 

#check these:   orders_train.select_dtypes(include=['datetime', 'object']).columns



#look at correlations, scatter plots of fulfill_duration vs. these vars
contnuous_vars = ['fulfill_duration', 'est_delivery_time_days', 'distance_km', 'approval_time_days', 'ttl_pd', 'ttl_price', 'ttl_freight', 'pmt_mthds_used', 'installments_used_ttl', 'payment_types_used', 'nbr_items', 'nbr_sellers', 'nbr_products', 'nbr_photos', 'ttl_wt', 'ttl_length', 'ttl_height', 'ttl_width', 'purchase_yr_and_mo', 'est_delivery_yr_and_mo', 'states_same_or_diff', 'lat_customer', 'long_customer', 'lat_seller', 'long_seller', 'shipping_limit_miss_amt', 'days_remaining', 'late_delivery_flag', 'shipping_limit_missed']
        
def corr_chart(df_corr, fig_size=(12,12), file_nm='plot-corr-map.pdf'):
    corr_matrix=df_corr.corr()
    #screen top half to get a triangle
    top = np.zeros_like(corr_matrix, dtype=np.bool)
    top[np.triu_indices_from(top)] = True
    fig=plt.figure()
    fig, ax = plt.subplots(figsize=fig_size)
    sns.heatmap(corr_matrix, mask=top, cmap='coolwarm', 
        center = 0, square=True, 
        linewidths=.5, cbar_kws={'shrink':.5}, 
        annot = True, annot_kws={'size': 9}, fmt = '.3f')           
    plt.xticks(rotation=90) # rotate variable labels on columns (x axis)
    plt.yticks(rotation=0) # use horizontal variable labels on rows (y axis)
    plt.title('Correlation Heat Map')   
    plt.savefig(file_nm, 
        bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
        orientation='portrait', papertype=None, format=None, 
        transparent=True, pad_inches=0.25, frameon=None)  
    
    return plt, corr_matrix

corr_plt, corr_mtrx = corr_chart(orders_train[contnuous_vars], (20,20), 'correlations_for_fulfill_duration.pdf')
 
corrs_w_fulfill_dur = corr_mtrx.iloc[0, ].reset_index()        
corrs_w_fulfill_dur.columns = ['predictor', 'corr_w_fulfill_dur']
corrs_w_fulfill_dur['corr_strength'] = abs(corrs_w_fulfill_dur.corr_w_fulfill_dur)
corrs_w_fulfill_dur = corrs_w_fulfill_dur.sort_values('corr_strength', ascending=False)

#remove these, as we want to predict earlier on, before these are known
corrs_w_fulfill_dur = corrs_w_fulfill_dur[ ~ corrs_w_fulfill_dur.predictor.isin(['late_delivery_flag', 'days_remaining', 'shipping_limit_missed', 'shipping_limit_miss_amt'])]
#ditto for 'approval_time_days',  but if this seems REALLY helpful might want to add it in as a second round "updated delivery" estimate
 
#might want to leave out lat_customer (esp.) as it's highly correlated with distance_km - ditto for other lat/long fields
list(corrs_w_fulfill_dur.head(25)['predictor'])
top_contnous_pred = ['distance_km', 'est_delivery_time_days', 'states_same_or_diff', 'ttl_freight', 'purchase_yr_and_mo', 'est_delivery_yr_and_mo', 'ttl_pd', 'ttl_wt', 'ttl_price', 'installments_used_ttl', 'ttl_height', 'nbr_sellers', 'ttl_length', 'nbr_photos', 'nbr_products', 'nbr_items', 'ttl_width', 'pmt_mthds_used', 'payment_types_used']
top_contnous_pred2 = ['lat_customer', 'long_customer', 'lat_seller', 'long_seller', 'approval_time_days']
 

#look at scatter plots
#for var in top_contnous_pred:
for var in top_contnous_pred2:
    fig=plt.figure()
    fig, ax = plt.subplots(figsize=(12,12))
    plt.title(f'Fulfill duration vs {var}')   
    plt.xlabel(var)
    plt.ylabel('Fulfill duration (days)')
    plt.scatter(orders_train[var], orders_train['fulfill_duration']) 





# look at categorical vars
# ========================
# do box plots of distros by categorical var when the var is low cardinality
      
catgrcl_vars = ['customer_zip_code_prefix', 'customer_city', 'customer_state', 'seller_id', 'seller_zip_code_prefix', 'seller_city', 'seller_state', 'state_pair', 'product_ctgry_mfu',  'payment_type_mfu', 'purchase_mo', 'purchase_yr_and_mo', 'purchase_day_of_wk', 'est_delivery_mo', 'est_delivery_day_of_wk']
# check this out??       'ship_limit_final', 

#orders_train[catgrcl_vars].nunique()
catgrcl_vars_lowC = ['customer_state', 'seller_state', 'product_ctgry_mfu',  'payment_type_mfu', 'purchase_mo', 'purchase_yr_and_mo', 'purchase_day_of_wk', 'est_delivery_mo', 'est_delivery_day_of_wk']
catgrcl_vars_hiC = ['customer_zip_code_prefix', 'customer_city', 'seller_id', 'seller_zip_code_prefix', 'seller_city', 'state_pair']

for var in catgrcl_vars_lowC:
    orders_train.boxplot(column='fulfill_duration', by=var)


df_stats_by_var_and_lvl = pd.DataFrame()

for var in catgrcl_vars_hiC:
    lvl_cnts = []
    lvl_means = []
    lvl_std = []
    lvl_min = []
    lvl_max = []
    lvl_1qtle = []
    lvl_3qtle = []
    nbr_lvls = len(pd.unique(orders_train[var]))
    print(f'\nNow calculating stats for the {var} field; ({nbr_lvls} distinct values therein)')

    lvl_lst = list(pd.unique(orders_train[var]))
    for lvl in lvl_lst:
        df_lvl = orders_train.loc[orders_train[var]==lvl, [var, 'fulfill_duration']]
        lvl_cnts.append( len(df_lvl) )
        lvl_means.append( df_lvl['fulfill_duration'].mean() ) 
        lvl_std.append( df_lvl['fulfill_duration'].std() ) 
        lvl_min.append( df_lvl['fulfill_duration'].min() ) 
        lvl_max.append( df_lvl['fulfill_duration'].max() ) 
        lvl_1qtle.append( df_lvl['fulfill_duration'].quantile(q=0.25) ) 
        lvl_3qtle.append( df_lvl['fulfill_duration'].quantile(q=0.75) ) 

    #add results to a dataframe
    df_var = pd.DataFrame( {'variabl' : [var for i in range(nbr_lvls)], 'ctgry_lvl': lvl_lst, 'nbr':lvl_cnts, 'lvl_mean':lvl_means, 'lvl_std':lvl_std, 'lvl_min':lvl_min, 'lvl_max':lvl_max, 'lvl_1qtle':lvl_1qtle, 'lvl_3qtle':lvl_3qtle } )
    #add the dataframe for this var to the master df
    df_stats_by_var_and_lvl = df_stats_by_var_and_lvl.append(df_var)

df_lvl.shape


df_stats_by_var_and_lvl.head(20)

big_movers = df_stats_by_var_and_lvl[ (df_stats_by_var_and_lvl.nbr > 50 ) &
                                      (   ( df_stats_by_var_and_lvl.lvl_1qtle > np.mean(orders_train.fulfill_duration) )
                                        | ( df_stats_by_var_and_lvl.lvl_3qtle < np.mean(orders_train.fulfill_duration) )
                                      ) ]
#282 rows
big_movers['variabl'].value_counts()
#customer_city               88
#seller_zip_code_prefix      70
#seller_id                   69
#state_pair                  27
#seller_city                 22
#customer_zip_code_prefix     6



def flag_big_mover_record(var_name, level_name):
    rcds_fnd = big_movers[ ( big_movers['variabl'] ==var ) & 
                           ( big_movers['ctgry_lvl'] ==level_name ) ]

    bFnd = len(rcds_fnd) > 0 
    
    return int(bFnd)


#flag_big_mover_record('state_pair', 'MG-GO')


for var in pd.unique(big_movers['variabl']):
    df_plt = orders_train[[var, 'fulfill_duration']].copy()
    
    df_plt['grp_col'] = orders_train.apply(lambda rww: flag_big_mover_record(var, rww[var]), axis=1)



