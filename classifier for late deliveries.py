# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 15:00:00 2019

@author: ashley
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta, datetime
from math import pi, sin, cos, acos

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

wkg_dir = 'C:/Users/ashle/Documents/Personal Data/Northwestern/2019-04  fall MSDS498_Sec56 Capstone/final dataset/'


orders_df = pd.read_csv(wkg_dir+'Order_level_dataset.csv')
orders_df.head()
orders_df.info()
orders_df.order_estimated_delivery_date = pd.to_datetime(orders_df.order_estimated_delivery_date)  
orders_df.order_purchase_timestamp = pd.to_datetime(orders_df.order_purchase_timestamp)  
orders_df.order_approved_at = pd.to_datetime(orders_df.order_approved_at)  
orders_df.order_delivered_carrier_date = pd.to_datetime(orders_df.order_delivered_carrier_date)  
orders_df.order_delivered_customer_date = pd.to_datetime(orders_df.order_delivered_customer_date)  
orders_df.ship_limit_initial = pd.to_datetime(orders_df.ship_limit_initial)  
orders_df.ship_limit_final = pd.to_datetime(orders_df.ship_limit_final)  
orders_df.earliest_review_dt = pd.to_datetime(orders_df.earliest_review_dt)  
orders_df.latest_review_dt = pd.to_datetime(orders_df.latest_review_dt)  



#, format='%Y-%m-%d %H:%M:S')
#dt_cols <- c('order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date', 'order_delivered_customer_date', 'order_estimated_delivery_date', 'shipping_limit_date')
#ship_limit_initial, ship_limit_final
#earliest_review_dt, latest_review_dt


#Goal:   classifier model for "will we miss our estimated date" y/n, with prediction made after the earlier of:  max(shipping_limit_date) of any item 
#                                         or delivery to carrier has occurred  -->  at this point, we know if shiping limit has been missed or not (is known, can be used as a predictor).

#Features that are likely to be important:
# approval time > 1 day
# # of items
# estimated delivery time (in days)
# shipping limit missed
# distance
# days remaining?

# =================
#   Prep dataset
# =================
#missed_delivery_date_flag :   the predicted variable in this early warning predictor
orders_df['late_delivery_flag'] = (orders_df.order_delivered_customer_date >= orders_df.order_estimated_delivery_date + timedelta(days=1)).astype(int)
#sum(orders_df.late_delivery_flag)  #6535

# Create calc'd variables that are suspected to be valuable for prediction
#approval time
orders_df['approval_time_days'] = (orders_df.order_approved_at - orders_df.order_purchase_timestamp).dt.total_seconds()/86400

# estimated delivery time (in days)
orders_df['est_delivery_time_days'] = (orders_df.order_estimated_delivery_date - orders_df.order_purchase_timestamp).dt.total_seconds()/86400
orders_df['est_delivery_time_days'] = np.ceil(orders_df['est_delivery_time_days']).astype(int)

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



    
# distance
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
orders_df.info()



#check the odds of being late by levels of categorical vars
orders_df['purchase_mo'] = orders_df.order_purchase_timestamp.dt.month
orders_df['purchase_mo2'] = orders_df.order_purchase_timestamp.dt.month_name()
orders_df['purchase_yr_and_mo'] = orders_df.order_purchase_timestamp.dt.year*100 + orders_df.order_purchase_timestamp.dt.month
orders_df['purchase_yr_and_mo2'] = orders_df.order_purchase_timestamp.dt.year + (orders_df.order_purchase_timestamp.dt.month-1)/12

orders_df['purchase_day_of_wk'] = orders_df.order_purchase_timestamp.dt.weekday
orders_df['purchase_day_of_wk2'] = orders_df.order_purchase_timestamp.dt.weekday_name

orders_df['state_pair'] = orders_df.customer_state.astype(str) + '-' + orders_df.seller_state.astype(str)
orders_df['states_same_or_diff'] = orders_df.customer_state.astype(str) == orders_df.seller_state.astype(str)
orders_df['states_same_or_diff'] = orders_df.states_same_or_diff.astype(int)



# outlier and missing handling
# ----------------------------
# drop rows with missing values as some techniques are sensitive to this (logistic regression)
orders_df2 = orders_df.loc[ ( pd.notnull(orders_df.order_delivered_customer_date) ) &
                            ( pd.notnull(orders_df.distance_km) ) & ( pd.notnull(orders_df.days_remaining) ) &
                            ( pd.notnull(orders_df.order_purchase_timestamp) ) & ( pd.notnull(orders_df.order_approved_at) ) &
                            ( pd.notnull(orders_df.order_delivered_customer_date) ) & ( pd.notnull(orders_df.order_delivered_carrier_date) ) &
                            ( pd.notnull(orders_df.order_estimated_delivery_date) ) & ( pd.notnull(orders_df.ship_limit_final) ) &
                            ( pd.notnull(orders_df.nbr_items) ) & ( pd.notnull(orders_df.est_delivery_time_days) ) &
                            ( pd.notnull(orders_df.shipping_limit_missed) ) & ( pd.notnull(orders_df.shipping_limit_miss_amt) ) ]

#check the # null in the subset we care about (delivered orders)
#for col in orders_df.columns:
#    print(col, sum(pd.isnull(orders_df.loc[pd.notnull(orders_df.order_delivered_customer_date), col])))
#order_approved_at 14
#order_delivered_carrier_date 1
#payment_type_mfu 1
#ttl_pd 1
#pmt_mthds_used 1
#installments_used_ttl 1
#payment_types_used 1
#product_ctgry_mfu 1351
#lat_customer 264
#long_customer 264
#lat_seller 216
#long_seller 216
#approval_time_days 14
#shipping_limit_miss_amt 1
#days_remaining 0
#distance_km 479

#important fields
# 'late_delivery_flag',  
#'ship_limit_final', 'est_delivery_time_days', 'shipping_limit_missed', 'shipping_limit_miss_amt', 'days_remaining', 'distance_km', 
#'order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date', 'order_delivered_customer_date', 'order_estimated_delivery_date', 

#'nbr_items', 'nbr_sellers', 'nbr_products', 
#
#'ttl_wt', 'ttl_length', 'ttl_height', 'ttl_width', 
#'ttl_price', 'ttl_freight',
#'approval_time_days', 

#fields that are irrelevant/unnecessary/ok by defn etc
#filtering for late delivery flag being known should all remaining rows should be delivered or 6 records cancelled, so this field is taken care of
#'order_status', 
       
#impute nulls as avgs / 0's in 1hot encoding, so no need to drop nulls
#'customer_id', 'customer_unique_id', 'customer_zip_code_prefix', 'customer_city', 'customer_state',
#'seller_id', 'seller_zip_code_prefix', 'seller_city', 'seller_state'
#'state_pair', 'states_same_or_diff']
#'product_id_mfu', 'product_ctgry_mfu', 
 
      
#shouldn't be null if ship limit final is not null, and not planning to use
#, 'ship_limit_initial', 

       
#not going to use - reviews come in after delivery so they are influenced by delivery, not vice versa 
#'nbr_rws', 'avg_score', 'earliest_review_dt', 'latest_review_dt', 
 
# nulls here will cause null dists => after trimmng for distancekm, these should all be populated (and not planning to use in the model directly anyway)   
#'lat_customer', 'long_customer', 'lat_seller', 'long_seller', 
 
#not planning to use in the model anyway       
#'purchase_mo', 'purchase_mo2', 'purchase_yr_and_mo', 'purchase_yr_and_mo2', 'purchase_day_of_wk', 'purchase_day_of_wk2', 
#'payment_type_mfu', 'ttl_pd', 'pmt_mthds_used', 'installments_used_ttl', 'payment_types_used', 
#'nbr_photos', 

#orders_df2.shape   #95,982 rows.   not too much lower than max possible for our subset (delivered orders), which has 96,476 records
    #len(orders_df[pd.notnull(orders_df.order_delivered_customer_date)])

#orders_df2.info()
#pd.value_counts(orders_df2.order_status)
#95k of them are delivered; 6 are cancelled.  shouldn't matter to delivery calcs as we've ensured that delivery did happen



# ============================
# Split into test and train
# ============================
#need to start doing EDA to look for important predictors, volumes, etc.
#  before doing that, split the data into training and test so that we are not data snooping into the test set.
#  need to do the EDA on just the training or it's cheating to consider the test set an unknown dataset

predicted_col = ['late_delivery_flag']

RANDOM_SEED=42

orders_train, orders_test, y_train, y_test = train_test_split(orders_df2, orders_df2[predicted_col], test_size=0.25, random_state=RANDOM_SEED)


    



# =============================================================
# EDA for important trends, insights, and explanatory variables
# =============================================================

#look at boxplot of these vars by late_deliv yes/no
contnuous_vars = ['shipping_limit_miss_amt', 'days_remaining', 'distance_km', 'approval_time_days', 'ttl_wt', 'est_delivery_time_days', 'ttl_height', 'ttl_length', 'ttl_width']

#orders_train.select_dtypes(exclude=['datetime', 'object']).columns
#orders_train.select_dtypes(include=['datetime', 'object']).columns
contnuous_vars_othr = ['ttl_pd', 'pmt_mthds_used', 'installments_used_ttl', 'payment_types_used', 'nbr_items', 'ttl_price', 'ttl_freight', 'lat_customer', 'long_customer', 'lat_seller', 'long_seller']
       

#checked elsewhere
'customer_zip_code_prefix', 'seller_zip_code_prefix'
'nbr_sellers', 'nbr_products', 'nbr_photos', 'ttl_wt', 'ttl_length', 'ttl_height', 'ttl_width', 'nbr_rws', 'late_delivery_flag', 'approval_time_days', 'est_delivery_time_days', 'shipping_limit_missed', 'shipping_limit_miss_amt', 'days_remaining', 'distance_km', 'purchase_mo', 'purchase_yr_and_mo', 'purchase_yr_and_mo2',
       'purchase_day_of_wk', 'states_same_or_diff'
       'avg_score', 
orders_train.info()

for var in contnuous_vars:
    fig1, ax1 = plt.subplots(figsize=(16,9))
    df_plt = orders_train.loc[:, [var, 'late_delivery_flag']]
    #df_plt = df_plt.sort_values(by=['ctgry_lvl'])
    df_plt.boxplot(column=var, by='late_delivery_flag', ax=ax1)
    #df_plt.plot.bar(x='ctgry_lvl', y='odds', ax=ax1, rot=90)
    plt.title(var)
    ax1.set_title(var)


#try a histogram
for var in contnuous_vars:
    fig1, ax1 = plt.subplots(figsize=(16,9))
    df_plt = orders_train.loc[:, [var, 'late_delivery_flag']]
    #df_plt = df_plt.sort_values(by=['ctgry_lvl'])
    df_plt.hist(column=var, by='late_delivery_flag', ax=ax1)
    #df_plt.plot.bar(x='ctgry_lvl', y='odds', ax=ax1, rot=90)
    plt.title(var)
    ax1.set_title(var)
#the histogram isn't really working for me.  the scatter plot below is much clearer
    
    

#try an plot of odds by percentile bucket
#rank the records by the desired dimension
#find the breakpoints for 2nd, 4th, 6th, ...100th percentile
#calculate the odds in each bucket
# scatterplot them, using midpoint of range for X, odds for Y    
for var in contnuous_vars_othr:
#for var in contnuous_vars:
#    var = 'shipping_limit_miss_amt'
    if var == 'shipping_limit_miss_amt':
        df_new = orders_train.loc[orders_train.shipping_limit_miss_amt > -60, [var, 'late_delivery_flag']]
    else:
        df_new = orders_train[[var, 'late_delivery_flag']]

    ntiles = [cutoff/100 for cutoff in range(0,102,2)]
    cuts, bins = pd.qcut(df_new[var], ntiles, retbins=True, duplicates='drop')
#    type(cuts)  #series
#    type(cuts[211])  #pandas._libs.interval.Interval
    
    df_new = pd.concat([df_new, cuts], axis='columns')
  
    newcols = list(df_new.columns[:-1])
    newcols.append('bin')
    df_new.columns = newcols

    df_ttls_by_bin = df_new.groupby('bin').agg(
            nbr=('late_delivery_flag', 'count'),
            nbr_late=('late_delivery_flag', 'sum'))
    df_ttls_by_bin = df_ttls_by_bin.reset_index()
    df_ttls_by_bin['pct_late'] = df_ttls_by_bin.nbr_late / df_ttls_by_bin.nbr
    df_ttls_by_bin['bin_midpt'] = df_ttls_by_bin.bin.apply(lambda val:  val.mid)    
    df_ttls_by_bin['bin_midpt'] = df_ttls_by_bin.bin_midpt.astype(float)
    fig1, ax1 = plt.subplots(figsize=(16,9))
    df_ttls_by_bin.plot.scatter(x='bin_midpt', y='pct_late', ax=ax1, rot=45)
    plt.title(var)
    ax1.set_title(var)
    plt.xlabel(var)
    
#'ttl_pd', 

#    'ttl_price', 'ttl_freight' also show positive correlation but presumably this is not diff enough to put them in, just put in ttl_pd???
#    'lat_customer', 'long_customer', 'lat_seller' also have some correlation but presumably distance_km is a better indicator?
#'installments_used_ttl' has some positive correlation but the slope is small and i'm almost positive there are very few pts other than 1
    
    
    
df_ttls_by_bin.info()
df_ttls_by_bin.head()
    



#followups:
#look at cases where the order was delivered to the carrier more than 20 days earlier than the limit --why was the limit so high???
way_early = orders_train[orders_train.shipping_limit_miss_amt < -20][['order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date', 'order_estimated_delivery_date', 'ship_limit_final', 'shipping_limit_miss_amt', 'order_delivered_customer_date', 'est_delivery_time_days', 'shipping_limit_missed']]
#some of these look like typos?  Maybe many of these products were on back order??






catgrcl_vars = ['customer_zip_code_prefix', 'customer_city', 'customer_state', 'seller_id', 'seller_zip_code_prefix', 'seller_city', 'seller_state',  'product_ctgry_mfu', 'shipping_limit_missed', 'nbr_sellers', 'nbr_products',  'nbr_photos', 'purchase_mo', 'purchase_yr_and_mo', 'purchase_day_of_wk', 'state_pair', 'states_same_or_diff']


df_odds_by_var_and_lvl = pd.DataFrame()

for var in catgrcl_vars:
    cnt_lvl = []
    cnt_lvl_and_late = []
    nbr_lvls = len(pd.unique(orders_train[var]))
    print(f'\nNow calculating odds for the {var} field; ({nbr_lvls} distinct values therein)')

    for lvl in orders_train[var].unique():
        cnt_lvl.append( sum(orders_train[var] == lvl) ) 
        cnt_lvl_and_late.append(  sum( (orders_train[var] == lvl) & (orders_train.late_delivery_flag == 1) )   )

    #add results to a dataframe
    df_var = pd.DataFrame( {'variabl' : [var for i in range(nbr_lvls)], 'ctgry_lvl': [lvl for lvl in pd.unique(orders_train[var])], 'nbr':cnt_lvl, 'nbr_late':cnt_lvl_and_late } )
    #add the dataframe for this var to the master df
    df_odds_by_var_and_lvl = df_odds_by_var_and_lvl.append(df_var)
    
    
#add in addnl categorical variables, if added later      (ex. DayOfWk, see if it shows anything interesting    
#add in samestate vs OutOfState;  see if it shows anything interesting    
#
#df_odds_by_var_and_lvl = df_odds_by_var_and_lvl_bkup.copy()
#df_odds_addnl_vars = pd.DataFrame()
#for var in ['state_pair', 'states_same_or_diff']:     #, 'purchase_day_of_wk']:
#    cnt_lvl = []
#    cnt_lvl_and_late = []
#    nbr_lvls = len(pd.unique(orders_train[var]))
#
#    for lvl in orders_train[var].unique():
#        cnt_lvl.append( sum(orders_train[var] == lvl) ) 
#        cnt_lvl_and_late.append(  sum( (orders_train[var] == lvl) & (orders_train.late_delivery_flag == 1) )   )
#
#    #add results to a dataframe
#    df_var = pd.DataFrame( {'variabl' : [var for i in range(nbr_lvls)], 'ctgry_lvl': [lvl for lvl in pd.unique(orders_train[var])], 'nbr':cnt_lvl, 'nbr_late':cnt_lvl_and_late } )
#    #add the dataframe for this var to the master df
#    df_odds_addnl_vars = df_odds_addnl_vars.append(df_var)
#
#
#df_odds_addnl_vars['odds'] = df_odds_addnl_vars.nbr_late / df_odds_addnl_vars.nbr
#df_odds_addnl_vars['variabl'] = df_odds_addnl_vars.variabl.astype(str)
#df_odds_addnl_vars['ctgry_lvl_str'] = df_odds_addnl_vars.ctgry_lvl.astype(str)
#
#df_odds_by_var_and_lvl = df_odds_by_var_and_lvl.append(df_odds_addnl_vars)


#make a backup
df_odds_by_var_and_lvl_bkup = df_odds_by_var_and_lvl.copy()

#checks
#df_odds_by_var_and_lvl.info()
#df_odds_by_var_and_lvl.head()

#calculate odds, and convert (esp. ctgry_lvls) to string so plotting is ok with it
df_odds_by_var_and_lvl['odds'] = df_odds_by_var_and_lvl.nbr_late / df_odds_by_var_and_lvl.nbr
df_odds_by_var_and_lvl['variabl'] = df_odds_by_var_and_lvl.variabl.astype(str)
df_odds_by_var_and_lvl['ctgry_lvl_str'] = df_odds_by_var_and_lvl.ctgry_lvl.astype(str)



#check counts by variabl (field)
df_odds_by_var_and_lvl.groupby('variabl').agg(NbrLvls=('ctgry_lvl_str', 'nunique'))
        


#plot the ones that have a limited # of choices and can be read clearly on a graph
for var in [vrbl for vrbl in catgrcl_vars if vrbl not in ['customer_city', 'seller_city', 'customer_zip_code_prefix', 'seller_zip_code_prefix', 'seller_id', 'state_pair']]:
    fig1, ax1 = plt.subplots(figsize=(16,9))
    df_plt = df_odds_by_var_and_lvl.loc[df_odds_by_var_and_lvl.variabl==var, ['ctgry_lvl', 'ctgry_lvl_str', 'odds']]
    df_plt = df_plt.sort_values(by=['ctgry_lvl'])
    #df_plt.boxplot(column='odds', by='ctgry_lvl_str', ax=ax1)
    df_plt.plot.bar(x='ctgry_lvl', y='odds', ax=ax1, rot=90)
    plt.title(var)
    ax1.set_title(var)
    

#review the results for variables, and levels thereof, that seem to have predictive value
##two seller states that are notably worse  
#df_odds_by_var_and_lvl.loc[df_odds_by_var_and_lvl.variabl=='seller_state', ['ctgry_lvl_str', 'odds']].sort_values(by=['odds'], ascending=False)
#df_seller_state = df_odds_by_var_and_lvl.loc[df_odds_by_var_and_lvl.variabl=='seller_state', ].sort_values(by=['odds'], ascending=False)
##AM  (but only 3 orders; missed 1 out of 3), MA (only 392 ttl, still very samll)    biggest is SP, and that's about 4th highest % late, but not exceptionally higher
#
##shipping limit missed makes a big diff - should already be in the model
##nbr sellers seems to make an INVERSE diff.   but not much diff (or much data) for 2+, and certainly not linear.   group into categories - single seller; multiple seller
##nbr products seems to make an INVERSE diff.   fairly linear down to 5+     group into categories - one, two, three, four, five+/nan
##purchase mo (do with Jan/Feb, not 0,1, to make it categorical not continuous)
#
##out of state sends are more likely to be late, as expected
#df_odds_by_var_and_lvl.loc[df_odds_by_var_and_lvl.variabl=='states_same_or_diff', ]
#
##which pairs are most problematic?
#df_odds_by_var_and_lvl.loc[df_odds_by_var_and_lvl.variabl=='state_pair', ].sort_values(by=['odds'], ascending=False)
#
#
#df_odds_by_var_and_lvl.loc[df_odds_by_var_and_lvl.variabl=='customer_state', ['ctgry_lvl_str', 'odds']].sort_values(by=['odds'], ascending=False)
#AL, MA


#don't need to one-hot; already is binary
#states_same_or_diff, shipping_limit_missed

#purchase_yr_and_mo, purchase_mo
#don't generalize well

#don't make sense, and too thin to be much good
#nbr_photos

#already in the model
#nbr_sellers
#nbr_products
#shipping_limit_missed



#global avg % late
len(orders_train[orders_train.late_delivery_flag == 1])  #6535 in the entire set, 4913 in the training set
len(orders_train)  #99441 in the entire set, 71,986 in the training set
#pct_late_avg = 6535/99441  #6.57%
pct_late_avg = 4913/71986   #6.8%

pct_late_upper_threshold = pct_late_avg * 1.25
pct_late_lower_threshold = pct_late_avg * 0.75
pct_late_upper_threshold2 = pct_late_avg * 1.5
pct_late_lower_threshold2 = pct_late_avg * 0.5


impactful_lvls = df_odds_by_var_and_lvl.loc[ ( df_odds_by_var_and_lvl.odds > pct_late_upper_threshold )  |  ( df_odds_by_var_and_lvl.odds < pct_late_lower_threshold ) ]
impactful_lvls2 = impactful_lvls.loc[impactful_lvls.variabl.isin(['customer_state', 'seller_state', 'product_ctgry_mfu']), ] 
impactful_lvls3 = impactful_lvls2[impactful_lvls2.nbr >= 37]
#these seem worth one-hot encoding.   45 records->45 new predictor cols ==> hopefully not much overfitting on 70k records  the 37 used to be 50, but since we took a 75/25 split, I scaled it back


#FORGET THIS WHOLE ANALYSIS.  Seems stupid to expect that a state will stand out in combo that doesn't already stand out due to outlying performance for either custs or sellers or both
#      actually this will be LESS sensitive as states with one side (cust or seller) extreme and the other normal or extreme the other way, will get averaged towards the middle
#check if there are any problem states that seem to always raise the odds of late delivery (whether the seller is there or the customer is)
#------------------------------------------------------------------------------------------------------------------------------------------
##first approach, don't like it much anymore as the columns don't really help get the odds (requires row analysis)
#df_problem_state_chk = pd.DataFrame()
#state_set = set(pd.unique(orders_train.seller_state))
#state_set = state_set.union(pd.unique(orders_train.customer_state))
#
#for state in state_set:
#    col_name = 'Involves_' + str(state)
#    df_new = pd.DataFrame({col_name: (orders_train.seller_state == state) | (orders_train.customer_state == state)})
#    df_problem_state_chk = pd.concat([df_problem_state_chk, df_new], axis='columns')
#
#df_problem_state_chk.info()

#approach2) get the odds for deliveries involving that state
#df_problem_state2 = pd.DataFrame()
#
#for state in state_set:
#    nbr = sum(  (orders_train.seller_state == state) | (orders_train.customer_state == state)  )
#    nbr_late = sum(  ( (orders_train.seller_state == state) | (orders_train.customer_state == state) ) & (orders_train.late_delivery_flag == 1)  )
#    if nbr != 0:
#        odds = nbr_late / nbr
#    else:
#        odds = -999
##    print(state, nbr, nbr_late, odds)
#    df_new = pd.DataFrame(np.array([[state, nbr, nbr_late, odds]]), columns=['state', 'nbr', 'nbr_late', 'odds'])    
#    df_problem_state2 = df_problem_state2.append(df_new)
#
#df_problem_state2.odds = df_problem_state2.odds.astype(float)
#
#df_plt = df_problem_state2.iloc[1:, ].copy()
#df_plt.odds = df_plt.odds.astype(float)
#fig1, ax1 = plt.subplots(figsize=(16,9))
#df_plt.plot.bar(x='state', y='odds', ax=ax1, rot=90)
#END OF FORGET THIS section



#check yr-and-mo
df_plt_prchs = df_odds_by_var_and_lvl.loc[df_odds_by_var_and_lvl.variabl=='purchase_yr_and_mo', ['ctgry_lvl_str', 'odds']]
df_plt_prchs['ctgry_lvl_str'] = df_plt_prchs.ctgry_lvl_str.astype(int)
df_plt_prchs['prch_yr_and_mo'] = round(df_plt_prchs.ctgry_lvl_str/100, 0) + (df_plt_prchs.ctgry_lvl_str % 100)/12
df_plt_prchs.plot.scatter(x='prch_yr_and_mo', y='odds')
#no discernible trend or seasonal pattern  ==> don't bother with this as a predictor.  won't be useful on new data from diff months.


#review the fields with really high cardinality, using histograms
#  (but only focus on the levels with more than 100 records  (75 in the training set)
impactful_hi_card_fields = pd.DataFrame()
for var in ['customer_city', 'seller_city', 'customer_zip_code_prefix', 'seller_zip_code_prefix', 'seller_id', 'state_pair']:
    df_plt = df_odds_by_var_and_lvl.loc[ (df_odds_by_var_and_lvl.variabl==var) & (df_odds_by_var_and_lvl.nbr >= 75) , ['variabl', 'ctgry_lvl_str', 'odds', 'nbr', 'nbr_late']]
    df_plt = df_plt.reset_index()
    df_plt = df_plt.sort_values(by=['odds'], ascending=False)
    fig1, ax1 = plt.subplots()
    ax1.hist(df_plt.odds)
    plt.title(var)
    ax1.set_title(var)
    impactful_hi_card_fields = impactful_hi_card_fields.append(df_plt[ ( df_plt.odds > pct_late_upper_threshold )  |  ( df_plt.odds < pct_late_lower_threshold ) ].copy() )

impactful_hi_card_fields['major_impact_flag']
impactful_hi_card_fields['major_impact_flag'] = ( (impactful_hi_card_fields.odds > pct_late_upper_threshold2 )  |  ( impactful_hi_card_fields.odds < pct_late_lower_threshold2 ) ).astype(int)

sum(impactful_hi_card_fields.major_impact_flag)  #281 with "major" impact - start with those so we don't overspecify the model


df_odds_by_var_and_lvl = df_odds_by_var_and_lvl.sort_values(by=['odds'], ascending=False)
#customer zip code has a big outlier
    #the rest of the metrics are more normal shaped-  some right skew but not too much
bad_cust_zips = df_odds_by_var_and_lvl.loc[ (df_odds_by_var_and_lvl.variabl=='customer_zip_code_prefix') & (df_odds_by_var_and_lvl.nbr >= 10) , ]


# purchase_yr_and_mo:   
#     trend doesn't show any clear pattern
#     seasonality month of year doesn't appear to be a consistent driver of high or low




# make buckets for 'ttl_pd',    price buckets?  freight buckets?  try corr vs late to see if it's worthwhile?
#       'payment_type_mfu', 'pmt_mthds_used', 'installments_used_ttl', 'payment_types_used', 'nbr_items', 'ttl_price', 'ttl_freight',

#mo of order purchase?  wk of yr??  ex. more delays around christmas/holiday/other?


#?? 'product_id_mfu', 
 
# should be the same as seller_id I believe 'seller_id_mfu',
# seems doubtful that these will drive late deliveries in a way that is clustered/diff from/in addition to the effect of distance, which we've already put in the model
# 'lat_customer', 'long_customer', 'lat_seller', 'long_seller',   

#predicted var  (nothting to check) 'late_delivery_flag', 

#continuous vars - check corr vs. late_flag?   figure out odds somehow???
#, 'ttl_wt', 'ttl_length', 'ttl_height', 'ttl_width', 'approval_time_days',
# 'est_delivery_time_days', 'shipping_limit_miss_amt', 'days_remaining', 'distance_km'


#reviews generally aren't provided until after delivery.  hence, reviews aren't driving delivery (is the other way around) and even if they were, they wouldn't be known in time for use in predicting late arrivals
# 'nbr_rws', 'avg_score', 'earliest_review_dt', 'latest_review_dt', 

    #sum(orders_train.earliest_review_dt > orders_train.order_delivered_customer_date)
    #91574     
    
    #sum(orders_train.earliest_review_dt < orders_train.order_delivered_customer_date)
    #4902 


#orders_train.drop('distance_rough', axis='columns', inplace=True)




# ==================
# prep the X dataset
# ==================
#  remove the predicted column (was just kept in for use in EDA)
#   and remove cols that don't seem useful as predictors
#   and add onehot encoders for categoricals that seem to be valuable.
    #check the Y dataset but I think it's ok
    
predictor_cols_basic = ['est_delivery_time_days', 'shipping_limit_missed', 'shipping_limit_miss_amt', 'days_remaining', 'distance_km', 'approval_time_days', 'nbr_items',
                  'nbr_sellers', 'nbr_products', 'ttl_pd', 'ttl_wt', 'ttl_length', 'ttl_height', 'ttl_width', 'states_same_or_diff']

ohe_biggest_vars = impactful_lvls3[ (impactful_lvls3.odds > pct_late_upper_threshold2 )  |  ( impactful_lvls3.odds < pct_late_lower_threshold2 ) ]
hi_card_major_predictors = impactful_hi_card_fields[ impactful_hi_card_fields.major_impact_flag == 1 ]
ohe_biggest_vars = ohe_biggest_vars.append( hi_card_major_predictors )


x_train = orders_train.copy()

#create dummy columns
predictor_cols_ohe = []
for rww in ohe_biggest_vars.itertuples(index=False):   
    col_new = rww.variabl + '_' + rww.ctgry_lvl_str
    predictor_cols_ohe.append(col_new)
    
    df_new = x_train.loc[:,[rww.variabl]].copy() 
    df_new[col_new] = ( df_new[rww.variabl] == rww.ctgry_lvl_str ).astype(int) 
  
    df_new.drop(rww.variabl, axis='columns', inplace = True)
    x_train = pd.concat([x_train, df_new], axis='columns')



#create similar columns in the test set
x_test = orders_test.copy()

#create dummy columns
for rww in ohe_biggest_vars.itertuples(index=False):   
    col_new = rww.variabl + '_' + rww.ctgry_lvl_str
    
    df_new = x_test.loc[:,[rww.variabl]].copy() 
    df_new[col_new] = ( df_new[rww.variabl] == rww.ctgry_lvl_str ).astype(int) 
  
    df_new.drop(rww.variabl, axis='columns', inplace = True)
    x_test = pd.concat([x_test, df_new], axis='columns')
    

len(predictor_cols_ohe)  #298





#'seller_city', 'seller_state', 'product_ctgry_mfu', 
#two seller states that are notably worse    AM  (but only 3 orders; missed 1 out of 3), MA (only 392 ttl, still very samll)    
    #biggest is SP, and that's about 4th highest % late, but not exceptionally higher

#shipping limit missed makes a big diff - should already be in the model
#nbr sellers seems to make an INVERSE diff.   but not much diff (or much data) for 2+, and certainly not linear.   group into categories - single seller; multiple seller
#nbr products seems to make an INVERSE diff.   fairly linear down to 5+     group into categories - one, two, three, four, five+/nan
#purchase mo (do with Jan/Feb, not 0,1, to make it categorical not continuous)



#x_train.columns
#  Other possibles:     'ship_limit_final', 'ship_limit_initial',  -- these were a maybe, but the model didn't like timestamp as an argument 
#                         'payment_type_mfu', 'ttl_pd', 'pmt_mthds_used', 'installments_used_ttl', 'payment_types_used', 'ttl_price', 'ttl_freight',
#                       'nbr_photos', 'order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date',     'order_delivered_customer_date', 'order_estimated_delivery_date', 'customer_unique_id', 'customer_zip_code_prefix', 'customer_city', 'customer_state',







#from sklearn.preprocessing import OneHotEncoder
#from sklearn.compose import ColumnTransformer
#state_cats = list(state_set)
#purchase_mo_cats = pd.unique(orders_train.purchase_mo2)
#purchase_dow_cats = pd.unique(orders_train.purchase_day_of_wk2)
#state_pair_cats = pd.unique(orders_train.state_pair)
#ohe_seller_states = OneHotEncoder(handle_unknown='ignore', categorical_features=['seller_state', 'purchase_mo2', 'purchase_day_of_wk2'])
#df_prd = x_train[predictor_cols].copy()
#ohe_seller_states.fit_transform(df_prd)
#transfrmr = ColumnTransformer(
#                [  ('catgrcal', OneHotEncoder(handle_unknown='ignore'), ['seller_state'])    ]
#                                )
#x_tranfrmd = transfrmr.fit_transform(df_prd)
#type(x_tranfrmd)




# Create models on training
# Validate on test set

mdl_lr = LogisticRegression(random_state = RANDOM_SEED, C=1)

#fit the model
mdl_lr.fit(x_train, y_train)

#score it on the test dataset
mdl_lr.score(x_test, y_test)
#0.9298   

#get the predicted late flags as per the model
y_predicted_lr = mdl_lr.predict(x_test)


#calculate precision-recall score (avg?), curves
precision_lr, recall_lr, _ = precision_recall_curve(y_test, y_predicted_lr)

average_precision_lr = average_precision_score(y_test, y_predicted_lr)
print('Average precision-recall score: {0:0.2f}'.format(average_precision_lr))
#0.12   #  :(

#plot precision/recall curve
# ----------------------------
# In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
step_kwargs = ( {'step': 'post'} if 'step' in signature(plt.fill_between).parameters else {} )

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall curve:  Logistic Regression.   AP={0:0.2f}'.format(average_precision_lr))
plt.step(recall_lr, precision_lr, color='b', alpha=0.2, where='post')
plt.fill_between(recall_lr, precision_lr, alpha=0.2, color='b', **step_kwargs)


# look at the confusion matrix
# ----------------------------
#type(y_test)  #df
actl_and_predicted = y_test.copy()
actl_and_predicted['predicted_late_flag'] = y_predicted_lr
actl_and_predicted.columns = ['actual_late_flag', 'predicted_late_flag']

actl_and_predicted.groupby(['actual_late_flag', 'predicted_late_flag']).agg(cnt=('actual_late_flag', 'count')).reset_index()
#   actual_late_flag  predicted_late_flag    cnt
#                 0                    0  22021
#                 0                    1     26
#                 1                    0   1502
#                 1                    1    111



#withOUT ship_limit_miss_amt, the model is lot worse:
#   actual_late_flag  predicted_late_flag    cnt
#                 0                    0  21982
#                 0                    1     23
#                 1                    0   1637
#                 1                    1     18



# --------------------------------------------
#   try a random forest classifier
# --------------------------------------------
x_train_pc1 = x_train.loc[pd.notnull(x_train.ttl_pd), predictor_cols_basic]
y_train_pc1 = y_train.loc[pd.notnull(x_train.ttl_pd), ]

x_test_pc1 = x_test.loc[pd.notnull(x_test.ttl_pd), predictor_cols_basic]
y_test_pc1 = y_test.loc[pd.notnull(x_test.ttl_pd), ]


mdl_rf1 = RandomForestClassifier(random_state=RANDOM_SEED)
mdl_rf1.fit(x_train_pc1, y_train_pc1)
mdl_rf1.score(x_train_pc1, y_train_pc1)   #0.988
mdl_rf1.score(x_test_pc1, y_test_pc1)  #0.9398

#look at the confusion matrix
y_pred_rf1 = mdl_rf1.predict(x_test_pc1)
actl_and_pred_rf1 = y_test_pc1.copy()
actl_and_pred_rf1['predicted_late_flag'] = y_pred_rf1
actl_and_pred_rf1.columns = ['actual_late_flag', 'predicted_late_flag']

actl_and_pred_rf1.groupby(['actual_late_flag', 'predicted_late_flag']).agg(cnt=('actual_late_flag', 'count')).reset_index()
# model is better with ship limit miss amt (correctly about 2x as many true lates, and also has fewer false positives )
#   actual_late_flag  predicted_late_flag    cnt
#                 0                    0  22352
#                 0                    1     67
#                 1                    0   1378
#                 1                    1    199

#old (previous test/train split)
#   actual_late_flag  predicted_late_flag    cnt
#                 0                    0  21987
#                 0                    1     60
#                 1                    0   1431
#                 1                    1    182
#vs prev result, without ship limit miss amt:
#   actual_late_flag  predicted_late_flag    cnt
#                 0                    0  21927
#                 0                    1     78
#                 1                    0   1563
#                 1                    1     92


#calculate precision-recall score (avg?), curves
precision_rf1, recall_rf1, _ = precision_recall_curve(y_test_pc1, y_pred_rf1)

average_precision_rf1 = average_precision_score(y_test_pc1, y_pred_rf1)
print('Average precision-recall score: {0:0.2f}'.format(average_precision_rf1))
#0.15   #  :(

#plot precision/recall curve
# ----------------------------
# In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
step_kwargs = ( {'step': 'post'} if 'step' in signature(plt.fill_between).parameters else {} )

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall curve:  Random Forest.   AP={0:0.2f}'.format(average_precision_rf1))
plt.step(recall_rf1, precision_rf1, color='b', alpha=0.2, where='post')
plt.fill_between(recall_rf1, precision_rf1, alpha=0.2, color='b', **step_kwargs)

#get feature importances
feature_importances_rf1 = pd.DataFrame( {'Feature_name': list(x_train_pc1.columns), 'Importance': list(mdl_rf1.feature_importances_) } )
feature_importances_rf1 = feature_importances_rf1.sort_values('Importance', ascending=False)
feature_importances_rf1.head(100)
feature_importances_rf1.info()



roc_auc_val_rf1 = roc_auc_score(y_test_pc1, y_pred_rf1)
print(roc_auc_val_rf1)  #0.562

fpr_rf1, tpr_rf1, thresholds_rf1 = roc_curve(y_test_pc1, y_pred_rf1)


#plot the ROC curve
plt.plot(fpr_rf1, tpr_rf1, lw=2, label='ROC curve for Random Forest model (area = %0.2f)' % roc_auc_val_rf1)
plt.plot([0,1],[0,1],color='grey', linestyle='--', lw=1)
plt.legend(loc="lower right", frameon = False)
plt.title('ROC curve - Random Forest model')
plt.savefig(wkg_dir + '/ROC_rf1.jpg', 
    bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
    orientation='portrait', papertype=None, format=None, 
    transparent=True, pad_inches=0.25, frameon=None)  
plt.show()











# ================================================
#  re-train with undersampled data 
#        use all the late delivery records, but a much lower % (same # as the late deliv records) from the ontime delivery so that our training set is balanced
#        hopefully that will make it more sensitive 
# ================================================

nbr_late_delivs = sum(x_train.late_delivery_flag)  #6394

#bring along the Rww_ID temporarily so that we can use it to create the validation set
predictor_cols2 = predictor_cols.copy()
predictor_cols2.append('Rww_ID') 
predicted_col2 = predicted_col.copy()
predicted_col2.append('Rww_ID')
#create the undersampled (us) dataset
#take a 25% sample of the records where delivery was late    ==> we should have about 6394*0.75 (=4795.5) records in training
x_train_us, x_test_us, y_train_us, y_test_us = train_test_split(x_train.loc[x_train.late_delivery_flag == True, predictor_cols2], 
                                                                x_train.loc[x_train.late_delivery_flag == True, predicted_col2],
                                                                test_size = 0.25)
#x_train_us.shape  #  (4795, 13)     good, matches expected size
#x_test_us.shape  #  (1599, 13)     good, matches expected size

x_train_us_0, x_test_us_0, y_train_us_0, y_test_us_0 = train_test_split(x_train.loc[x_train.late_delivery_flag == False, predictor_cols2],
                                                                        x_train.loc[x_train.late_delivery_flag == False, predicted_col2],
                                                                        train_size = 4795, test_size = 1599) 
#x_train_us_0.shape  #  (4795, 13)     good, matches expected size
#x_test_us_0.shape  #  (1599, 13)     good, matches expected size
#sum(y_train_us_0.late_delivery_flag)

#merge the samples for late and ontime together
x_train_us = x_train_us.append(x_train_us_0)
y_train_us = y_train_us.append(y_train_us_0)

#x_train.info()
#for the validation, run on all the records that were not used in training, or on these test records???
x_test_us = x_test_us.append(x_test_us_0)
y_test_us = y_test_us.append(y_test_us_0)

x_valdn_us = x_train.loc[ ~ x_train.Rww_ID.isin(x_train_us.Rww_ID) , predictor_cols]
y_valdn_us = x_train.loc[ ~ x_train.Rww_ID.isin(y_train_us.Rww_ID) , predicted_col]
    #x_valdn_us.shape  # 85048, 13
    #y_valdn_us.shape  # 85048, 1


#drop the Rww_ID so it doesn't get fed to the model
x_train_us.drop('Rww_ID', axis='columns', inplace = True)
x_test_us.drop('Rww_ID', axis='columns', inplace = True)
y_train_us.drop('Rww_ID', axis='columns', inplace = True)
y_test_us.drop('Rww_ID', axis='columns', inplace = True)




#fit and score a Random Forest classifier
mdl_rf2_us = RandomForestClassifier(n_estimators = 100, random_state=RANDOM_SEED)
mdl_rf2_us.fit(x_train_us, y_train_us)
mdl_rf2_us.score(x_test_us, y_test_us)   #0.662. up from 0.643 with n_estimators = 10 (default)
mdl_rf2_us.score(x_valdn_us, y_valdn_us)  #0.71  :( :(


y_pred_rf_us = mdl_rf2_us.predict(x_valdn_us)
actl_and_pred_rf_us = y_valdn_us.copy()
actl_and_pred_rf_us['predicted_late_flag'] = y_pred_rf_us
actl_and_pred_rf_us.columns = ['actual_late_flag', 'predicted_late_flag']

#look at the confusion matrix
actl_and_pred_rf_us.groupby(['actual_late_flag', 'predicted_late_flag']).agg(cnt=('actual_late_flag', 'count')).reset_index()
# model predicts a lot more of the actual late values, but is way overpredicting late values for the on-time deliveries
#     this would only make economic sense if the cost of a false positive was WAY less than the cost of a false negative
#  actual_late_flag  predicted_late_flag    cnt
#                 0                    0  60382
#                 0                    1  23067
#                 1                    0    667
#                 1                    1    932



#calculate precision-recall score (avg?), curves
precision_rf_us, recall_rf_us, _ = precision_recall_curve(y_valdn_us, y_pred_rf_us)

average_precision_rf_us = average_precision_score(y_valdn_us, y_pred_rf_us)
print('Average precision-recall score: {0:0.2f}'.format(average_precision_rf_us))
#0.15   #  :(

#plot precision/recall curve
# ----------------------------
# In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
step_kwargs = ( {'step': 'post'} if 'step' in signature(plt.fill_between).parameters else {} )

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall: Random Forest w Undersampling.   AP={0:0.2f}'.format(average_precision_rf_us))
plt.step(recall_rf_us, precision_rf_us, color='b', alpha=0.2, where='post')
plt.fill_between(recall_rf_us, precision_rf_us, alpha=0.2, color='b', **step_kwargs)
plt.savefig(wkg_dir + '/precision_recall_rf_us.jpg', 
    bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
    orientation='portrait', papertype=None, format=None, 
    transparent=True, pad_inches=0.25, frameon=None)  
plt.show()


roc_auc_val_rf_us = roc_auc_score(y_valdn_us, y_pred_rf_us)
print(roc_auc_val_rf_us)  #0.669, up from 0.643

fpr_rf_us, tpr_rf_us, thresholds_rf_us = roc_curve(y_valdn_us, y_pred_rf_us)


#plot the ROC curve
plt.plot(fpr_rf_us, tpr_rf_us, lw=2, label='ROC curve for Random Forest model (area = %0.2f)' % roc_auc_val_rf_us)
plt.plot([0,1],[0,1],color='grey', linestyle='--', lw=1)
plt.legend(loc="lower right", frameon = False)
plt.title('ROC curve - Random Forest model')
plt.savefig(wkg_dir + '/ROC_rf_us.jpg', 
    bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
    orientation='portrait', papertype=None, format=None, 
    transparent=True, pad_inches=0.25, frameon=None)  
plt.show()







#predictor_cols_basic = ['est_delivery_time_days', 'shipping_limit_missed', 'shipping_limit_miss_amt', 'days_remaining', 'distance_km', 'approval_time_days', 'nbr_items',
#                  'nbr_sellers', 'nbr_products', 'ttl_wt', 'ttl_length', 'ttl_height', 'ttl_width', 'states_same_or_diff']
#
#
predictor_cols = predictor_cols_basic + predictor_cols_ohe



# ---------------------------------------------------------------------------
#   random forest classifier with addnl columns ("major" categorical factors)
# ---------------------------------------------------------------------------
#TODO:   fix to exclude the row that is missing ttl_pd info, rename other vars _rf3
x_train_pc = x_train[predictor_cols]
x_test_pc = x_test[predictor_cols]


mdl_rf3 = RandomForestClassifier(random_state=RANDOM_SEED)
mdl_rf3.fit(x_train_pc, y_train)
mdl_rf3.score(x_train_pc, y_train)  #0.9885
mdl_rf3.score(x_test_pc, y_test)    #0.938
    #train is performing better than test - is overfitting.   --> adding more columns won't help, will only make it further overfit
    

y_pred_rf = mdl_rf3.predict(x_test_pc)
actl_and_pred_rf = y_test.copy()
actl_and_pred_rf['predicted_late_flag'] = y_pred_rf
actl_and_pred_rf.columns = ['actual_late_flag', 'predicted_late_flag']

#look at the confusion matrix
actl_and_pred_rf.groupby(['actual_late_flag', 'predicted_late_flag']).agg(cnt=('actual_late_flag', 'count')).reset_index()
# model is better with ship limit miss amt (correctly about 2x as many true lates, and also has fewer false positives )
#   actual_late_flag  predicted_late_flag    cnt
#                 0                    0  22324
#                 0                    1     95
#                 1                    0   1392
#                 1                    1    185

#this is WORSE! than the RF with no categorical columns  - has more false positives and the exact same # of false negatives
#   actual_late_flag  predicted_late_flag    cnt
#                 0                    0  22338
#                 0                    1     81
#                 1                    0   1392
#                 1                    1    185



#calculate precision-recall score (avg?), curves
precision_rf, recall_rf, _ = precision_recall_curve(y_test, y_pred_rf)

average_precision_rf = average_precision_score(y_test, y_pred_rf)
print('Average precision-recall score: {0:0.2f}'.format(average_precision_rf))
#0.14   #  :(

#plot precision/recall curve
# ----------------------------
# In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
step_kwargs = ( {'step': 'post'} if 'step' in signature(plt.fill_between).parameters else {} )

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall curve:  Random Forest.   AP={0:0.2f}'.format(average_precision_rf))
plt.step(recall_rf, precision_rf, color='b', alpha=0.2, where='post')
plt.fill_between(recall_rf, precision_rf, alpha=0.2, color='b', **step_kwargs)

#get feature importances
feature_importances_rf3 = pd.DataFrame( {'Feature_name': list(x_train_pc.columns), 'Importance': list(mdl_rf3.feature_importances_) } )
feature_importances_rf3 = feature_importances_rf3.sort_values('Importance', ascending=False)
feature_importances_rf3.head(100)
feature_importances_rf3.info()



#mdl_rf3.get_params()



roc_auc_val_rf = roc_auc_score(y_test, y_pred_rf)
print(roc_auc_val_rf)  #0.555

fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, y_pred_rf)


#plot the ROC curve
plt.plot(fpr_rf, tpr_rf, lw=2, label='ROC curve for Random Forest model (area = %0.2f)' % roc_auc_val_rf)
plt.plot([0,1],[0,1],color='grey', linestyle='--', lw=1)
plt.legend(loc="lower right", frameon = False)
plt.title('ROC curve - Random Forest model')
plt.savefig(wkg_dir + '/ROC_rf.jpg', 
    bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
    orientation='portrait', papertype=None, format=None, 
    transparent=True, pad_inches=0.25, frameon=None)  
plt.show()



