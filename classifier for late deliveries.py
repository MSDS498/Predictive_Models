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
orders_df['est_delivery_time'] = (orders_df.order_estimated_delivery_date - orders_df.order_purchase_timestamp).dt.total_seconds()/86400
orders_df['est_delivery_time_days'] = np.ceil(orders_df['est_delivery_time']).astype(int)

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


#orders_df.drop('distance_rough', axis='columns', inplace=True)

# outlier and missing handling
# ----------------------------
# drop rows with missing values as some techniques are sensitive to this (logistic regression)
orders_df2 = orders_df.loc[ ( pd.notnull(orders_df.distance_km) ) & ( pd.notnull(orders_df.days_remaining) ) &
                            ( pd.notnull(orders_df.order_purchase_timestamp) ) & ( pd.notnull(orders_df.order_approved_at) ) &
                            ( pd.notnull(orders_df.order_delivered_customer_date) ) & ( pd.notnull(orders_df.order_delivered_carrier_date) ) &
                            ( pd.notnull(orders_df.order_estimated_delivery_date) ) & ( pd.notnull(orders_df.ship_limit_final) ) &
                            ( pd.notnull(orders_df.nbr_items) ) & ( pd.notnull(orders_df.est_delivery_time_days) ) &
                            ( pd.notnull(orders_df.shipping_limit_missed) ) & ( pd.notnull(orders_df.est_delivery_time) ) &
                            ( pd.notnull(orders_df.product_ctgry_mfu) ) ]
    #add this if we try regression    product_ctgry_mfu
    
#orders_df2.info()
#pd.value_counts(orders_df2.order_status)
#95k of them are delivered; 6 are cancelled.  shouldn't matter to delivery calcs as we've ensured that delivery did happen


# ============================
# Split into test and train
# ============================
predictor_cols = ['est_delivery_time_days', 'shipping_limit_missed', 'shipping_limit_miss_amt', 'days_remaining', 'distance_km', 'approval_time_days', 'nbr_items',
                  'nbr_sellers', 'nbr_products', 'ttl_wt', 'ttl_length', 'ttl_height', 'ttl_width']

#'seller_city', 'seller_state', 'product_ctgry_mfu', 

#orders_df2.columns
#  Other possibles:     'ship_limit_final', 'ship_limit_initial',  -- these were a maybe, but the model didn't like timestamp as an argument 
#                         'payment_type_mfu', 'ttl_pd', 'pmt_mthds_used', 'installments_used_ttl', 'payment_types_used', 'ttl_price', 'ttl_freight',
#                       'nbr_photos', 'order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date',     'order_delivered_customer_date', 'order_estimated_delivery_date', 'customer_unique_id', 'customer_zip_code_prefix', 'customer_city', 'customer_state',
predicted_col = ['late_delivery_flag']

RANDOM_SEED=42
x_train, x_test, y_train, y_test = train_test_split(orders_df2[predictor_cols], orders_df2[predicted_col], test_size=0.25, random_state=RANDOM_SEED)



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

mdl_rf = RandomForestClassifier(random_state=RANDOM_SEED)
mdl_rf.fit(x_train, y_train)
mdl_rf.score(x_test, y_test)
#0.931

y_pred_rf = mdl_rf.predict(x_test)
actl_and_pred_rf = y_test.copy()
actl_and_pred_rf['predicted_late_flag'] = y_pred_rf
actl_and_pred_rf.columns = ['actual_late_flag', 'predicted_late_flag']

#look at the confusion matrix
actl_and_pred_rf.groupby(['actual_late_flag', 'predicted_late_flag']).agg(cnt=('actual_late_flag', 'count')).reset_index()
# model is better with ship limit miss amt (correctly about 2x as many true lates, and also has fewer false positives )
#   actual_late_flag  predicted_late_flag    cnt
#                 0                    0  21987
#                 0                    1     60
#                 1                    0   1431
#                 1                    1    182

#   actual_late_flag  predicted_late_flag    cnt
#                 0                    0  21927
#                 0                    1     78
#                 1                    0   1563
#                 1                    1     92


#calculate precision-recall score (avg?), curves
precision_rf, recall_rf, _ = precision_recall_curve(y_test, y_pred_rf)

average_precision_rf = average_precision_score(y_test, y_pred_rf)
print('Average precision-recall score: {0:0.2f}'.format(average_precision_rf))
#0.15   #  :(

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

#mdl_rf.feature_importances_
#mdl_rf.get_params()



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











# ================================================
#  re-train with undersampled data 
#        use all the late delivery records, but a much lower % (same # as the late deliv records) from the ontime delivery so that our training set is balanced
#        hopefully that will make it more sensitive 
# ================================================

nbr_late_delivs = sum(orders_df2.late_delivery_flag)  #6394

#bring along the Rww_ID temporarily so that we can use it to create the validation set
predictor_cols2 = predictor_cols.copy()
predictor_cols2.append('Rww_ID') 
predicted_col2 = predicted_col.copy()
predicted_col2.append('Rww_ID')
#create the undersampled (us) dataset
#take a 25% sample of the records where delivery was late    ==> we should have about 6394*0.75 (=4795.5) records in training
x_train_us, x_test_us, y_train_us, y_test_us = train_test_split(orders_df2.loc[orders_df2.late_delivery_flag == True, predictor_cols2], 
                                                                orders_df2.loc[orders_df2.late_delivery_flag == True, predicted_col2],
                                                                test_size = 0.25)
#x_train_us.shape  #  (4795, 13)     good, matches expected size
#x_test_us.shape  #  (1599, 13)     good, matches expected size

x_train_us_0, x_test_us_0, y_train_us_0, y_test_us_0 = train_test_split(orders_df2.loc[orders_df2.late_delivery_flag == False, predictor_cols2],
                                                                        orders_df2.loc[orders_df2.late_delivery_flag == False, predicted_col2],
                                                                        train_size = 4795, test_size = 1599) 
#x_train_us_0.shape  #  (4795, 13)     good, matches expected size
#x_test_us_0.shape  #  (1599, 13)     good, matches expected size
#sum(y_train_us_0.late_delivery_flag)

#merge the samples for late and ontime together
x_train_us = x_train_us.append(x_train_us_0)
y_train_us = y_train_us.append(y_train_us_0)

#orders_df2.info()
#for the validation, run on all the records that were not used in training, or on these test records???
x_test_us = x_test_us.append(x_test_us_0)
y_test_us = y_test_us.append(y_test_us_0)

x_valdn_us = orders_df2.loc[ ~ orders_df2.Rww_ID.isin(x_train_us.Rww_ID) , predictor_cols]
y_valdn_us = orders_df2.loc[ ~ orders_df2.Rww_ID.isin(y_train_us.Rww_ID) , predicted_col]
    #x_valdn_us.shape  # 85048, 13
    #y_valdn_us.shape  # 85048, 1


#drop the Rww_ID so it doesn't get fed to the model
x_train_us.drop('Rww_ID', axis='columns', inplace = True)
x_test_us.drop('Rww_ID', axis='columns', inplace = True)
y_train_us.drop('Rww_ID', axis='columns', inplace = True)
y_test_us.drop('Rww_ID', axis='columns', inplace = True)




#fit and score a Random Forest classifier
mdl_rf_us = RandomForestClassifier(n_estimators = 100, random_state=RANDOM_SEED)
mdl_rf_us.fit(x_train_us, y_train_us)
mdl_rf_us.score(x_test_us, y_test_us)   #0.662. up from 0.643 with n_estimators = 10 (default)
mdl_rf_us.score(x_valdn_us, y_valdn_us)  #0.71  :( :(


y_pred_rf_us = mdl_rf_us.predict(x_valdn_us)
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
