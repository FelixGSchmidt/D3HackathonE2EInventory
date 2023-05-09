# -*- coding: utf-8 -*-
# @Author: chenxinma
# @Date:   2018-10-09 11:00:00
# @Last Modified by:   Yuanyuan Shi
# @Last Modified at:   2019-07-26 22:07:03
import numpy as np
import pandas as pd
import os,math
from math import ceil 
import warnings
import pickle
from scipy.stats import gamma
import datetime as dt
import sys
sys.path.append('../')
from utils.demand_pkg import *
from configs.config import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class data_parser:

    def __init__(self):

        pass

    def process_vlt_data(self, path, path_to, filename):

        file_to_save = filename.split('.')[0] + '_prep.csv'
        if  os.path.exists(path_to+file_to_save):
            df_vlt = pd.read_csv('%s%s' %(path_to, file_to_save), parse_dates=['create_tm','complete_dt','dt'])
            print('Vlt processed data read!')
        else:
            if not os.path.isdir(os.path.join(os.getcwd(), path_to)):
                os.makedirs(path_to)
            df_vlt = pd.read_csv('%s%s' %(path, filename), parse_dates=['create_tm','complete_dt','dt'])

            df_vlt['x_index'] = (df_vlt['create_tm']- dt.datetime.strptime('2018-03-01','%Y-%m-%d')).dt.days
            df_vlt['create_tm_index'] = (df_vlt['create_tm']- dt.datetime.strptime('2018-03-01','%Y-%m-%d')).dt.days
            df_vlt['complete_dt_index'] = (df_vlt['complete_dt']- dt.datetime.strptime('2018-03-01','%Y-%m-%d')).dt.days

            df_vlt = df_vlt.dropna(how='any')
            df_vlt = df_vlt.sort_values(['item_sku_id','int_org_num','create_tm_index','dt'], ascending=[True, True, True, True])


            #df_vlt.insert(1, 'sku_id', df_vlt['item_sku_id'])
            df_vlt['item_sku_id'] = df_vlt[['item_sku_id', 'int_org_num']].astype(str).apply(lambda x: '#'.join(x), axis=1)
            df_vlt = df_vlt.drop_duplicates(['item_sku_id', 'create_tm_index'], keep='last')
            df_vlt = df_vlt.reset_index(drop=True)
            
            df_vlt.insert(1, 'sku_id', df_vlt['item_sku_id'])
            df_vlt.loc[:,'sku_id'] = df_vlt['sku_id'].apply(lambda x: x.split('#')[0])
            
            df_vlt.to_csv('%s%s' %(path_to, file_to_save), index=False)
            print('Vlt raw data processed!')

        return df_vlt


    def process_sales_data(self, quantile_list, quantile_window_list, mean_window_list, path, path_to, filename):

        df_sl = pd.read_csv('%s%s' %(path, filename), index_col=0)
        df_sl.rename(columns=lambda x: (dt.datetime(2018,3,1) + dt.timedelta(days=int(x)-1520)).date(), inplace=True)
        print('Sales data read!')

        get_rolling_mean(df_sl, mean_window_list, path=path_to)
        get_rolling_quantile(df_sl, quantile_list, quantile_window_list, path=path_to)

        print('Sales features generated!')
        return df_sl


    def read_sales_data(self, quantile_list, quantile_window_list, mean_window_list, path):

        quantile_feature, mean_feature = read_feature_data(quantile_list, quantile_window_list, mean_window_list, path)        
        print('Sales features read!')
        return quantile_feature, mean_feature



    def get_vlt_sales_feature(self, quantile_list, quantile_window_list, mean_window_list, pred_len_list, 
                              raw_path, process_path, path_to,
                              vlt_file, filled_sale_file, file_to_save,
                              Model='M2Q', is_far=False):

        if os.path.exists('%s%s' %(path_to, file_to_save)):
            X = pd.read_csv('%s%s' %(path_to, file_to_save), parse_dates=['create_tm','complete_dt','dt'] )
            print('VLT and sales feature read!')

        else:
            df_vlt = self.process_vlt_data(raw_path, process_path, vlt_file)
            df_sl = self.process_sales_data(quantile_list, quantile_window_list, mean_window_list, 
                                            raw_path, process_path, filled_sale_file)
            quantile_feature, mean_feature = self.read_sales_data(quantile_list, quantile_window_list, mean_window_list, process_path)

            test_date = df_vlt['create_tm'].dt.normalize().rename('train_date')
            pred_len = pred_len_list[0]
            q = quantile_list[0]

            X = prepare_training_data(df_sl, q, 
                                     is_far, 
                                     pred_len, 
                                     Model, 
                                     test_date, 
                                     mean_window_list, 
                                     quantile_window_list, 
                                     quantile_feature,
                                     mean_feature,
                                     df_vlt['item_sku_id'],
                                     is_train=False)

            X = pd.concat([df_vlt, X], axis=1)
            X = X[X['complete_dt']<dt.datetime(2019,5,31)]
            X = X.sort_values(['item_sku_id','int_org_num','create_tm'], ascending=[True, True, True])
            X = X.drop_duplicates(['item_sku_id', 'create_tm'], keep='last')
            X.to_csv('%s%s' %(path_to, file_to_save), index=False)

            print('VLT and sales feature obtained and aggregated!')
        return X


    def add_stock_feature(self, X, raw_path, path_to, stock_file, file_to_save):

        df_st = pd.read_csv('%s%s' %(raw_path, stock_file), index_col=0)
        df_st.columns = pd.to_datetime(df_st.columns)

        X['initial_stock'] = X.apply(lambda x: df_st.loc[x['item_sku_id'], x['create_tm'].normalize() ] \
                                if x['item_sku_id'] in df_st.index else np.nan, axis=1)
        X['VLT'] = (X['complete_dt'] - X['create_tm']) / timedelta (days=1)
        X['review_period'] = X['create_tm'].diff().shift(-1)/ timedelta (days=1)
        X.loc[X['review_period'] <=0, 'review_period'] = (dt.datetime(2019,5,31) - \
                                    X.loc[X['review_period'] <=0, 'create_tm'])/ timedelta (days=1)
        X['review_period'] = X['review_period'].fillna(0)
        # X.to_csv('%s%s' %(path_to, file_to_save), index=False)
        print('Stock features aggregated!')
        return X


    def add_3b_feature(self, X, raw_path, path_to, stock_file, file_to_save):
        # Add features based on benchmark
        Z90 = SCALAR # Disguised value
        PERCENT = SCALAR #Disguised value
        '''--------------------- Normal Basestock -----------------------'''
        sale_mean = X['mean_112']
        sale_std = X['std_140']
        VLT_mean = X['vendor_vlt_mean']
        VLT_std = 0
        T = X['review_period'] + X['vendor_vlt_mean']
        X['normal'] = (sale_mean*T + Z90*np.sqrt(T*sale_std**2+sale_std**2*VLT_std)-X['initial_stock']).fillna(0).clip(0)

        '''--------------------- Gamma Basestock -----------------------'''
        theta = sale_std**2/(sale_mean+0.0000001)
        k = sale_mean/(theta+0.0000001)
        k_sum = T*k
        X['gamma'] = pd.Series(gamma.ppf(PERCENT, a=k_sum, scale = theta)) - X['initial_stock']
        X['gamma'] = X['gamma'].fillna(0).clip(0)

        '''--------------------- Empirical Quantile Basestock -----------------------'''
        X['eq'] = (X['q_112'] * T - X['initial_stock']).clip(0)
        return X



    def add_target(self, X, path, filename, path_to, file_to_save):
        # Load csv
        df_sl = pd.read_csv('%s%s' %(path, filename), index_col=0)
        '''Preprocessing:
        Convert datetime index to date-time. 
        Drop duplicates.
        Drop NA.
        Get order complete date.
        Get next order complete date.
        X['target_decision] = Calculate target decision following Theorem 1
        '''
        print('Targets aggregated!')
        return X


    def add_more_and_labels(self, X, raw_path, output_path, filled_sale_file, stock_file, file_to_save):

        X = self.add_stock_feature(X, raw_path, output_path, stock_file, file_to_save)
        X = self.add_3b_feature(X, raw_path, output_path, stock_file, file_to_save)
        X = self.add_target(X, raw_path, filled_sale_file, output_path, file_to_save)
        return X


    def add_time_series(self, X, lw, rw, path, process_path, sale_file, file_read, file_to_save):

        o0 = X.copy()

        df_sl = pd.read_csv(process_path+sale_file, index_col=0)
        df_sl.rename(columns=lambda x: (dt.datetime(2018,3,1) + dt.timedelta(days=int(x)-1520)).date(), inplace=True)

        o0['Enc_X'] = o0.apply(lambda x: [
                            np.log1p(df_sl.loc[x['item_sku_id'], 
                                     x['create_tm'].date()-dt.timedelta(days=lw+180):
                                          x['create_tm'].date()-dt.timedelta(days=180)
                                    ]).values.tolist(),
                            np.log1p(df_sl.loc[x['item_sku_id'], 
                                     x['create_tm'].date()-dt.timedelta(days=lw+92):
                                          x['create_tm'].date()-dt.timedelta(days=92)
                                     ]).values.tolist(),
                           ], axis=1)

        o0['Enc_y'] = o0.apply(lambda x: [np.log1p(df_sl.loc[x['item_sku_id'], 
                                     x['create_tm'].date()-dt.timedelta(days=lw):x['create_tm'].date()
                                    ]).values.tolist(),
                           ], axis=1)

        o0['Dec_X'] = o0.apply(lambda x: [
                            np.log1p(df_sl.loc[x['item_sku_id'], 
                                     x['create_tm'].date()-dt.timedelta(days=180):
                                          x['create_tm'].date()-dt.timedelta(days=180-rw)
                                    ]).values.tolist(),
                            np.log1p(df_sl.loc[x['item_sku_id'], 
                                     x['create_tm'].date()-dt.timedelta(days=92):
                                          x['create_tm'].date()-dt.timedelta(days=92-rw)
                                     ]).values.tolist(),
                           ], axis=1)
        o0['Dec_y'] = o0.apply(lambda x: [np.log1p(df_sl.loc[x['item_sku_id'], 
                                     x['create_tm'].date():x['create_tm'].date()+dt.timedelta(days=rw)
                                    ]).values.tolist(),
                           ], axis=1)
        o0['Dec_y'] = o0['Dec_y'].apply(lambda x:  [x[0]+[x[0][-1]]*(rw+1-len(x[0]))] 
                                                    if len(x[0])<rw+1 else x)
        return o0

def dummy_cut(X, path, file_read, file_to_save):
    '''
    More processing on X, rename column names
    dump processed X as pickle
    '''
    with open(path+file_to_save, 'wb') as file_pkl:
        pickle.dump(o2r, file_pkl, protocol=pickle.HIGHEST_PROTOCOL)


raw_path = PATH
process_path = PATH
output_path = PATH

filled_sale_file = 'rdc_sales_1320.csv' # Input raw sales data
vlt_file = 'vlt_2019_0305.csv'   # Input raw vlt data
stock_file = 'stock.csv'    # Input raw stock data
feature_file = 'features_v1.csv'  # Generated intermediate dataset: features
feature_file2 = 'features_v11.csv' # Generated intermediate dataset: features
feature_file4 = 'features_v13.pkl'  # Generated intermediate dataset: features


quantile_list = LIST_OF_QUANTILE # Disguised. Example: [85]
quantile_window_list = LIST_OF_WINDOW # Disguised. Example: [7, 14, 30]
mean_window_list = LIST_OF_QUANTILE # Disguised. Example: [7, 14, 30]
pred_len_list = LIST_OF_LEN # Disguised. Example: [7, 3]

o = data_parser()
X = o.get_vlt_sales_feature(quantile_list, quantile_window_list, mean_window_list, pred_len_list,
                  raw_path, process_path, output_path, vlt_file, filled_sale_file, feature_file)
X = o.add_more_and_labels(X, raw_path, output_path, filled_sale_file, stock_file, feature_file2)
X = o.add_time_series(X, 90, 30, output_path, process_path, filled_sale_file, feature_file2, feature_file4)
dummy_cut(X, output_path, feature_file4, 'df_e2e.pkl')
# 'df_e2e.pkl' contains the processed dataset for training and testing
