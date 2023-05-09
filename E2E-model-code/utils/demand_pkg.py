import pandas as pd
import numpy as np

from datetime import date, timedelta, datetime, time
from dateutil.relativedelta import relativedelta
import copy, os
import pickle

def get_formated_index(quantile_list):
    return pd.Float64Index(data=sorted([x/100 for x in quantile_list]), name='Quantile')

def reformat(x, quantile_list, pred_len_list):
    temp = pd.DataFrame(np.reshape(x.values, [len(quantile_list), len(pred_len_list)]),
             index=get_formated_index(quantile_list), columns=sorted(pred_len_list))
    return pd.DataFrame(np.sort(temp.values, axis=0), index=temp.index, columns=temp.columns)

def get_rolling_quantile(df, quantile_list, window_list, path="feature_data"):
    """
    Get rolling quantile values
    :param df: a dataframe of input sales data
    :param quantile_list: a list of quantile values
    :param window_list: a list of time windows
    :param path: directory to store feature data files
    :return: save dataframe as csv file to local disk
    """
    if not os.path.isdir(os.path.join(os.getcwd(), path)):
        os.makedirs(path)
    for quantile in quantile_list:
        for window in window_list:
            quantile_key = 'window_{window}_quantile_{quantile}'.format(window=str(window), quantile=str(quantile))
            file_to_save = path+'/'+quantile_key
            if not os.path.exists(file_to_save):
                print (quantile_key)
                temp = df.T.rolling(window).quantile(float(quantile)/100).T
                temp.to_csv(path+'/'+quantile_key)
    return


def get_rolling_mean(df, window_list, path="feature_data"):
    """
    Get rolling mean values
    :param df: a dataframe of input sales data
    :param window_list: a list of time windows
    :param feature_data: directory to store feature data files
    :return: save dataframe as csv file to local disk
    """
    if not os.path.isdir(os.path.join(os.getcwd(), path)):
        os.makedirs(path)
    for window in window_list:
        mean_key = 'window_{window}_mean'.format(window=str(window))
        file_to_save = path+'/'+mean_key
        if not os.path.exists(file_to_save):
            print (mean_key)
            temp = df.T.rolling(window).mean().T
            temp.to_csv(path+'/'+mean_key)
    return


def read_feature_data(quantile_list, quantile_window_list, mean_window_list, path="feature_data"):
    """
    read feature_data from path
    :param quantile_list: a list of quantile values
    :param window_list: a list of window values
    :param path: directory of the feature data
    :return: two dictionaries of dataframes
    """
    quantile_feature = dict()
    mean_feature = dict()
    for window in quantile_window_list:
        for quantile in quantile_list:
            quantile_key = 'window_{window}_quantile_{quantile}'.format(window=str(window), quantile=str(quantile))
            print ('Reading ', quantile_key)
            df = pd.read_csv(path+'/'+quantile_key, index_col=0)
            df.columns = pd.to_datetime(df.columns)
            quantile_feature[quantile_key] = df
            
        
    for window in mean_window_list:
        mean_key = 'window_{window}_mean'.format(window=str(window))
        print ('Reading ', mean_key)
        df = pd.read_csv(path+'/'+mean_key, index_col=0)
        df.columns = pd.to_datetime(df.columns)
        mean_feature[mean_key] = df
           
    return quantile_feature, mean_feature


def get_quantile_value(quantile, window, quantile_feature, sku_dc_pair, train_date):
    train_date = train_date-pd.DateOffset(days=1)
    quantile_key = 'window_{window}_quantile_{quantile}'.format(window=str(window), quantile=str(quantile))
    data = quantile_feature.get(quantile_key, None)
    df_key = pd.concat([sku_dc_pair, train_date], axis=1)
    if data is None:
        return
    return df_key.apply(lambda x: data.loc[x['item_sku_id'], x['train_date']] \
            if x['item_sku_id'] in data.index else np.nan, axis=1)


def get_mean_value(window, mean_feature, sku_dc_pair, train_date):
    train_date = train_date-pd.DateOffset(days=1)
    mean_key = 'window_{window}_mean'.format(window=window)
    data = mean_feature.get(mean_key, None)
    df_key = pd.concat([sku_dc_pair, train_date], axis=1)
    if data is None:
        return
    return df_key.apply(lambda x: data.loc[x['item_sku_id'], x['train_date']] \
            if x['item_sku_id'] in data.index else np.nan, axis=1)


def get_back_timespan(df, sku_dc_pair, train_date, minus, period):
    df_key = pd.concat([sku_dc_pair, train_date], axis=1)
    df_key['date_common'] = df_key.apply(lambda x: \
                pd.date_range(start=x['train_date']-timedelta(days=minus), periods=period) , axis=1)
    return df_key.apply(lambda x: pd.Series(df.loc[x['item_sku_id'], x['date_common'] ].values \
                            if x['item_sku_id'] in df.index else [np.nan]*period
                            ), axis=1)


def get_forward_timespan(df, sku_dc_pair, train_date, period):
    df_key = pd.concat([sku_dc_pair, train_date], axis=1)
    df_key['date_common'] = df_key.apply(lambda x: \
                pd.date_range(start=x['train_date'], periods=period) , axis=1)
    df_key['len_common'] = df_key['date_common'].apply(len)
    if df_key['len_common'].min() < period:
        print(df_key['len_common'].min, period)

    return df_key.apply(lambda x: pd.Series(df.loc[x['item_sku_id'], x['date_common'] ].values \
                            if x['item_sku_id'] in df.index else [np.nan]*period
                            ), axis=1)


def get_summary_stats(df, sku_dc_pair, train_date, model):
    X = pd.DataFrame()
    window_list = DISGUISED WINDOW LIST #Confidential
    print('summary s1 begin.')
    for i in range(len(window_list[0])-1, -1, -1):
        window = window_list[0][i]
        if i == len(window_list[0])-1:
            tmp = get_back_timespan(df, sku_dc_pair, train_date, window, window)
            tmp_copy = tmp.copy()
        else:
            tmp = tmp.iloc[:, -window:]
        X['diff_%s_mean' % window] = tmp.diff(axis=1).mean(axis=1)
        X['mean_%s_decay' % window] = (tmp * np.power(0.9, np.arange(window)[::-1])).sum(axis=1)
        X['median_%s' % window] = tmp.median(axis=1).values # median is exactly 50th quantile
        X['min_%s' % window] = tmp.min(axis=1)
        X['max_%s' % window] = tmp.max(axis=1)
        X['std_%s' % window] = tmp.std(axis=1)

    print('summary s2 begin.')
    for i in range(len(window_list[1])-1, -1, -1):
        window = window_list[1][i]
        if i == len(window_list[1])-1:
            if window ==  window_list[0][-1]:
                 tmp = tmp_copy.copy()
            else:
                tmp = get_back_timespan(df, sku_dc_pair, train_date, window, window)
        else:
            tmp = tmp.iloc[:, -window:]
        X['has_sales_days_in_last_%s' % window] = (tmp > 0).sum(axis=1)
        X['last_has_sales_day_in_last_%s' % window] = window - ((tmp > 0) * np.arange(window)).max(axis=1)
        X['first_has_sales_day_in_last_%s' % window] = ((tmp > 0) * np.arange(window, 0, -1)).max(axis=1)
    return X



def get_label(df, sku_dc_pair, train_date, quantile, pred_len, model, is_train):
    if not is_train: return
    tmp = get_forward_timespan(df, sku_dc_pair, train_date, pred_len)
    if model =='M2Q':
        y = np.sqrt(np.sum(tmp, axis=1))  # Yuze Quantile regression #^(1/2) smooth
    else:
        y_sum = []
        pred_len_tmp = int(np.max([pred_len, 31]))
        for i in range(pred_len_tmp):
            y_sum.append(np.sum(get_back_timespan(df, sku_dc_pair, train_date, pred_len_tmp // 2 - i, pred_len), axis=1))
        y = np.sqrt(np.percentile(y_sum, quantile, axis=0)).transpose()
    return y


def get_features(quantile, quantile_window_list, mean_window_list, sku_dc_pair, train_date, quantile_feature, mean_feature): # done
    X = pd.DataFrame()
    for window in quantile_window_list:
        quantile_column = 'window_{window}_quantile_{quantile}'.format(window=window, quantile=quantile) #TODO: change this hard coded thing
        quantile_column_name = 'q_{window}'.format(window=window)
        X[quantile_column_name] = get_quantile_value(quantile, window, quantile_feature, sku_dc_pair, train_date)
        
    for window in mean_window_list:
        mean_column = 'window_{window}_mean'.format(window=window)
        mean_column_name = 'mean_{window}'.format(window=window)
        X[mean_column_name] = get_mean_value(window, mean_feature, sku_dc_pair, train_date)
    return X


def prepare_dataset(df, sku_dc_pair, train_date, quantile, quantile_window_list, mean_window_list, 
                    pred_len, model, quantile_feature, mean_feature, is_train):
    quantile_mean_data = get_features(quantile, quantile_window_list, mean_window_list, sku_dc_pair, train_date, quantile_feature, mean_feature)
    summary_stats_data = get_summary_stats(df, sku_dc_pair, train_date, model)
    y = get_label(df, sku_dc_pair, train_date, quantile, pred_len, model,is_train)
    all_data = [quantile_mean_data, summary_stats_data]
    data = pd.concat(all_data, axis=1)
    return data, y


def get_train_date(test_date, pred_len, is_far, rolling_num, rolling_span, time_back_num):
    if is_far: #Yuze hardcode jump parameters 3
        train_date = test_date - pd.DateOffset(years=1)-  pd.DateOffset(days = rolling_span * time_back_num) # set earliest jump start 
    else: # ensure NOT TOO NEAR
        pred_len_tmp = int(np.max([pred_len,31]))
        train_date = test_date -  pd.DateOffset(days= 2 * pred_len_tmp + (rolling_num-1) * rolling_span)  ##########
    return train_date


def prepare_training_data(df, quantile, is_far, pred_len, model, test_date, mean_window_list, 
                    quantile_window_list, quantile_feature, mean_feature, sku_dc_pair, 
                    is_train, time_back_num = 2, rolling_num=5, rolling_span=7):
    if is_train:
        X = [None]*rolling_num
        y = [None]*rolling_num
    
        train_date = get_train_date(test_date, pred_len, is_far, rolling_num, rolling_span, time_back_num)
        for i in range(rolling_num):
            delta = timedelta(days=rolling_span * i)  # Yuze jumps around, from earliest to latest
            X_tmp, y_tmp = prepare_dataset(df, sku_dc_pair, train_date + delta, quantile, quantile_window_list, 
                                            mean_window_list, pred_len, model, quantile_feature, mean_feature, is_train)
            X[i] = X_tmp
            y[i] = y_tmp
        X = pd.concat(X, axis=0)
        y = [i for yy in y for i in yy]
        return X, y
    
    else: # 2018-08-30 bug fixed
        X, _ = prepare_dataset(df, sku_dc_pair, test_date , quantile, quantile_window_list, mean_window_list, pred_len, model, 
                                quantile_feature, mean_feature, is_train)
        return X

