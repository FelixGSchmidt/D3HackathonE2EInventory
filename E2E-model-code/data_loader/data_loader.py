# -*- coding: utf-8 -*-
# @Author: chenxinma
# @Date:   2018-10-09 11:00:00
# @Last Modified by:   Yuanyuan Shi, Meng Qi
# @Last Modified at:   2022-05-15 22:07:03


import torch
import torch.utils.data as data
import pickle 
from math import ceil 
import pandas as pd  
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from configs.config import *
import datetime as dt

class E2E_Dataset(data.Dataset):

    def __init__(self, o2, sku_train, sku_test, phase, model_name, b, device):

        self.phase = phase
        self.device = device
        self.model_name = model_name
        o2[LABEL_sf[0]] =  np.log1p(o2[LABEL_sf[0]])

        df_train = o2
        df_test = df_train[0:1000]

        self.train_len = len(df_train)
        self.test_len = len(df_test)

        if b != None:
            LABEL = ['demand_RV_%i' %b]
        df_train = pd.concat([df_train[VLT_FEA+SF_FEA+CAT_FEA_HOT+MORE_FEA+IS_FEA], \
                                df_train[LABEL], df_train[LABEL_vlt+LABEL_sf+IDX+SEQ2SEQ]], axis=1)
        print('\n\n\n', len(df_train))
        df_test = pd.concat([df_test[VLT_FEA+SF_FEA+CAT_FEA_HOT+MORE_FEA+IS_FEA], \
                                df_test[LABEL], df_test[LABEL_vlt+LABEL_sf+IDX+SEQ2SEQ+
                                ['initial_stock', 'next_complete_dt']]], axis=1)

        X_train_ns, y_train_ns, id_train = df_train[SCALE_FEA], df_train[LABEL], df_train[IDX]
        X_test_ns, y_test_ns, id_test = df_test[SCALE_FEA], df_test[LABEL], df_test[IDX]

        self.n_train, self.n_test = len(X_train_ns), len(X_test_ns)
        self.X_scaler = MinMaxScaler() # For normalizing dataset
        self.y_scaler = MinMaxScaler() # For normalizing dataset
        self.X_train = pd.DataFrame(self.X_scaler.fit_transform(X_train_ns), columns=X_train_ns.columns)
        self.y_train = pd.DataFrame(self.y_scaler.fit_transform(y_train_ns), columns=y_train_ns.columns)

        self.X_test = pd.DataFrame(self.X_scaler.transform(X_test_ns), columns=X_test_ns.columns)
        self.y_test = pd.DataFrame(self.y_scaler.transform(y_test_ns), columns=y_test_ns.columns)

        pd_scaler = pd.concat([pd.DataFrame([self.y_scaler.data_min_, self.y_scaler.scale_], columns=y_train_ns.columns),
                    pd.DataFrame([self.X_scaler.data_min_, self.X_scaler.scale_], columns=X_train_ns.columns)], axis=1)
        pd_scaler.to_csv('../data2/1320_feature/scaler.csv', index=False)

        if self.phase=='test':
            df = self.X_test.astype(float)
            lb = self.y_test.astype(float)
            s2s = df_test
            pd.concat([id_test, df_test[['vlt_actual', 'initial_stock', 'mean_140', 'std_140','mean_14',
                                        'vendor_vlt_mean', 'vlt_std',
                                        'review_period', 'next_complete_dt']]], axis=1)\
                                          .to_csv('../logs/torch/pred_sku.csv', index=False)
        else:
            df = self.X_train.astype(float)
            lb = self.y_train.astype(float)
            s2s = df_train

        self.X = pd.concat([df[VLT_FEA+SF_FEA+CAT_FEA_HOT+MORE_FEA+IS_FEA], lb, df[LABEL_vlt+LABEL_sf]], \
            axis=1)
        self.S1 = torch.cat([torch.FloatTensor(s2s['Enc_X']).view(-1,rnn_hist_long,2), 
                             torch.FloatTensor(s2s['Enc_y']).view(-1,rnn_hist_long,1)], 2)
        self.S2 = torch.cat([torch.FloatTensor(s2s['Dec_X']).view(-1,rnn_pred_long,2), 
                             torch.FloatTensor(s2s['Dec_y']).view(-1,rnn_pred_long,1)], 2)
        print(self.X.isnull().sum().sum(), self.train_len, self.test_len)


        


    def __getitem__(self, idx):

        if 'v5' in self.model_name:
            return self.X[idx].to(self.device)
        else:
            cur_state = self.X.loc[idx]
            cur_state = cur_state
            a = np.array(cur_state.values)
            a = torch.tensor(a.reshape(-1))
            return a.to(self.device), self.S1[idx].type('torch.FloatTensor').to(self.device), self.S2[idx].type('torch.FloatTensor').to(self.device)
        

    def __len__(self):

        if self.phase=='train':
            return self.train_len
        else:
            return self.test_len




def get_loader(batch_size, device, model_name, b=None, eval=0, test_sku='None', data_dir='../data2/', num_workers=0):

    with open(data_dir + '1320_feature/df_e2e.pkl', 'rb') as fp:
        o2 = pickle.load(fp)

    o2 = o2.replace(np.inf, 0)
    sku_set = o2.sku_id.unique()
    sku_train, sku_test = train_test_split(sku_set, random_state=12, train_size=0.9, test_size=0.1)
    print('Train sku number:', len(sku_train), 'Test sku number:', len(sku_test))
    if b != None:
        o2['demand_RV_%i' %b] = o2.apply(lambda x: sum(x['demand_RV_list']\
                                [:-ceil((x['review_period']+x['vlt_actual'])//(b+1))]) 
                                if ceil((x['review_period']+x['vlt_actual'])//(b+1))>0 \
                                else sum(x['demand_RV_list']), axis=1)

    if eval == 0:
        train_set = E2E_Dataset(o2, sku_train, sku_test, 'train', model_name, b, device)
        data_loader = torch.utils.data.DataLoader(dataset=train_set,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=0,
                                              # pin_memory=True
                                              )
        test_set = E2E_Dataset(o2, sku_train, sku_test, 'test', model_name, b, device)
        test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=0,
                                                  # pin_memory=True
                                                  )
    else: 
        if test_sku != 'None':
            sku_test = [test_sku]
        data_loader = {}
        test_set = E2E_Dataset(o2, sku_train, sku_test, 'test', model_name, b, device)
        test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                                  batch_size=test_set.test_len,
                                                  shuffle=False,
                                                  num_workers=0,
                                                  # pin_memory=True
                                                  )

    return data_loader, test_loader