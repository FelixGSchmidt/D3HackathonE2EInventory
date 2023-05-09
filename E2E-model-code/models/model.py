# -*- coding: utf-8 -*-
# @Author: chenxinma, yuanyuanshi, Meng Qi
# @Date:   2018-10-09 11:00:00
# @Last Modified by:   Meng Qi
# @Last Modified at:   2022-05-12 22:07:03
# Template:
# https://github.com/yunjey/domain-transfer-network/blob/master/model.py

import tensorflow as tf
# import sys
# sys.path.append('../')
from configs.config import *
# from model_s2s import *
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim




class Decoder_MLP(nn.Module):

    def __init__(self, x_dim, hidden_size, context_size, num_quantiles, pred_long, bridge_size):
        """Set the hyper-parameters and build the layers."""
        super(Decoder_MLP, self).__init__()
        
        self.x_dim = x_dim
        self.hidden_size = hidden_size
        self.context_size = context_size
        self.num_quantiles = num_quantiles
        self.pred_long = pred_long

        self.global_mlp = nn.Linear(hidden_size + x_dim * pred_long, context_size*(pred_long+1))
        self.local_mlp = nn.Linear(context_size * 2 + x_dim, num_quantiles)
        self.bridge_mlp = nn.Linear(context_size * 2 + x_dim, bridge_size)

        self.init_weights()
    
    def init_weights(self):
        """Initialize weights."""
        # Disguised initialization

        
        
    def forward(self, hidden_states, x_future):
        # hidden_states: N x hidden_size
        # x_future:      N x pred_long x x_dim
        # y_future:      N x pred_long

        x_future_1 = x_future.view(x_future.size(0),x_future.size(1)*x_future.size(2))

        # global MLP
        hidden_future_concat = torch.cat((hidden_states, x_future_1),dim=1)
        context_vectors = torch.sigmoid( self.global_mlp(hidden_future_concat) )
        ca = context_vectors[:, self.context_size*self.pred_long:]

        results, bridges = [], []
        for k in range(self.pred_long):
            xk = x_future[:,k,:]
            ck = context_vectors[:, k*self.context_size:(k+1)*self.context_size]
            cak = torch.cat((ck,ca,xk),dim=1)
            # local MLP
            quantile_pred = self.local_mlp(cak)
            quantile_pred = quantile_pred.view(quantile_pred.size(0),1,quantile_pred.size(1))
            results.append(quantile_pred)

            hidden_bridge = self.bridge_mlp(cak)
            hidden_bridge = hidden_bridge.view(hidden_bridge.size(0),1,hidden_bridge.size(1))
            bridges.append(hidden_bridge)


        result = torch.cat(results,dim=1)
        bridge = torch.cat(bridges,dim=1)

        
        return result, bridge
    
    

class MQ_RNN(nn.Module):

    def __init__(self, input_dim, hidden_size, context_size, num_quantiles, pred_long, 
                        hist_long, bridge_size, num_layers=1):
        """Set the hyper-parameters and build the layers."""
        super(MQ_RNN, self).__init__()

        self.decoder = Decoder_MLP(context_size, hidden_size, context_size, num_quantiles, 
                            pred_long, bridge_size)
        self.lstm = nn.LSTM(context_size+1, hidden_size, num_layers, batch_first=True)
        
        self.linear_encoder = nn.Linear(input_dim, context_size)  

        self.input_dim = input_dim
        self.num_quantiles = num_quantiles
        self.pred_long = pred_long
        self.hist_long = hist_long
        self.total_long = pred_long + hist_long
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights."""
        # Disguised initialization

    
    def forward(self, x_seq_hist, y_seq_hist, x_seq_pred):
        
        bsize = x_seq_hist.size(0)

        assert x_seq_hist.size(1) == self.hist_long
        assert x_seq_hist.size(2) == self.input_dim
        assert x_seq_pred.size(1) == self.pred_long

        x_feat_hist = torch.tanh(self.linear_encoder(x_seq_hist))
        x_feat_pred = torch.tanh(self.linear_encoder(x_seq_pred))
        
        
        self.lstm.flatten_parameters()
        
        y_seq_hist_1 = y_seq_hist.view(y_seq_hist.size(0), y_seq_hist.size(1), 1)

        x_total_hist =  torch.cat([x_feat_hist,y_seq_hist_1], dim=2)
        x_total_pred =  torch.cat([x_feat_pred], dim=2)

        hiddens, (ht, c) = self.lstm(x_total_hist)
        
        ht = ht.view(ht.size(1),ht.size(2))

        result, hidden_bridge = self.decoder(ht, x_total_pred)
        result = result.view(-1, rnn_pred_long*num_quantiles)
        hidden_bridge = result.view(-1, rnn_pred_long*num_quantiles)

        return result, hidden_bridge


class End2End_v6_tc(nn.Module):


    def __init__(self, device, tf_graph=False):
        
        super(End2End_v6_tc, self).__init__()
        self.name='v6'
        self.device = device
        self.tf_graph = tf_graph

        self.cat_dim = len(CAT_FEA_HOT)
        self.vlt_dim = len(VLT_FEA)
        self.sf_dim = len(SF_FEA)
        self.oth_dim = len(MORE_FEA)
        self.is_dim = len(IS_FEA)
        # details of CAT_FEA_HOT, VLT_FEA, SF_FEA, MORE_FEA, and IS_FEA is provided in the document "original_E2E_section4.pdf". Detailed data of these features is confidential.

        self.input_dim =  self.vlt_dim + self.sf_dim + self.oth_dim + self.is_dim +self.cat_dim
        self.hidden_dim = Dimension # Disguised dimension sizes. Up to user's choice.
        self.output_dim = 1

        # Disguised initialization for fully connected layers, due to confidential dimension sizes.
        self.fc_vlt_1 = FULLY_CONNECTED_LAYER
        self.fc_vlt_2 = FULLY_CONNECTED_LAYER
        self.fc_vlt_3 = FULLY_CONNECTED_LAYER
        self.fc_3 = FULLY_CONNECTED_LAYER
        self.fc_4 = FULLY_CONNECTED_LAYER
        self.fc_4 = FULLY_CONNECTED_LAYER
        self.fc_5 = FULLY_CONNECTED_LAYER

        self.sf_mqrnn = MQ_RNN(rnn_input_dim, rnn_hidden_len, rnn_cxt_len, num_quantiles,
                               rnn_pred_long, rnn_hist_long, self.hidden_dim[2][2])
        self.init_weights()


    def init_weights(self):
        """Initialize weights."""
        # Disguised initialization



    def forward(self, enc_X, enc_y, dec_X, x_vlt, x_cat, x_oth, x_is):

        x1 = self.fc_vlt_1(torch.cat([x_vlt, x_cat], 1).float())
        x1 = F.relu(x1)
        x1 = self.fc_vlt_2(x1)
        x1 = F.relu(x1)
        o_vlt = self.fc_vlt_3(x1)

        o_sf, h_sf = self.sf_mqrnn(enc_X, enc_y, dec_X)
        x = self.fc_3(torch.cat([x1, h_sf, x_oth, x_is],1))
        x = F.relu(x)
        x = self.fc_4(x)
        x = F.relu(x)
        x = self.fc_5(x)
        return x, o_vlt, o_sf



