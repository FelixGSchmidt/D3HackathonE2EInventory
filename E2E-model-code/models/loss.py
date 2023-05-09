# -*- coding: utf-8 -*-
# @Author: chenxinma
# @Date:   2018-10-09 11:00:00
# @Last Modified by:   Yuanyuan Shi, Meng Qi
# @Last Modified at:   2022-05-15 21:08:58


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import pickle  
import pandas as pd  
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from configs.config import *
import datetime as dt




class MeanSquareLossNew(nn.Module):
    def __init__(self, y_norm=True, size_average=True, use_square=False):
        super(MeanSquareLossNew, self).__init__()
        #self.quantile = quantile
        self.size_average = size_average
        self.y_norm = y_norm
        self.use_square = use_square
    
    # score is N x 31
    def forward(self, input, target):
        
        if self.use_square:
            input = torch.pow(input, 2)
        loss = (input - target) ** 2 
        
        if self.y_norm:
            loss /= max(1.0, target.sum())
        
        return loss.mean() if self.size_average else loss.sum()


class QauntileLoss(nn.Module):
    def __init__(self, y_norm=True, size_average=True, use_square=True):
        super(QauntileLoss, self).__init__()
        #self.quantile = quantile
        self.size_average = size_average
        self.y_norm = y_norm
        self.use_square = use_square
    
    # score is N x 31
    def forward(self, input, target, quantile):
        
        if self.use_square:
            input = torch.pow(input, 2)
        
        #print(input.size(),target.size())
        diff = input - target
        zero_1 = torch.zeros_like(diff)#.cuda()
        zero_2 = torch.zeros_like(diff)#.cuda()
        loss = quantile * torch.max(-diff,zero_1) + (1-quantile) * torch.max(diff,zero_2)
        
        #print(target.size(), target.sum())
        if self.y_norm:
            loss /= max(1.0, target.sum())
        
        return loss.mean() if self.size_average else loss.sum()
        


class E2E_v6_loss(nn.Module):
    def __init__(self, device, qtl, size_average=True):
        super(E2E_v6_loss, self).__init__()

        self.device = device
        self.size_average = size_average
        self.lmd_1 = VALUE1 # Scalared value. Up to user's choice. Disguised due to confidentiality.
        self.lmd_2 = VALUE2 # Scalared value. Up to user's choice. Disguised due to confidentiality.
        self.quantile_loss = QauntileLoss(y_norm=False, size_average=True, use_square=False)

    # score is N x 31
    def forward(self, vlt_o, vlt_t, sf_o, sf_t, out, target):
        # Disguised loss function
        self.q_loss = LOSS_FUN1(out, target) #LOSS_FUN1 up to user's choice. Can be self-defined or nn.MSELoss, QauntileLoss, etc.
        self.vlt_loss = LOSS_FUN2(vlt_o, vlt_t) #LOSS_FUN2 up to user's choice. Can be self-defined or nn.MSELoss, QauntileLoss, etc.
        self.sf_loss = LOSS_FUN3(sf_o, sf_t) # Loss_FUN3 should be quantile losses summed over each quantile
        self.loss = self.qtl_loss + self.lmd_1 * self.vlt_loss + self.lmd_2 * self.sf_loss

        return self.qtl_loss, self.loss


  