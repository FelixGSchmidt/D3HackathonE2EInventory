# -*- coding: utf-8 -*-
# @Author: chenxinma
# @Date:   2018-10-01 16:42:42
# @Last Modified by:   Yuanyuan Shi, Meng Qi
# @Last Modified at:   2022-08-28 18:18:03


IDX = ['item_sku_id','sku_id','create_tm','complete_dt']

CAT_FEA = [
    DISGUISED FEATURE DETAILS
    ]

VLT_FEA = [
    DISGUISED FEATURE DETAILS
        ]

SF_FEA = [
    DISGUISED FEATURE DETAILS
            ]
   
MORE_FEA =[
    DISGUISED FEATURE DETAILS
            ]

IS_FEA = [
    DISGUISED FEATURE DETAILS
        ]
    
CAT_FEA_HOT = [
    DISGUISED FEATURE DETAILS
             ]

TO_SCALE = [
    DISGUISED FEATURE DETAILS
            ]

SEQ2SEQ = ['Enc_X', 'Enc_y', 'Dec_X', 'Dec_y']


# LABEL = ['target_decision']    
LABEL = ['demand_RV_dp']    
LABEL_vlt = ['vlt_actual']    
LABEL_sf = ['label_sf']    

p1 = len(VLT_FEA) 
p2 = p1 + len(SF_FEA)
p3 = p2 + len(CAT_FEA_HOT)
p4 = p3 + len(MORE_FEA)
p5 = p4 + len(IS_FEA)
p6 = p5 + 1
p7 = p6 + 1
p8 = p7 + 1


SCALE_FEA =  VLT_FEA + SF_FEA + CAT_FEA_HOT + MORE_FEA + IS_FEA  + LABEL_vlt + LABEL_sf
CUT_FEA = VLT_FEA + SF_FEA + MORE_FEA
MODEL_FEA = VLT_FEA + SF_FEA + MORE_FEA + IS_FEA + CAT_FEA_HOT


#Disguised rnn parameters
rnn_hidden_len = DISGUISED #Possible values: 5~200
rnn_cxt_len = DISGUISED  #Possible values: 1~10
rnn_pred_long = DISGUISED #Possible values: 7~60
rnn_hist_long = DISGUISED #Possible values: 30~180
rnn_total_long = rnn_pred_long + rnn_hist_long
rnn_input_dim = DISGUISED #Possible value: 2
quantiles = [DISGUISED] # Example: [0.1, 0.3, 0.5, 0.7, 0.9]
num_quantiles = len(quantiles)