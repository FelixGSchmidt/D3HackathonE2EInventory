
# -*- coding: utf-8 -*-
# @Author: chenxinma
# @Date:   2018-10-01 16:04:49
# @Last Modified by:   Yuanyuan Shi
# @Last Modified at:   2019-08-16 10:07:03

from models.model import *
from models.loss import *
from data_loader.data_loader import *
import os
from subprocess import call
from tensorboardX import SummaryWriter
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import time 
# import seaborn as sns

class Trainer(object):

    def __init__(self, model, args, device):
        
        self.model = model

        self.batch_size = args.bs
        self.model_name = args.model_name
        self.data_path = args.data_path
        self.model_path = args.model_path
        self.log_path = args.log_path
        self.num_epochs = args.num_epochs
        self.model_to_load = args.model_to_load
        self.pred_to_save = args.pred_to_save
        self.train_check = args.train_check
        self.log_step = args.log_step
        self.save_step = args.save_step
        self.learning_rate = args.learning_rate
        self.test_sku = args.test_sku
        self.b_value = args.b_value

        self.device = device

        

    def train_v6_tc(self):


        if os.path.exists('%s/%s' %(self.log_path, self.model_name)):
            call(['rm', '-r', '%s/%s' %(self.log_path, self.model_name)])
        call(['mkdir', '%s/%s' %(self.log_path, self.model_name)])

        train_writer = SummaryWriter('%s/%s/train/' %(self.log_path, self.model_name))
        test_writer = SummaryWriter('%s/%s/test/' %(self.log_path, self.model_name))


        data_loader, test_loader = get_loader(self.batch_size, self.device, self.model_name, b=self.b_value)
        test_loader = data_loader
        print('train_loader', len(data_loader),'test_loader', len(test_loader))

        qtl = self.b_value / (self.b_value + 1) - .1
        e2e_loss = E2E_v6_loss(self.device, qtl)

        params = list(self.model.parameters())
        optimizer = torch.optim.Adam(params, lr=self.learning_rate)
        curr_epoch = 0

        if self.train_check != 'None':
            checkpoint = torch.load(self.model_path+self.train_check, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            curr_epoch = checkpoint['epoch']
            loss = checkpoint['loss']
            print('load model!')

        start = time.time()
        for epoch in range(curr_epoch, self.num_epochs):
            train_loss0, train_loss1 = 0, 0

            for i, (X, S1, S2) in enumerate(data_loader):
                X = X.float()
                S1 = S1.float()
                S2 = S2.float()
                out, out_vlt, out_sf = self.model(S1[:,:,:2], S1[:,:,2:], S2[:,:,:2], X[:,:p1], X[:,p2:p3], X[:,p3:p4], X[:,p4:p5])
                batch_loss1, batch_loss0 = e2e_loss(out_vlt, X[:,p6:p7], out_sf, S2[:,:,2], out, X[:,p5:p6])

                # if epoch>0:
                optimizer.zero_grad()
                batch_loss0.backward()
                optimizer.step()

                train_loss1 += batch_loss1.item()
                train_loss0 += batch_loss0.item()

                if (i+1) % self.log_step == 0:

                    test_loss0, test_loss1 = 0, 0
                    for _, (X, S1, S2) in enumerate(test_loader):
                        X = X.float()
                        S1 = S1.float()
                        S2 = S2.float()
                        out, out_vlt, out_sf = self.model(S1[:,:,:2], S1[:,:,2:], S2[:,:,:2], X[:,:p1], X[:,p2:p3], X[:,p3:p4], X[:,p4:p5])
                        loss1, loss0 = e2e_loss(out_vlt, X[:,p6:p7], out_sf, S2[:,:,2], out, X[:,p5:p6])
                        test_loss1 += loss1.item()
                        test_loss0 += loss0.item()

                    print('Epoch %d pct %.3f, loss_1 %.5f, loss_ttl %.5f, test_loss_1 %.5f, test_loss_ttl %.5f' % 
                                    (epoch,(i+1)/len(data_loader),train_loss1/self.log_step,train_loss0/self.log_step,
                                        test_loss1/len(test_loader), test_loss0/len(test_loader)))

                    for name, param in self.model.named_parameters():
                        train_writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch*len(data_loader)+i)
                    train_writer.add_scalar('train_loss_out',train_loss1/self.log_step, epoch*len(data_loader)+i)
                    train_writer.add_scalar('train_loss_ttl',train_loss0/self.log_step, epoch*len(data_loader)+i)
                    test_writer.add_scalar('test_loss_out', test_loss1/len(test_loader), epoch*len(data_loader)+i)
                    test_writer.add_scalar('test_loss_ttl', test_loss0/len(test_loader), epoch*len(data_loader)+i)
                    train_writer.export_scalars_to_json('%s/%s/train/scalars_train.json' %(self.log_path, self.model_name))
                    test_writer.export_scalars_to_json('%s/%s/test/scalars_test.json' %(self.log_path, self.model_name))
                    train_loss0, train_loss1 = 0, 0
            end = time.time()
            print('Training time:', end-start)
            if (epoch+1) % self.save_step == 0:
                # torch.save(model.state_dict(), os.path.join('../logs/torch/', 'e2e_v6_%d.pkl' %(epoch+1)))
                torch.save({
                            'epoch': epoch+1,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': batch_loss1,
                            }, os.path.join('../logs/torch4/', 'e2e_%s_%d.pkl' %(self.model_name,epoch+1)))

        train_writer.close()
        test_writer.close()

    def eval_v6_tc(self):
        _, test_loader = get_loader(self.batch_size, self.device, self.model_name, eval=1, test_sku=self.test_sku, b=self.b_value)

        checkpoint = torch.load(self.model_path+self.model_to_load, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print('load model!')
        
        for _, (X, S1, S2) in enumerate(test_loader):
            out, out_vlt, out_sf = self.model(S1[:,:,:2], S1[:,:,2:], S2[:,:,:2], X[:,:p1], X[:,p2:p3], X[:,p3:p4], X[:,p4:p5])

        pd_scaler = pd.read_csv('../data2/1320_feature/scaler.csv')
        df_idx = pd.read_csv('../logs/torch/pred_sku.csv')
        if self.b_value != None:
            LABEL = ['demand_RV_%i' %self.b_value]
        out = out.detach().cpu().numpy() / pd_scaler.loc[1, LABEL[0]] + pd_scaler.loc[0, LABEL[0]]

        out_sf = out_sf.view(-1, rnn_pred_long, num_quantiles).detach().cpu().numpy()
        out_sf = np.exp(out_sf) - 1
        out_vlt = out_vlt.detach().cpu().numpy() / pd_scaler.loc[1, LABEL_vlt[0]] + pd_scaler.loc[0, LABEL_vlt[0]]
        pred = pd.DataFrame(out, columns=['E2E_RNN_pred']) 
        pred_vlt = pd.DataFrame(out_vlt, columns=['E2E_NN_vlt_pred'])
        pred = pd.concat([df_idx, pred, pred_vlt], axis=1)
        if self.pred_to_save == 'None':
            pred.to_csv('%s/pred_v6.csv' %self.model_path, index=False)
        else:
            pred.to_csv('%s/%s' %(self.model_path, self.pred_to_save), index=False)

        out_sf.dump('%s/pred_E2E_SF_RNN.pkl'  %self.model_path)
        out_vlt.dump('%s/pred_E2E_VLT_RNN.pkl'  %self.model_path)





