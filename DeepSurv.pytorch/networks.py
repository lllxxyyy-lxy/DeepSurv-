from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

class Regularization(object):
    def __init__(self, order, weight_decay):
      
        super(Regularization, self).__init__()
        self.order = order
        self.weight_decay = weight_decay

    def __call__(self, model):
       
        reg_loss = 0
        for name, w in model.named_parameters():
            if 'weight' in name:
                reg_loss = reg_loss + torch.norm(w, p=self.order)
        reg_loss = self.weight_decay * reg_loss
        return reg_loss

class DeepSurv(nn.Module):
   
    def __init__(self, config):
        super(DeepSurv, self).__init__()
        self.drop = config['drop']
        self.norm = config['norm']
        self.dims = config['dims']
        self.activation = config['activation']
        self.model = self._build_network()

    def _build_network(self):
        layers = []
        for i in range(len(self.dims)-1):
            if i and self.drop is not None: 
                layers.append(nn.Dropout(self.drop))
           
            layers.append(nn.Linear(self.dims[i], self.dims[i+1]))
            if self.norm: 
                layers.append(nn.BatchNorm1d(self.dims[i+1]))
            
            layers.append(eval('nn.{}()'.format(self.activation)))
        
        return nn.Sequential(*layers)

    def forward(self, X):
        return self.model(X)

class NegativeLogLikelihood(nn.Module):
    def __init__(self, config):
        super(NegativeLogLikelihood, self).__init__()
        self.L2_reg = config['l2_reg']
        self.reg = Regularization(order=2, weight_decay=self.L2_reg)
    def forward(self, risk_pred, y, e, model):
        mask = torch.ones(y.shape[0], y.shape[0])
        mask[(y.T - y) > 0] = 0
        log_loss = torch.exp(risk_pred) * mask
        log_loss = torch.sum(log_loss, dim=0) / torch.sum(mask, dim=0)
        log_loss = torch.log(log_loss).reshape(-1, 1)
        neg_log_loss = -torch.sum((risk_pred-log_loss) * e) / torch.sum(e)
        l2_loss = self.reg(model)
        return neg_log_loss + l2_loss
    
