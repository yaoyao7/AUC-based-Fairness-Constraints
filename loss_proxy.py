import torch 
import math

def diff_matrix(x, y):
   x_m = x.repeat(y.shape[0], 1)
   x_m = torch.transpose(x_m, 0, 1)
   y_m = y.repeat(x.shape[0], 1)
   return x_m - y_m

def single_loss_lamb(pred_pos, pred_neg):
    
    pred_pos = pred_pos.reshape(-1, 1)
    pred_neg = pred_neg.reshape(-1, 1)
    
    for p in range(100):
        diff = diff_matrix(pred_neg, pred_pos)
        diff[diff <= 0] = 0
        diff[diff > 0] = 1
        loss = torch.mean(diff)
    return loss

def single_loss(pred_pos, pred_neg):
    
    pred_pos = pred_pos.reshape(-1, 1)
    pred_neg = pred_neg.reshape(-1, 1)
    
    for p in range(100):
        diff = diff_matrix(pred_neg, pred_pos)
        loss = torch.mean((1+diff)**2)
    return loss

def single_constraint_lamb(pred_pos, pred_neg):
    pred_pos = pred_pos.reshape(-1, 1)
    pred_neg = pred_neg.reshape(-1, 1)
    
    for p in range(100):
        diff = diff_matrix(pred_pos, pred_neg)
        diff[diff <= 0] = 0
        diff[diff > 0] = 1
        constraint = torch.mean(diff)
    return constraint

def single_constraint(pred_pos, pred_neg):
    
    pred_pos = pred_pos.reshape(-1, 1)
    pred_neg = pred_neg.reshape(-1, 1)
    
    for p in range(100):
        diff = diff_matrix(pred_pos, pred_neg)
        constraint = torch.mean((1+diff)**2)
    return constraint

class loss_con1(torch.nn.Module):
    def __init__(self):
        super(loss_con1, self).__init__()
    
    def forward(self, y_pred_pos, y_pred_neg, g_pred_pos, g_pred_neg, lam):
        
        loss1 = single_loss(y_pred_pos, y_pred_neg)
        loss2 = single_constraint(g_pred_pos, g_pred_neg)
        loss3 = single_constraint(g_pred_neg, g_pred_pos)
                    
        loss = lam[0]*loss1 + lam[1]*loss2 + lam[2]*loss3
                              
        return loss
    
class loss_con2(torch.nn.Module):
    def __init__(self):
        super(loss_con2, self).__init__()
    
    def forward(self, y_pred_pos,y_pred_neg,y_pos_g_pos,y_pos_g_neg,y_neg_g_pos,y_neg_g_neg, lam):
        
        loss1 = single_loss(y_pred_pos, y_pred_neg)
        loss2 = single_constraint(y_pos_g_neg, y_neg_g_pos) + single_constraint(y_neg_g_neg, y_pos_g_pos)
        loss3 = single_constraint(y_pos_g_pos, y_neg_g_neg) + single_constraint(y_neg_g_pos, y_pos_g_neg)
                    
        loss = lam[0]*loss1 + lam[1]*loss2 + lam[2]*loss3
                              
        return loss
    
class loss_con3(torch.nn.Module):
    def __init__(self):
        super(loss_con3, self).__init__()
    
    def forward(self, y_pred_pos,y_pred_neg,y_pos_g_pos,y_pos_g_neg,y_neg_g_pos,y_neg_g_neg, lam):
        
        loss1 = single_loss(y_pred_pos, y_pred_neg)
        loss2 = single_constraint(y_pos_g_pos, y_neg_g_pos) + single_constraint(y_neg_g_neg, y_pos_g_neg)
        loss3 = single_constraint(y_pos_g_neg, y_neg_g_neg) + single_constraint(y_neg_g_pos, y_pos_g_pos)
                    
        loss = lam[0]*loss1 + lam[1]*loss2 + lam[2]*loss3
                              
        return loss