import torch 
import math

def diff_matrix(x, y):
    x_m = x.repeat(y.shape[0], 1)
    x_m = torch.transpose(x_m, 0, 1)
    y_m = y.repeat(x.shape[0], 1)
    return x_m - y_m

def corr(s_pos,g_pos,y_pos,s_neg,g_neg,y_neg):
    s_pos = s_pos.reshape(-1, 1)
    y_pos = y_pos.reshape(-1, 1)
    g_pos = g_pos.reshape(-1, 1)
    s_neg = s_neg.reshape(-1, 1)
    y_neg = y_neg.reshape(-1, 1)
    g_neg = g_neg.reshape(-1, 1)
    
    s_matrix = diff_matrix(s_pos, s_neg)
    g_matrix = diff_matrix(g_pos, g_neg)
    y_matrix = diff_matrix(y_pos, y_neg)
    a_matrix = s_matrix * y_matrix
    b_matrix = g_matrix * y_matrix
    cor = (torch.mean(a_matrix*b_matrix)-torch.mean(a_matrix)*torch.mean(b_matrix))/torch.std(a_matrix)/torch.std(b_matrix)
    
    if torch.isnan(cor):
        cor = 0
    
    print(cor)
    return cor

def single_loss(pred_pos, pred_neg):
    
    pred_pos = pred_pos.reshape(-1, 1)
    pred_neg = pred_neg.reshape(-1, 1)
    
    for p in range(pred_pos.shape[0]):
        diff = diff_matrix(pred_neg, pred_pos)
        loss = torch.mean((1+diff)**2)
    return loss

class loss_con1(torch.nn.Module):
    def __init__(self):
        super(loss_con1, self).__init__()
    
    def forward(self, y_pred_pos, y_pred_neg, s_pos, s_neg, y_pos, y_neg, g_pos, g_neg, lamb):
        
        obj = single_loss(y_pred_pos, y_pred_neg)
        correlation = abs(corr(s_pos,g_pos,y_pos,s_neg,g_neg,y_neg))
        
        loss = obj + lamb * correlation
                              
        return loss
    