import torch 
import math

def single_loss(aa, cc, y_pos_pred, y_neg_pred, a, b, alpha):
    y_pos_pred = y_pos_pred.reshape(-1,1)
    y_neg_pred = y_neg_pred.reshape(-1,1)
    
    loss = cc**2 + torch.mean( (y_pos_pred-a)**2 ) + torch.mean( (y_neg_pred-b)**2 ) \
           - 2*(cc+alpha)*torch.mean(y_pos_pred) + 2*(cc+alpha)*torch.mean(y_neg_pred)
    loss *= aa
    return loss

def single_cons(aa, cc, y_pos_pred, y_neg_pred, a, b, alpha):
    y_pos_pred = y_pos_pred.reshape(-1,1)
    y_neg_pred = y_neg_pred.reshape(-1,1)
    
    loss = cc**2 + torch.mean( (y_pos_pred-a)**2 ) + torch.mean( (y_neg_pred-b)**2 ) \
           + 2*(cc+alpha)*torch.mean(y_pos_pred) - 2*(cc+alpha)*torch.mean(y_neg_pred)
    loss *= aa
    return loss

class loss_con1(torch.nn.Module):
    def __init__(self):
        super(loss_con1, self).__init__()
        self.a = torch.zeros((3,1), dtype=torch.float32, device="cuda", requires_grad=True).cuda()
        self.b = torch.zeros((3,1), dtype=torch.float32, device="cuda", requires_grad=True).cuda()
        
    def forward(self, y_pos_pred, y_neg_pred, g_pos_pred, g_neg_pred, lam, alp, aa, cc):
        
        loss1 = single_loss(aa, cc, y_pos_pred, y_neg_pred, self.a[0], self.b[0], alp[0])
        loss2 = single_cons(aa, cc, g_pos_pred, g_neg_pred, self.a[1], self.b[1], alp[1])
        loss3 = single_cons(aa, cc, g_neg_pred, g_pos_pred, self.a[2], self.b[2], alp[2])
            
        loss = lam[0]*loss1 + lam[1]*loss2 + lam[2]*loss3
                              
        return loss

class loss_con2(torch.nn.Module):
    def __init__(self):
        super(loss_con2, self).__init__()
        self.a = torch.zeros((5,1), dtype=torch.float32, device="cuda", requires_grad=True).cuda()
        self.b = torch.zeros((5,1), dtype=torch.float32, device="cuda", requires_grad=True).cuda()
    
    def forward(self, y_pos_pred,y_neg_pred,y_pos_g_pos,y_pos_g_neg,y_neg_g_pos,y_neg_g_neg, lam, alp, aa, cc):
        
        loss1 = single_loss(aa, cc, y_pos_pred, y_neg_pred, self.a[0], self.b[0], alp[0])
        loss2 = single_cons(aa, cc, y_pos_g_neg, y_neg_g_pos, self.a[1], self.b[1], alp[1])+single_cons(aa, cc, y_neg_g_neg, y_pos_g_pos, self.a[2], self.b[2], alp[2])
        loss3 = single_cons(aa, cc, y_pos_g_pos, y_neg_g_neg, self.a[3], self.b[3], alp[3])+single_cons(aa, cc, y_neg_g_pos, y_pos_g_neg, self.a[4], self.b[4], alp[4])    
                    
        loss = lam[0]*loss1 + lam[1]*loss2 + lam[2]*loss3
                              
        return loss


class loss_con3(torch.nn.Module):
    def __init__(self):
        super(loss_con3, self).__init__()
        self.a = torch.zeros((5,1), dtype=torch.float32, device="cuda", requires_grad=True).cuda()
        self.b = torch.zeros((5,1), dtype=torch.float32, device="cuda", requires_grad=True).cuda()
    
    def forward(self, y_pos_pred,y_neg_pred,y_pos_g_pos,y_pos_g_neg,y_neg_g_pos,y_neg_g_neg, lam, alp, aa, cc):
        
        loss1 = single_loss(aa, cc, y_pos_pred, y_neg_pred, self.a[0], self.b[0], alp[0])
        loss2 = single_cons(aa, cc, y_pos_g_pos, y_neg_g_pos, self.a[1], self.b[1], alp[1])+single_cons(aa, cc, y_neg_g_neg, y_pos_g_neg, self.a[2], self.b[2], alp[2])
        loss3 = single_cons(aa, cc, y_pos_g_neg, y_neg_g_neg, self.a[3], self.b[3], alp[3])+single_cons(aa, cc, y_neg_g_pos, y_pos_g_pos, self.a[4], self.b[4], alp[4])    
                    
        loss = lam[0]*loss1 + lam[1]*loss2 + lam[2]*loss3
                              
        return loss    
