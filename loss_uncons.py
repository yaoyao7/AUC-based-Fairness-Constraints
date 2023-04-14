import torch 

def single_loss(aa, cc, y_pos_pred, y_neg_pred, a, b, alpha):
    y_pos_pred = y_pos_pred.reshape(-1,1)
    y_neg_pred = y_neg_pred.reshape(-1,1)
    
    loss = cc**2 + torch.mean( (y_pos_pred-a)**2 ) + torch.mean( (y_neg_pred-b)**2 ) \
           - 2*(cc+alpha)*torch.mean(y_pos_pred) + 2*(cc+alpha)*torch.mean(y_neg_pred) - alpha**2
    loss *= aa
    
    return loss


class loss_auc(torch.nn.Module):
    def __init__(self):
        super(loss_auc, self).__init__()
        self.a = torch.zeros(1, dtype=torch.float32, device="cuda", requires_grad=True).cuda()
        self.b = torch.zeros(1, dtype=torch.float32, device="cuda", requires_grad=True).cuda()
        self.alpha = torch.zeros(1, dtype=torch.float32, device="cuda", requires_grad=True).cuda()
        
    def forward(self, y_pos_pred, y_neg_pred, aa, cc):
        
        loss = single_loss(aa, cc, y_pos_pred, y_neg_pred, self.a, self.b, self.alpha)
                              
        return loss