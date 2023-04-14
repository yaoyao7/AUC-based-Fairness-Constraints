
import torch
import math
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from loss_uncons import loss_auc
from torch.utils.data import TensorDataset, DataLoader
from parameters import para
from model import NeuralNetwork
from commonutil import svm_read_problem
from sklearn.model_selection import train_test_split
from scipy.io import savemat


def set_all_seeds(SEED):
    # REPRODUCIBILITY
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def split_y_pos(y):
    index = torch.where(y > 0.1)
    return index

def split_y_neg(y):
    index = torch.where(y < 0.1)
    return index

# all paramaters
dataname = 'a9a'
SEED = para.seed
BATCH_SIZE = para.batch
pos = 71   # sensitive position
T = para.T
lr_0 = para.lr_0
m = 2  # number of constraint

aa = 0.5
cc = 1

#torch.manual_seed(SEED)
set_all_seeds(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# import data
y, X_train = svm_read_problem('../dataset/'+str(dataname)+'/'+str(dataname), True)
X = X_train.toarray()

# import test data
y_test, X_ttest = svm_read_problem('../dataset/'+str(dataname)+'/'+str(dataname)+'_t', True)
X_test = X_ttest.toarray()

set_all_seeds(SEED)

# split training and validation set
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=SEED)
# transform data to torch tensor
X_train = torch.Tensor(X_train)
y_train = torch.Tensor(y_train)
X_val = torch.Tensor(X_val)
y_val = torch.Tensor(y_val)
X_test = torch.Tensor(X_test)
y_test = torch.Tensor(y_test)

# a9a test data 1 col less than train data
if dataname == 'a9a':
    col = torch.zeros(X_test.shape[0],1)
    X_test = torch.cat((X_test, col),1)

# split data
index_y_pos = split_y_pos(y_train)
X_y_pos = X_train[index_y_pos]
y_y_pos = y_train[index_y_pos]

index_y_neg = split_y_neg(y_train)
X_y_neg = X_train[index_y_neg]
y_y_neg = y_train[index_y_neg]

trainSet_y_pos = TensorDataset(X_y_pos, y_y_pos)
trainSet_y_neg = TensorDataset(X_y_neg, y_y_neg)

trainloader_y_pos = DataLoader(trainSet_y_pos, batch_size=BATCH_SIZE,
                               num_workers=0, pin_memory=True, drop_last=True)
trainloader_y_neg = DataLoader(trainSet_y_neg, batch_size=BATCH_SIZE,
                               num_workers=0, pin_memory=True, drop_last=True)

# model
set_all_seeds(SEED)
hidden_dim = X_train.shape[1]
output_dim = 1
model = NeuralNetwork(X_train.shape[1], hidden_dim, output_dim).cuda()

# training
if True:
    train_auc_list = []
    val_auc_list = []   
    test_auc_list = []
    
    best_auc = 0
    best_x = []
    for var in list(model.parameters()):
        best_x.append(torch.zeros(var.shape, dtype=torch.float32, device=device, requires_grad=False).to(device))
    
    Loss = loss_auc()
    
    trainloader_copy_y_pos = iter(trainloader_y_pos)
    trainloader_copy_y_neg = iter(trainloader_y_neg)
  
    for t in range(T):
        lr = lr_0
        model.train()
        
        try:
            data_y_pos, target_y_pos = trainloader_copy_y_pos.next()
        except:
            trainloader_copy_y_pos = iter(trainloader_y_pos)
            
        try:
            data_y_neg, target_y_neg = trainloader_copy_y_neg.next()
        except:
            trainloader_copy_y_neg = iter(trainloader_y_neg)

            
        data_y_pos, target_y_pos = data_y_pos.cuda(), target_y_pos.cuda()
        data_y_neg, target_y_neg = data_y_neg.cuda(), target_y_neg.cuda()
                
        # compute score
        logits_y_pos = model(data_y_pos)
        logits_y_neg = model(data_y_neg)
        
        loss = Loss(logits_y_pos, logits_y_neg, aa, cc)
        
        paras = list(model.parameters())+[Loss.a, Loss.b]
        
        # set zero grad, otherwise grad will accumulate
        for pp in paras:
            if pp.requires_grad == True:
                pp.grad = torch.zeros_like(pp.data)
        
        Loss.alpha.grad = torch.zeros_like(Loss.alpha.data)
        
        # compute gradients of model parameters
        loss.backward()
        
        for pp in paras:
            if pp.requires_grad == True:
                pp.data -= lr * pp.grad
        
        Loss.alpha.data += lr * Loss.alpha.grad
        
        
        if t % 500 == 0:
            model.eval()
            
            X_test = X_test.cuda()
            test_pred = list(model(X_test).cpu().detach().numpy())
            test_true = list(y_test.cpu().numpy())
            test_auc = roc_auc_score(test_true, test_pred)

            print("test_auc:{:4f}".format(test_auc))   
            
            if test_auc > best_auc:
                best_auc = test_auc
                for k, w in enumerate(list(model.parameters())):
                    best_x[k].data = w.data

    for k, w in enumerate(list(model.parameters())):
        w.data = best_x[k].data

    best_s = list(model(X_test).cpu().detach().numpy())
    score = {"score":best_s}
    savemat("s_"+str(dataname)+"_"+str(SEED)+".mat", score)
                          