import torch
import math
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from loss_proxy import loss_con2
from loss_proxy import single_loss_lamb, single_constraint_lamb
from commonutil import svm_read_problem
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from torch import nn
from parameters import para
import numpy.linalg as LA
from model import NeuralNetwork

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

def split_y_pos_g_pos(y, X, pos):
    index = torch.where((y > 0.1) & (X[:,pos] > 0.1))
    return index

def split_y_pos_g_neg(y, X, pos):
    index = torch.where((y > 0.1) & (X[:,pos] < 0.1))
    return index

def split_y_neg_g_pos(y, X, pos):
    index = torch.where((y < 0.1) & (X[:,pos] > 0.1))
    return index

def split_y_neg_g_neg(y, X, pos):
    index = torch.where((y < 0.1) & (X[:,pos] < 0.1))
    return index

def add_col(X, pos):
    total_feat = X.shape[1]
    sens_feat = X[:,pos].reshape(-1,1)
    pre_sen = sens_feat.repeat(1, pos)
    pos_sen = sens_feat.repeat(1, (total_feat-pos-2))
    new_pre = X[:,0:pos] * pre_sen
    new_pos = X[:,(pos+2):total_feat] * pos_sen
    new_X = torch.cat((X, new_pre, new_pos), 1)
    return new_X

# all parameters
dataname = 'a9a'
SEED = para.seed
BATCH_SIZE = para.batch
pos = 71   # sensitive position
T = para.T
lr_0 = para.lr_0
c = para.c
fair_con = para.fair_con
m = 2

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

index_y_pos_g_pos = split_y_pos_g_pos(y_train, X_train, pos)
X_y_pos_g_pos = X_train[index_y_pos_g_pos]
y_y_pos_g_pos = y_train[index_y_pos_g_pos]

index_y_pos_g_neg = split_y_pos_g_neg(y_train, X_train, pos)
X_y_pos_g_neg = X_train[index_y_pos_g_neg]
y_y_pos_g_neg = y_train[index_y_pos_g_neg]

index_y_neg_g_pos = split_y_neg_g_pos(y_train, X_train, pos)
X_y_neg_g_pos = X_train[index_y_neg_g_pos]
y_y_neg_g_pos = y_train[index_y_neg_g_pos]

index_y_neg_g_neg = split_y_neg_g_neg(y_train, X_train, pos)
X_y_neg_g_neg = X_train[index_y_neg_g_neg]
y_y_neg_g_neg = y_train[index_y_neg_g_neg]

val_i_y_pos_g_pos = split_y_pos_g_pos(y_val, X_val, pos)
val_i_y_pos_g_neg = split_y_pos_g_neg(y_val, X_val, pos)
val_i_y_neg_g_pos = split_y_neg_g_pos(y_val, X_val, pos)
val_i_y_neg_g_neg = split_y_neg_g_neg(y_val, X_val, pos)
val_y_y_pos_g_pos = y_val[val_i_y_pos_g_pos]
val_y_y_pos_g_neg = y_val[val_i_y_pos_g_neg]
val_y_y_neg_g_pos = y_val[val_i_y_neg_g_pos]
val_y_y_neg_g_neg = y_val[val_i_y_neg_g_neg]

test_i_y_pos_g_pos = split_y_pos_g_pos(y_test, X_test, pos)
test_i_y_pos_g_neg = split_y_pos_g_neg(y_test, X_test, pos)
test_i_y_neg_g_pos = split_y_neg_g_pos(y_test, X_test, pos)
test_i_y_neg_g_neg = split_y_neg_g_neg(y_test, X_test, pos)
test_y_y_pos_g_pos = y_test[test_i_y_pos_g_pos]
test_y_y_pos_g_neg = y_test[test_i_y_pos_g_neg]
test_y_y_neg_g_pos = y_test[test_i_y_neg_g_pos]
test_y_y_neg_g_neg = y_test[test_i_y_neg_g_neg]

trainSet_y_pos = TensorDataset(X_y_pos, y_y_pos)
trainSet_y_neg = TensorDataset(X_y_neg, y_y_neg)
trainSet_y_pos_g_pos = TensorDataset(X_y_pos_g_pos, y_y_pos_g_pos)
trainSet_y_pos_g_neg = TensorDataset(X_y_pos_g_neg, y_y_pos_g_neg)
trainSet_y_neg_g_pos = TensorDataset(X_y_neg_g_pos, y_y_neg_g_pos)
trainSet_y_neg_g_neg = TensorDataset(X_y_neg_g_neg, y_y_neg_g_neg)


trainloader_y_pos = DataLoader(trainSet_y_pos, batch_size=BATCH_SIZE,
                               num_workers=0, pin_memory=True, drop_last=True)
trainloader_y_neg = DataLoader(trainSet_y_neg, batch_size=BATCH_SIZE,
                               num_workers=0, pin_memory=True, drop_last=True)
trainloader_y_pos_g_pos = DataLoader(trainSet_y_pos_g_pos, batch_size=BATCH_SIZE,
                               num_workers=0, pin_memory=True, drop_last=True)
trainloader_y_pos_g_neg = DataLoader(trainSet_y_pos_g_neg, batch_size=BATCH_SIZE,
                               num_workers=0, pin_memory=True, drop_last=True)
trainloader_y_neg_g_pos = DataLoader(trainSet_y_neg_g_pos, batch_size=BATCH_SIZE,
                               num_workers=0, pin_memory=True, drop_last=True)
trainloader_y_neg_g_neg = DataLoader(trainSet_y_neg_g_neg, batch_size=BATCH_SIZE,
                               num_workers=0, pin_memory=True, drop_last=True)

hidden_dim = X_train.shape[1]
output_dim = 1
model = NeuralNetwork(X_train.shape[1], hidden_dim, output_dim).cuda()

# training

if True:
    train_auc_list = []
    train_con1_list = []
    train_con2_list = []
    
    val_auc_list = []
    val_con1_list = []
    val_con2_list = []
    
    test_auc_list = []
    test_con1_list = []
    test_con2_list = []
    diff_list = []
    
    Loss = loss_con2()
        
    # initialize lambda
    M = torch.ones([m+1,m+1], requires_grad=False)
    M /= m+1
    lamb = torch.zeros((3,1), dtype=torch.float32, device=device, requires_grad=False)
    
    c_trainloader_y_pos = iter(trainloader_y_pos)
    c_trainloader_y_neg = iter(trainloader_y_neg)
    c_trainloader_y_pos_g_pos = iter(trainloader_y_pos_g_pos)
    c_trainloader_y_pos_g_neg = iter(trainloader_y_pos_g_neg)
    c_trainloader_y_neg_g_pos = iter(trainloader_y_neg_g_pos)
    c_trainloader_y_neg_g_neg = iter(trainloader_y_neg_g_neg)
    
    # level set algorithm
    for k in range(T):
        print(k)
        
        M = M.detach().numpy()
        eigenvals, eigenvects = np.linalg.eig(M)
        '''
        Find the indexes of the eigenvalues that are close to one.
        Use them to select the target eigen vectors. Flatten the result.
        '''
        close_to_1_idx = np.isclose(eigenvals,1)
        target_eigenvect = eigenvects[:,close_to_1_idx]
        target_eigenvect = target_eigenvect[:,0]
        stationary_distrib = target_eigenvect / sum(target_eigenvect)
        stationary_distrib_real = stationary_distrib.real
        
        lamb = torch.from_numpy(stationary_distrib_real)
        
        lamb = lamb.cuda()
        
        lr = lr_0 / math.sqrt(k+1)
        
        model.train()
        
        # sample four batches
        try:
            data_y_pos, target_y_pos = c_trainloader_y_pos.next()
        except:
            c_trainloader_y_pos = iter(trainloader_y_pos)
            
        try:
            data_y_neg, target_y_neg = c_trainloader_y_neg.next()
        except:
            c_trainloader_y_neg = iter(trainloader_y_neg)
            
        try:
            data_y_pos_g_pos, target_y_pos_g_pos = c_trainloader_y_pos_g_pos.next()
        except:
            c_trainloader_y_pos_g_pos = iter(trainloader_y_pos_g_pos)
            
        try:
            data_y_pos_g_neg, target_y_pos_g_neg = c_trainloader_y_pos_g_neg.next()
        except:
            c_trainloader_y_pos_g_neg = iter(trainloader_y_pos_g_neg)
            
        try:
            data_y_neg_g_pos, target_y_neg_g_pos = c_trainloader_y_neg_g_pos.next()
        except:
            c_trainloader_y_neg_g_pos = iter(trainloader_y_neg_g_pos)
            
        try:
            data_y_neg_g_neg, target_y_neg_g_neg = c_trainloader_y_neg_g_neg.next()
        except:
            c_trainloader_y_neg_g_neg = iter(trainloader_y_neg_g_neg)
        
        data_y_pos, target_y_pos = data_y_pos.cuda(), target_y_pos.cuda()
        data_y_neg, target_y_neg = data_y_neg.cuda(), target_y_neg.cuda()
        data_y_pos_g_pos, target_y_pos_g_pos = data_y_pos_g_pos.cuda(), target_y_pos_g_pos.cuda()
        data_y_pos_g_neg, target_y_pos_g_neg = data_y_pos_g_neg.cuda(), target_y_pos_g_neg.cuda()
        data_y_neg_g_pos, target_y_neg_g_pos = data_y_neg_g_pos.cuda(), target_y_neg_g_pos.cuda()
        data_y_neg_g_neg, target_y_neg_g_neg = data_y_neg_g_neg.cuda(), target_y_neg_g_neg.cuda()
        
        # compute score
        logits_y_pos = model(data_y_pos)
        logits_y_neg = model(data_y_neg)
        logits_y_pos_g_pos = model(data_y_pos_g_pos)
        logits_y_pos_g_neg = model(data_y_pos_g_neg)
        logits_y_neg_g_pos = model(data_y_neg_g_pos)
        logits_y_neg_g_neg = model(data_y_neg_g_neg)
        
        loss = Loss(logits_y_pos, logits_y_neg, 
                    logits_y_pos_g_pos, logits_y_pos_g_neg,
                    logits_y_neg_g_pos, logits_y_neg_g_neg, lamb)
        
        # compute gradients of model parameters
        grads = torch.autograd.grad(loss, model.parameters())
        
        # update model parameter
        for g, (name, w)in zip(grads, model.named_parameters()):
            w.data -= lr * g
            
        # update lambda
        obj = single_loss_lamb(logits_y_pos, logits_y_neg)
        con1 = single_constraint_lamb(logits_y_pos_g_neg, logits_y_neg_g_pos) + \
               single_constraint_lamb(logits_y_neg_g_neg, logits_y_pos_g_pos)
        con2 = single_constraint_lamb(logits_y_pos_g_pos, logits_y_neg_g_neg) + \
               single_constraint_lamb(logits_y_neg_g_pos, logits_y_pos_g_neg)
        
        grad_lamb = torch.tensor([0, con1-c, con2-c], dtype=torch.float32, device="cuda")
        grad_lamb = grad_lamb.reshape(3,1)
        lamb = lamb.reshape(1,3)
        
        M = torch.from_numpy(M)
        
        M = M.cuda()
        
        M_tilde = torch.mul(M, torch.exp(lr * torch.matmul(grad_lamb,lamb)))
        for i in range(3):
            M[:,i] = M_tilde[:,i] / torch.norm(M_tilde[:,i], p=1)
        
        M = M.cpu()
        
        model.eval()
        
        X_train = X_train.cuda()
        train_score = model(X_train).detach()
        train_pred = list(train_score.cpu().numpy())
        train_true = list(y_train.cpu().numpy())
        train_auc = roc_auc_score(train_true, train_pred)
        train_pred_y_pos_g_pos = train_score[index_y_pos_g_pos]
        train_pred_y_pos_g_neg = train_score[index_y_pos_g_neg]
        train_pred_y_neg_g_pos = train_score[index_y_neg_g_pos]
        train_pred_y_neg_g_neg = train_score[index_y_neg_g_neg]
        
        train_pred_group1 = list(torch.cat((train_pred_y_neg_g_pos,train_pred_y_pos_g_neg),0).cpu().numpy())
        train_pred_group2 = list(torch.cat((train_pred_y_pos_g_pos,train_pred_y_neg_g_neg),0).cpu().numpy())
        train_true_group1 = list(torch.cat((y_y_neg_g_pos,y_y_pos_g_neg),0).cpu().numpy())
        train_true_group2 = list(torch.cat((y_y_pos_g_pos,y_y_neg_g_neg),0).cpu().numpy())
        
        train_con1_auc = roc_auc_score(train_true_group1, train_pred_group1)
        train_con2_auc = roc_auc_score(train_true_group2, train_pred_group2)
        
        train_auc_list.append(train_auc)
        train_con1_list.append(train_con1_auc)
        train_con2_list.append(train_con2_auc)
        
        # print results
        print("train_auc:{:4f}, train_con1_auc:{:4f}, train_con2_auc:{:4f}".format(train_auc, train_con1_auc, train_con2_auc))
        
        X_val = X_val.cuda()
        val_score = model(X_val).detach()
        val_pred = list(val_score.cpu().numpy())
        val_true = list(y_val.cpu().numpy())
        val_auc = roc_auc_score(val_true, val_pred)
        val_pred_y_pos_g_pos = val_score[val_i_y_pos_g_pos]
        val_pred_y_pos_g_neg = val_score[val_i_y_pos_g_neg]
        val_pred_y_neg_g_pos = val_score[val_i_y_neg_g_pos]
        val_pred_y_neg_g_neg = val_score[val_i_y_neg_g_neg]
        
        val_pred_group1 = list(torch.cat((val_pred_y_neg_g_pos,val_pred_y_pos_g_neg),0).cpu().numpy())
        val_pred_group2 = list(torch.cat((val_pred_y_pos_g_pos,val_pred_y_neg_g_neg),0).cpu().numpy())
        val_true_group1 = list(torch.cat((val_y_y_neg_g_pos, val_y_y_pos_g_neg),0).cpu().numpy())
        val_true_group2 = list(torch.cat((val_y_y_pos_g_pos, val_y_y_neg_g_neg),0).cpu().numpy())
        
        val_con1_auc = roc_auc_score(val_true_group1, val_pred_group1)
        val_con2_auc = roc_auc_score(val_true_group2, val_pred_group2)
        
        val_auc_list.append(val_auc)
        val_con1_list.append(val_con1_auc)
        val_con2_list.append(val_con2_auc)
        
        # print results
        # print("val_auc:{:4f}, val_con1_auc:{:4f}, val_con2_auc:{:4f}".format(val_auc, val_con1_auc, val_con2_auc))
        
        X_test = X_test.cuda()
        test_score = model(X_test).detach()
        test_pred = list(test_score.cpu().numpy())
        test_true = list(y_test.cpu().numpy())
        test_auc = roc_auc_score(test_true, test_pred)
        test_pred_y_pos_g_pos = test_score[test_i_y_pos_g_pos]
        test_pred_y_pos_g_neg = test_score[test_i_y_pos_g_neg]
        test_pred_y_neg_g_pos = test_score[test_i_y_neg_g_pos]
        test_pred_y_neg_g_neg = test_score[test_i_y_neg_g_neg]
        
        test_pred_group1 = list(torch.cat((test_pred_y_neg_g_pos, test_pred_y_pos_g_neg),0).cpu().numpy())
        test_pred_group2 = list(torch.cat((test_pred_y_pos_g_pos, test_pred_y_neg_g_neg),0).cpu().numpy())
        test_true_group1 = list(torch.cat((test_y_y_neg_g_pos, test_y_y_pos_g_neg),0).cpu().numpy())
        test_true_group2 = list(torch.cat((test_y_y_pos_g_pos, test_y_y_neg_g_neg),0).cpu().numpy())
        
        test_con1_auc = roc_auc_score(test_true_group1, test_pred_group1)
        test_con2_auc = roc_auc_score(test_true_group2, test_pred_group2)

        # print results
        print("test_auc:{:4f}, test_con1_auc:{:4f}, test_con2_auc:{:4f}".format(test_auc, test_con1_auc, test_con2_auc))
        
        test_auc_list.append(test_auc)
        test_con1_list.append(test_con1_auc)
        test_con2_list.append(test_con2_auc)
        diff_list.append(abs(test_con1_auc-test_con2_auc))
                
    df1 = pd.DataFrame(train_auc_list, columns=['train_auc'])
    df2 = pd.DataFrame(train_con1_list, columns=['train_con1'])
    df3 = pd.DataFrame(train_con2_list, columns=['train_con2'])
    df4 = pd.DataFrame(val_auc_list, columns=['val_auc'])
    df5 = pd.DataFrame(val_con1_list, columns=['val_con1'])
    df6 = pd.DataFrame(val_con2_list, columns=['val_con2'])
    df7 = pd.DataFrame(test_auc_list, columns=['test_auc'])
    df8 = pd.DataFrame(test_con1_list, columns=['test_con1'])
    df9 = pd.DataFrame(test_con2_list, columns=['test_con2'])
    df10 = pd.DataFrame(diff_list, columns=['diff'])

    d1 = df1.join(df2)
    d2 = d1.join(df3)
    d3 = d2.join(df4)
    d4 = d3.join(df5)
    d5 = d4.join(df6)
    d6 = d5.join(df7)
    d7 = d6.join(df8)
    d8 = d7.join(df9)
    d9 = d8.join(df10)
    
    s = "proxy_con2_fair="+str(para.fair_con)+"_seed="+str(SEED)+"_lr_0="+str(para.lr_0)+"_T="+str(para.T)+"_c="+str(para.c)+".csv"
    d9.to_csv(s)                
                