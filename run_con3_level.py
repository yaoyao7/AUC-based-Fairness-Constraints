import torch
import math
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from loss_minmax import loss_con3
from loss_minmax import single_loss, single_cons
from torch.utils.data import TensorDataset, DataLoader
from parameters import para
from model import NeuralNetwork
from commonutil import svm_read_problem
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

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

def rho(tau, alp_old, alp, z):
    # a = p*(1-p) + 1/tau
    # b = -2*alp/tau - z
    # c = alp**2 / tau
    # rho = -b**2/(4*a) + c
    
    # rho = (-4*alp**2-4*alp*z*tau-(z*tau)**2)/(4*tau*(p*(1-p)+1)) + alp/tau
    rho = aa*alp**2-z*alp+1/tau*(alp-alp_old)**2

    return rho

def cal_alpha_til(tau, alp, z):
    # a = p*(1-p) + 1/tau
    # b = -2*alp/tau - z
    # c = alp**2 / tau
    # alp_til = -b/(2*a)
    
    alp_til = (2/tau*alp+z) / (2*aa+2/tau)
    return alp_til

# all paramaters
dataname = 'a9a'
SEED = para.seed
BATCH_SIZE = para.batch
pos = 71   # sensitive position
mu = para.mu
eps = para.eps
T = para.T
outer_iter = para.outer_iter
num_iter = para.num_iter
lr_0 = para.lr_0
tau_0 = para.tau_0
c = para.c
fair_con = para.fair_con
B = para.B
m = 2  # number of constraint

aa = 0.5
cc = 1
theta = 1

c = 1 + eps

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

# split training and validation set
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=2)

set_all_seeds(SEED)

X_train, y_train = shuffle(X_train, y_train, random_state=SEED)

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

# model
set_all_seeds(SEED)
hidden_dim = X_train.shape[1]
output_dim = 1
model = NeuralNetwork(X_train.shape[1], hidden_dim, output_dim).cuda()

x = []
for var in list(model.parameters()):
    x.append(torch.zeros(var.shape, dtype=torch.float32, device=device, requires_grad=False).to(device))
    
for k, w in enumerate(list(model.parameters())):
    x[k].data = w.data

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
    
    Loss = loss_con3()
        
    sum_w = []
    avg_w = []
    w_old = []
    for var in list(model.parameters()):
        sum_w.append(torch.zeros(var.shape, dtype=torch.float32, device=device, requires_grad=False).to(device))
        avg_w.append(torch.zeros(var.shape, dtype=torch.float32, device=device, requires_grad=False).to(device))
        w_old.append(torch.zeros(var.shape, dtype=torch.float32, device=device, requires_grad=False).to(device))
    
    
    # initialize lambda and alpha_bar
    lamb = torch.tensor([1/3, 1/3, 1/3], dtype=torch.float32, device=device, requires_grad=False)
    alpha = torch.zeros(5, dtype=torch.float32, device=device, requires_grad=False)

    r0 = 1
    
    print("r0 = ", r0)
    
    c_trainloader_y_pos = iter(trainloader_y_pos)
    c_trainloader_y_neg = iter(trainloader_y_neg)
    c_trainloader_y_pos_g_pos = iter(trainloader_y_pos_g_pos)
    c_trainloader_y_pos_g_neg = iter(trainloader_y_pos_g_neg)
    c_trainloader_y_neg_g_pos = iter(trainloader_y_neg_g_pos)
    c_trainloader_y_neg_g_neg = iter(trainloader_y_neg_g_neg)
    
    for t in range(T):
        
        # level set algorithm
        for j in range(outer_iter):
            # smd: stochastic mirror descent
            sum_w = []
            avg_w = []
            sum_lr = 0.
            for var in list(model.parameters()):
                sum_w.append(torch.zeros(var.shape, dtype=torch.float32, device=device, requires_grad=False).to(device))
                avg_w.append(torch.zeros(var.shape, dtype=torch.float32, device=device, requires_grad=False).to(device))
                
            for i in range(num_iter):
                # update learning rate
                lr = lr_0 / math.sqrt(i+1)
                tau = tau_0 / math.sqrt(i+1)
                
                sum_grad_lamb = torch.zeros((3,1), dtype=torch.float32, device=device, requires_grad=False).to(device)
                sum_grad_alpha_bar = torch.zeros((5,1), dtype=torch.float32, device=device, requires_grad=False).to(device)
                u = torch.zeros((3,1), dtype=torch.float32, device=device, requires_grad=False).to(device)
                
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
                
                loss = Loss(logits_y_pos, logits_y_neg, logits_y_pos_g_pos, logits_y_pos_g_neg,logits_y_neg_g_pos,logits_y_neg_g_neg, lamb, alpha, aa, cc)
                
                for k, w in enumerate(list(model.parameters())):
                    w_old[k].data = w.data
                
                paras = list(model.parameters())+[Loss.a, Loss.b]
                
                # set zero grad, otherwise grad will accumulate
                for pp in paras:
                    if pp.requires_grad == True:
                        pp.grad = torch.zeros_like(pp.data)
                
                # compute gradients of model parameters
                loss.backward()
                
                for pp in paras:
                    if pp.requires_grad == True:
                        pp.data -= lr * pp.grad
                        
                for k, w in enumerate(list(model.parameters())):
                    w.data -= lr*mu*(w_old[k].data-x[k].data)
                
                # update dual variable
                # compute gradient w.r.t dual vars
                # compute norm of y-x_t
                norm = torch.zeros(1, dtype=torch.float32, device=device, requires_grad=False)
                for k, w in enumerate(list(model.parameters())):
                    norm.data += torch.linalg.norm(w_old[k]-x[k].data)
                
                z_lam_0 = single_loss(aa, cc, logits_y_pos, logits_y_neg, Loss.a.data[0], Loss.b.data[0], alpha[0]) + mu*norm.data/2
                z_lam_0 -= r0
                
                z_lam_1 = single_cons(aa, cc, logits_y_pos_g_pos, logits_y_neg_g_pos, Loss.a.data[1], Loss.b.data[1], alpha[1]) + \
                          single_cons(aa, cc, logits_y_neg_g_neg, logits_y_pos_g_neg, Loss.a.data[2], Loss.b.data[2], alpha[2]) + \
                          mu*norm.data/2
                z_lam_1 -= c
                
                z_lam_2 = single_cons(aa, cc, logits_y_pos_g_neg, logits_y_neg_g_neg, Loss.a.data[3], Loss.b.data[3], alpha[3]) + \
                          single_cons(aa, cc, logits_y_neg_g_pos, logits_y_pos_g_pos, Loss.a.data[4], Loss.b.data[4], alpha[4]) + \
                          mu*norm.data/2
                z_lam_2 -= c
                
                z_alp_0 = 2*aa*(torch.mean(logits_y_neg)-torch.mean(logits_y_pos))
                z_alp_1 = 2*aa*(torch.mean(logits_y_pos_g_pos)-torch.mean(logits_y_neg_g_pos))
                z_alp_2 = 2*aa*(torch.mean(logits_y_neg_g_neg)-torch.mean(logits_y_pos_g_neg))
                z_alp_3 = 2*aa*(torch.mean(logits_y_pos_g_neg)-torch.mean(logits_y_neg_g_neg))
                z_alp_4 = 2*aa*(torch.mean(logits_y_neg_g_pos)-torch.mean(logits_y_pos_g_pos))
                
                # clone alpha
                alpha_old = torch.clone(alpha).detach()
                
                # compute alpha/alpha_tilde
                alpha.data[0] = cal_alpha_til(tau, alpha.data[0], z_alp_0)
                alpha.data[1] = cal_alpha_til(tau, alpha.data[1], z_alp_1)
                alpha.data[2] = cal_alpha_til(tau, alpha.data[2], z_alp_2)
                alpha.data[3] = cal_alpha_til(tau, alpha.data[3], z_alp_3)
                alpha.data[4] = cal_alpha_til(tau, alpha.data[4], z_alp_4)
                
                # compute rho
                rho0 = rho(tau, alpha_old[0], alpha.data[0], z_alp_0)
                rho1 = rho(tau, alpha_old[1], alpha.data[1], z_alp_1)
                rho2 = rho(tau, alpha_old[2], alpha.data[2], z_alp_2)
                rho3 = rho(tau, alpha_old[3], alpha.data[3], z_alp_3)
                rho4 = rho(tau, alpha_old[4], alpha.data[4], z_alp_4)
                
                # l1 = r0 + rho1 - z_lam_1
                l0 = rho0 - z_lam_0
                l1 = rho1+rho2 - z_lam_1
                l2 = rho3+rho4 - z_lam_2
                
                l = torch.stack([l0, l1, l2])
                l = l.reshape(3)
                lamb.data *= torch.exp(-l/(2*(1+B)**2/tau))
                sum_lamb = torch.sum(lamb)
                lamb.data /= sum_lamb
                
                # print(lamb)
                
                for k, w in enumerate(list(model.parameters())):
                   sum_w[k].data = sum_w[k].data + lr * w.data
                
                sum_lr += lr
                
                sum_grad_lamb[0].data += lr * z_lam_0.data
                sum_grad_lamb[1].data += lr * z_lam_1.data
                sum_grad_lamb[2].data += lr * z_lam_2.data
                
                
                sum_grad_alpha_bar[0].data += lr * (z_alp_0.data)
                sum_grad_alpha_bar[1].data += lr * (z_alp_1.data)
                sum_grad_alpha_bar[2].data += lr * (z_alp_2.data)
                sum_grad_alpha_bar[3].data += lr * (z_alp_3.data)
                sum_grad_alpha_bar[4].data += lr * (z_alp_4.data)
                
                
                model.eval()
                
                X_test = X_test.cuda()
                test_score = model(X_test).detach()
                test_pred = list(test_score.cpu().numpy())
                test_true = list(y_test.cpu().numpy())
                test_auc = roc_auc_score(test_true, test_pred)
                test_pred_y_pos_g_pos = test_score[test_i_y_pos_g_pos]
                test_pred_y_pos_g_neg = test_score[test_i_y_pos_g_neg]
                test_pred_y_neg_g_pos = test_score[test_i_y_neg_g_pos]
                test_pred_y_neg_g_neg = test_score[test_i_y_neg_g_neg]
                
                test_pred_group1 = list(torch.cat((test_pred_y_neg_g_pos, test_pred_y_pos_g_pos),0).cpu().numpy())
                test_pred_group2 = list(torch.cat((test_pred_y_pos_g_neg, test_pred_y_neg_g_neg),0).cpu().numpy())
                test_true_group1 = list(torch.cat((test_y_y_neg_g_pos, test_y_y_pos_g_pos),0).cpu().numpy())
                test_true_group2 = list(torch.cat((test_y_y_pos_g_neg, test_y_y_neg_g_neg),0).cpu().numpy())
                
                test_con1_auc = roc_auc_score(test_true_group1, test_pred_group1)
                test_con2_auc = roc_auc_score(test_true_group2, test_pred_group2)
                
                if abs(abs(test_con1_auc-test_con2_auc) - fair_con) < 0.01:
                    test_auc_list.append(test_auc)
                    test_con1_list.append(test_con1_auc)
                    test_con2_list.append(test_con2_auc)
                    diff_list.append(0)
                   
                    print("test_auc:{:4f}, test_con1_auc:{:4f}, test_con2_auc:{:4f}".format(test_auc, test_con1_auc, test_con2_auc))
            
            print("test_auc:{:4f}, test_con1_auc:{:4f}, test_con2_auc:{:4f}".format(test_auc, test_con1_auc, test_con2_auc))
            
            for k, param in enumerate(sum_w):
               avg_w[k].data = sum_w[k].data / sum_lr
            
            u[0] = ((sum_grad_alpha_bar[0]/sum_lr)**2)/4/aa + sum_grad_lamb[0]/sum_lr
            u[1] = ((sum_grad_alpha_bar[1]/sum_lr)**2)/4/aa + ((sum_grad_alpha_bar[2]/sum_lr)**2)/4/aa + sum_grad_lamb[1]/sum_lr
            u[2] = ((sum_grad_alpha_bar[3]/sum_lr)**2)/4/aa + ((sum_grad_alpha_bar[4]/sum_lr)**2)/4/aa + sum_grad_lamb[2]/sum_lr
            U = torch.max(u)
            
            
            print("upper bound = ", U)
            
            print ('-'*60)
            
            r0 += U / theta
        
        for k, w in enumerate(list(model.parameters())):
           x[k].data = w.data
        
                
    # df1 = pd.DataFrame(train_auc_list, columns=['train_auc'])
    # df2 = pd.DataFrame(train_con1_list, columns=['train_con1'])
    # df3 = pd.DataFrame(train_con2_list, columns=['train_con2'])
    # df4 = pd.DataFrame(val_auc_list, columns=['val_auc'])
    # df5 = pd.DataFrame(val_con1_list, columns=['val_con1'])
    # df6 = pd.DataFrame(val_con2_list, columns=['val_con2'])
    df7 = pd.DataFrame(test_auc_list, columns=['test_auc'])
    df8 = pd.DataFrame(test_con1_list, columns=['test_con1'])
    df9 = pd.DataFrame(test_con2_list, columns=['test_con2'])
    df10 = pd.DataFrame(diff_list, columns=['diff'])

    # d1 = df1.join(df2)
    # d2 = d1.join(df3)
    # d3 = d2.join(df4)
    # d4 = d3.join(df5)
    # d5 = d4.join(df6)
    # d6 = d5.join(df7)
    # d7 = d6.join(df8)
    # d8 = d7.join(df9)
    
    d1 = df7.join(df8)
    d2 = d1.join(df9)
    d3 = d2.join(df10)
    
    s = "level_con3"+"_eps="+str(para.eps)+"_seed="+str(para.seed)+"_mu="+str(para.mu)+"_T="+str(para.T)+"_outer="+str(para.outer_iter)+"_inner="+str(para.num_iter)+".csv"
    d3.to_csv(s)   
    
    