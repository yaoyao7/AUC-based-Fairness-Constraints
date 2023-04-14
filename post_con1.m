%% import data
load('a9a.mat')
[n,d] = size(A);
% split training data and validation data
seed = 2;
ptrain = 0.9;

% add columns
add = [A(:,1:71),A(:,73:d)];
add = add.*A(:,72);
A = [A,add];

A = [A,ones(n,1)];

rng(seed);
% random permutation without repeating
idx = randperm(n);
A_train = A(idx(1:round(n*ptrain)),:);
A_valid = A(idx(round(n*ptrain)+1):n,:);
b_train = b(idx(1:round(n*ptrain)));
b_valid = b(idx(round(n*ptrain)+1):n);

A = A_train;
b = b_train;
[n,d] = size(A);

% import test data
[label_t, instance_t] = libsvmread('/Users/.../data/a9a/a9a_t');
[n_t,d_t]=size(instance_t);
A_test = instance_t;
A_test = [A_test, zeros(n_t,1)];
b_test = label_t;

add_t = [A_test(:,1:71),A_test(:,73:d_t+1)];
add_t = add_t.*A_test(:,72);
A_test = [A_test,add_t];

A_test = [A_test,ones(n_t,1)];

sv=full(A(:,72)); %%=0 male; =1 female
sv_valid=full(A_valid(:,72));
sv_test=full(A_test(:,72));

%%
% calculate unconstrained score
x=x_best;
s_train=A*x;
s_valid=A_valid*x;
s_test=A_test*x;

s_train_g_pos=s_train(sv==1);
s_valid_g_pos=s_valid(sv_valid==1);
s_test_g_pos=s_test(sv_test==1);

a_list = 0:0.05:5;
b_list = -3:0.1:3;
n_a = length(a_list);
n_b = length(b_list);

test_auc_list_1 = [];
test_con1_list_1 = [];
test_auc_list_2 = [];
test_con1_list_2 = [];
test_auc_list_3 = [];
test_con1_list_3 = [];
test_auc_list_4 = [];
test_con1_list_4 = [];
test_auc_list_5 = [];
test_con1_list_5 = [];

for i = 1:n_a
    aa = a_list(i);
    for j = 1:n_b
        bb = b_list(j);
        trans_s_train_g_pos=aa*s_train_g_pos+bb;
        trans_s_valid_g_pos=aa*s_valid_g_pos+bb;
        trans_s_test_g_pos=aa*s_test_g_pos+bb;
        
        s_train(sv==1)=trans_s_train_g_pos;
        s_valid(sv_valid==1)=trans_s_valid_g_pos;
        s_test(sv_test==1)=trans_s_test_g_pos;
        
        [~,~,~,auc] = perfcurve(b,s_train,1);
        [~,~,~,auc_g1] = perfcurve(sv,s_train,1);
        [~,~,~,auc_valid] = perfcurve(b_valid,s_valid,1);
        [~,~,~,auc_g1_valid] = perfcurve(sv_valid,s_valid,1);
        [~,~,~,auc_test] = perfcurve(b_test,s_test,1);
        [~,~,~,auc_g1_test] = perfcurve(sv_test,s_test,1);
        
        if abs(auc_g1_test-0.5)<0.01
            test_auc_list_1=[test_auc_list_1;auc_test];
            test_con1_list_1=[test_con1_list_1;auc_g1_test];
        end
        if abs(auc_g1_test-0.45)<0.01
            test_auc_list_2=[test_auc_list_2;auc_test];
            test_con1_list_2=[test_con1_list_2;auc_g1_test];
        end
        if abs(auc_g1_test-0.4)<0.01
            test_auc_list_3=[test_auc_list_3;auc_test];
            test_con1_list_3=[test_con1_list_3;auc_g1_test];
        end
        if abs(auc_g1_test-0.35)<0.01
            test_auc_list_4=[test_auc_list_4;auc_test];
            test_con1_list_4=[test_con1_list_4;auc_g1_test];
        end
        if abs(auc_g1_test-0.3)<0.01
            test_auc_list_5=[test_auc_list_5;auc_test];
            test_con1_list_5=[test_con1_list_5;auc_g1_test];
        end
    end
end
