%% import data
load('a9a.mat')
[n,d] = size(A);
% split training data and validation data
seed = 13;
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
[label_t, instance_t] = libsvmread('...');
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
b_test_g_pos_y_pos = b_test(sv_test==1 & b_test==1);
b_test_g_neg_y_neg = b_test(sv_test==0 & b_test==-1);
b_test_g_neg_y_pos = b_test(sv_test==0 & b_test==1);
b_test_g_pos_y_neg = b_test(sv_test==1 & b_test==-1);

b_test_g1 = [b_test(sv_test==1 & b_test==1);b_test(sv_test==0 & b_test==-1)];
b_test_g2 = [b_test(sv_test==0 & b_test==1);b_test(sv_test==1 & b_test==-1)];

s_test_g_pos_y_pos = score(sv_test==1 & b_test==1);
s_test_g_neg_y_neg = score(sv_test==0 & b_test==-1);
s_test_g_neg_y_pos = score(sv_test==0 & b_test==1);
s_test_g_pos_y_neg = score(sv_test==1 & b_test==-1);

s_test = score;
[~,~,~,auc_test_initial] = perfcurve(b_test,s_test,1);
[~,~,~,auc_g1_test_init] = perfcurve(b_test_g1,[s_test_g_pos_y_pos;s_test_g_neg_y_neg;],1);
[~,~,~,auc_g2_test_init] = perfcurve(b_test_g2,[s_test_g_neg_y_pos;s_test_g_pos_y_neg;],1);


a_list = 0:0.05:2;
b_list = -2:0.1:2;
n_a = length(a_list);
n_b = length(b_list);

test_auc_list_1 = [];
test_con1_list_1 = [];
test_con2_list_1 = [];
test_auc_list_2 = [];
test_con1_list_2 = [];
test_con2_list_2 = [];
test_auc_list_3 = [];
test_con1_list_3 = [];
test_con2_list_3 = [];

for i = 1:n_a
    aa = a_list(i);
    for j = 1:n_b
        bb = b_list(j);
%         trans_s_test_g_pos_y_pos = aa*s_test_g_pos_y_pos+bb;
%         trans_s_test_g_pos_y_neg = aa*s_test_g_pos_y_neg+bb;
%         
        trans_s_test_g_neg_y_neg = aa*s_test_g_neg_y_neg+bb;
        trans_s_test_g_neg_y_pos = aa*s_test_g_neg_y_pos+bb;
        
        b_test=[b_test_g_pos_y_pos;
                b_test_g_neg_y_neg;
                b_test_g_neg_y_pos;
                b_test_g_pos_y_neg;
               ];
%         s_test=[trans_s_test_g_pos_y_pos;
%                 s_test_g_neg_y_neg;
%                 s_test_g_neg_y_pos;
%                 trans_s_test_g_pos_y_neg;
%                ];
        s_test=[s_test_g_pos_y_pos;
                trans_s_test_g_neg_y_neg;
                trans_s_test_g_neg_y_pos;
                s_test_g_pos_y_neg;
               ];
        
        [~,~,~,auc_test] = perfcurve(b_test,s_test,1);
%         [~,~,~,auc_g1_test] = perfcurve(b_test_g1,[trans_s_test_g_pos_y_pos;s_test_g_neg_y_neg;],1);
%         [~,~,~,auc_g2_test] = perfcurve(b_test_g2,[s_test_g_neg_y_pos;trans_s_test_g_pos_y_neg;],1);
%         
        [~,~,~,auc_g1_test] = perfcurve(b_test_g1,[s_test_g_pos_y_pos;trans_s_test_g_neg_y_neg;],1);
        [~,~,~,auc_g2_test] = perfcurve(b_test_g2,[trans_s_test_g_neg_y_pos;s_test_g_pos_y_neg;],1);
        
        if abs(abs(auc_g1_test-auc_g2_test)-0)<0.01
            test_auc_list_1=[test_auc_list_1;auc_test];
            test_con1_list_1=[test_con1_list_1;auc_g1_test];
            test_con2_list_1=[test_con2_list_1;auc_g2_test];
        end
        if abs(abs(auc_g1_test-auc_g2_test)-0.05)<0.01
            test_auc_list_2=[test_auc_list_2;auc_test];
            test_con1_list_2=[test_con1_list_2;auc_g1_test];
            test_con2_list_2=[test_con2_list_2;auc_g2_test];
        end
        if abs(abs(auc_g1_test-auc_g2_test)-0.1)<0.01
            test_auc_list_3=[test_auc_list_3;auc_test];
            test_con1_list_3=[test_con1_list_3;auc_g1_test];
            test_con2_list_3=[test_con2_list_3;auc_g2_test];
        end

    end
end