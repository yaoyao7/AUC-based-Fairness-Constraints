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
load('a9a_t.mat')
[n_t,d_t]=size(instance_t);
A_test = instance_t;
A_test = [A_test, zeros(n_t,1)];
b_test = label_t;

add_t = [A_test(:,1:71),A_test(:,73:d_t+1)];
add_t = add_t.*A_test(:,72);
A_test = [A_test,add_t];

A_test = [A_test,ones(n_t,1)];

numclass=length(unique(b));
kappa = 1.5;
fair_con = 0.05;
delta = 0.1;
m=2;    %Number of constraints
optr=0.0000;
varepsilon=0.001;
theta=1;
a=0.5;
c=1;
constraint_bound = ones(2,1)+kappa;
regularization_term = 10000;
D=regularization_term;
datanorm=sqrt(mean(vecnorm(A,2,2).^2));
M=sqrt(2*regularization_term^2/2*datanorm^2+2*(1/2-1/2/numclass)*datanorm^2*regularization_term^2*numclass); 
Q=datanorm*regularization_term;
eta0=0.1;
tau0=0.1;
r0=0.5;
B=1;
tic

% Define ouput variable
train_auc_list = [];
train_con1_list = [];
train_con2_list = [];
valid_auc_list = [];
valid_con1_list = [];
valid_con2_list = [];
test_auc_list = [];
test_con1_list = [];
test_con2_list = [];

output_objectivevalue = [];
output_constraintvalue = [];
output_infeasibility =[];
output_iterall =[];
output_iterouter =[];
output_residualouter=[];
output_objectivevalue=[];
output_objectivevalueouter=[];
output_infeasibilityouter=[];
output_L = [];
output_U = [];
output_r = [];
output_time = [];
optimality_residual = [];
data_pass = [];
data_pass_outer=[];
total_datapass=0;
total_iteration=0;
batchsize=100;

sv=full(A(:,72)); %%=0 male; =1 female
sv_valid=full(A_valid(:,72));
sv_test=full(A_test(:,72));
idmale=find(sv==0);
idfemale=find(sv==1);
numpos=sum(b==1);
numneg=sum(b==-1);
numposmale=sum(b==1 & sv==0);
numnegmale=sum(b==-1 & sv==0);
numposfemale=sum(b==1 & sv==1);
numnegfemale=sum(b==-1 & sv==1);
x=zeros(d,numclass-1);
y = 1/(m+1)*ones(m+1,1);
aaobj = 0;
bbobj = 0;
aacst = zeros(m,2);
bbcst = zeros(m,2);
alphaobj = 0;
alphaobjy = alphaobj/y(1);
alphacst = zeros(m,2);
alphacsty =  diag(1./y(2:m+1))*alphacst;

for iter=1:60
    r=[r0;constraint_bound];
    Ax=A*x;
    temp_optimalvalue = a*(c^2+sum(Ax(b==1).^2)/numpos+sum(Ax(b==-1).^2)/numneg-2*c*sum(Ax(b==1))/numpos+2*c*sum(Ax(b==-1))/numneg-2*sum(Ax(b==1))*sum(Ax(b==-1))/numpos/numneg);
    temp_constraintvalue = [a*(c^2+sum(Ax(b==1 & sv==0).^2)/numposmale+sum(Ax(b==-1 & sv==0).^2)/numnegmale+2*c*sum(Ax(b==1 & sv==0))/numposmale-2*c*sum(Ax(b==-1 & sv==0))/numnegmale-2*sum(Ax(b==1 & sv==0))*sum(Ax(b==-1 & sv==0))/numposmale/numnegmale)+a*(c^2+sum(Ax(b==1 & sv==1).^2)/numposfemale+sum(Ax(b==-1 & sv==1).^2)/numnegfemale+2*c*sum(Ax(b==-1 & sv==1))/numnegfemale-2*c*sum(Ax(b==1 & sv==1))/numposfemale-2*sum(Ax(b==1 & sv==1))*sum(Ax(b==-1 & sv==1))/numposfemale/numnegfemale);
                            a*(c^2+sum(Ax(b==1 & sv==1).^2)/numposfemale+sum(Ax(b==-1 & sv==1).^2)/numnegfemale-2*c*sum(Ax(b==-1 & sv==1))/numnegfemale+2*c*sum(Ax(b==1 & sv==1))/numposfemale-2*sum(Ax(b==1 & sv==1))*sum(Ax(b==-1 & sv==1))/numposfemale/numnegfemale)+a*(c^2+sum(Ax(b==1 & sv==0).^2)/numposmale+sum(Ax(b==-1 & sv==0).^2)/numnegmale-2*c*sum(Ax(b==1 & sv==0))/numposmale+2*c*sum(Ax(b==-1 & sv==0))/numnegmale-2*sum(Ax(b==1 & sv==0))*sum(Ax(b==-1 & sv==0))/numposmale/numnegmale);
                            ];
    temp_infeasibility = max(temp_constraintvalue- constraint_bound);
    temp_residual=max(temp_optimalvalue - optr,temp_infeasibility);
    output_iterouter =[output_iterouter; total_iteration];
    output_residualouter=[output_residualouter; temp_residual];
    output_infeasibilityouter=[output_infeasibilityouter;temp_infeasibility ];
    output_objectivevalueouter=[output_objectivevalueouter;temp_optimalvalue ];
    data_pass_outer=[data_pass_outer;total_datapass];
    output_r = [output_r; r0];
    deltak=delta/2^iter;
    SSGD_DDC_a9a;
    
    output_L = [output_L; temp_L];
    output_U = [output_U; temp_U];    
    if iter==1
        U0=temp_U;
    end
    r0=r0+temp_U/theta;
end

result = [test_auc_list, test_con1_list, test_con2_list];
save('a9a_con3_0.05_seed2.mat','x','train_auc_list','train_con1_list','train_con2_list','valid_auc_list','valid_con1_list','valid_con2_list','test_auc_list','test_con1_list','test_con2_list')
toc

