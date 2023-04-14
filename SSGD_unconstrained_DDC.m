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
[label_t, instance_t] = libsvmread('...');
[n_t,d_t]=size(instance_t);
A_test = instance_t;
A_test = [A_test, zeros(n_t,1)];
b_test = label_t;

add_t = [A_test(:,1:71),A_test(:,73:d_t+1)];
add_t = add_t.*A_test(:,72);
A_test = [A_test,add_t];

A_test = [A_test,ones(n_t,1)];

numclass=length(unique(b));
kappa = 0.1;
delta=0.1;
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
r0=1;
B=1;
tic

% Define ouput variable
test_auc_list = [];
AUC_best = 0;

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
batchsize=500;

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
aaobj = 0;
bbobj = 0;
alphaobj = 0;

numiterSGD=6000;

SGDobj=[];
xbar=x;
aaobjbar=aaobj;
bbobjbar=bbobj;
alphaobjbar = alphaobj;

bxsum=x*0;
axsum=0;
etasum=0;
temp_U=+Inf;
temp_L=-Inf;
errortemp=+Inf;
Inner_L=[];
Inner_U=[];
gradientcst=zeros(d,2);
for k = 1:numiterSGD
    eta=eta0/sqrt(k);
    tau=tau0/sqrt(k);
    if mod(k,100)==1
        Ax=A*xbar;
        temp_optimalvalue = a*(c^2+sum(Ax(b==1).^2)/numpos+sum(Ax(b==-1).^2)/numneg-2*c*sum(Ax(b==1))/numpos+2*c*sum(Ax(b==-1))/numneg-2*sum(Ax(b==1))*sum(Ax(b==-1))/numpos/numneg);
        temp_constraintvalue = [0;0];                    
        temp_infeasibility = max(temp_constraintvalue- constraint_bound);
    end
    total_datapass=total_datapass+2*batchsize/n;
    total_iteration=total_iteration+1;
    output_objectivevalue = [output_objectivevalue; temp_optimalvalue];
    output_infeasibility = [output_infeasibility; temp_infeasibility];
    output_constraintvalue = [output_constraintvalue; temp_constraintvalue];
    optimality_residual = [optimality_residual; max(temp_optimalvalue - optr,temp_infeasibility)];
    data_pass = [data_pass; total_datapass];
    output_iterall = [output_iterall; total_iteration];
    output_time = [output_time,toc];
    tempSGDobj=max(temp_optimalvalue - r0,temp_infeasibility);
    SGDobj=[SGDobj tempSGDobj];
    fprintf('Iter=%f,k=%f,r0=%f,U=%f,L=%f,obj=%f,infsbl=%f,datapass=%f\n',1,k,r0,temp_U,temp_L,temp_optimalvalue,temp_infeasibility,total_datapass);
    temp_id = datasample(1:n,batchsize);
    tempAx=A(temp_id,:)*x;
    tempb=b(temp_id);
    tempsv=sv(temp_id);
    
    fvalalpha = 2*a*mymean(tempAx(tempb==-1)) - 2*a*mymean(tempAx(tempb==1));
    
    gradientobj=zeros(d,1);
    gradientaaobj=0;
    gradientbbobj=0;
    if sum(tempb==1)>0
        gradientobj=gradientobj+2*a*A(temp_id(tempb==1),:)'*(tempAx(tempb==1)-aaobj)/sum(tempb==1)-2*a*(c+alphaobj)*mean(A(temp_id(tempb==1),:))';
        gradientaaobj=gradientaaobj+2*a*mean(aaobj-tempAx(tempb==1));
    end
    if sum(tempb==-1)>0
        gradientobj=gradientobj+2*a*A(temp_id(tempb==-1),:)'*(tempAx(tempb==-1)-bbobj)/sum(tempb==-1)+2*a*(c+alphaobj)*mean(A(temp_id(tempb==-1),:))';
        gradientbbobj=gradientbbobj+2*a*mean(bbobj-tempAx(tempb==-1));
    end    
    
    gradient=gradientobj;
    
    errortemp=0;
    etasum=etasum+eta;
    bxsum=bxsum+eta*gradient;
    bysum=bysum+eta*grady;
    axsum=axsum+eta*(grady'*y)-eta*(gradient'*reshape(x,d*numclass,1));
    aysum=aysum+eta*(grady'*y)-eta*(grady'*y);
    temp_bxsum=reshape(bxsum,d,numclass);
    temp=sqrt(diag(temp_bxsum'*temp_bxsum));
    tempid=find(temp~=0);
    temp(tempid)=1./temp(tempid);
    tempx=-temp_bxsum*diag(temp)*regularization_term;
    temp_L=axsum/etasum+trace(tempx'*temp_bxsum)/etasum-errortemp;
    [temp_ymax,temp_idmax]=max(bysum);
    temp_idmax=temp_idmax(1);
    tempy=zeros(m+1,1);
    tempy(temp_idmax)=1;
    temp_U=aysum/etasum+tempy'*bysum/etasum+errortemp;
    temp_U=max([temp_optimalvalue-r0;temp_infeasibility]);
    Inner_L=[Inner_L;temp_L];
    Inner_U=[Inner_U;temp_U];
    
    [~,~,~,AUC_test] = perfcurve(b_test,A_test*x,1);
    test_auc_list=[test_auc_list;AUC_test];
    
    if AUC_test > AUC_best
        AUC_best = AUC_test;
        x_best = x;
    end
    
    xold=x;
    x=x-eta*gradient;
    if norm(x)>regularization_term
            x=x/norm(x)*regularization_term;
    end
    aaobjold=aaobj;
    aaobj=aaobj-eta*gradientaaobj;
    bbobjold=bbobj;
    bbobj=bbobj-eta*gradientbbobj;
    
    alphaobjold=alphaobj;
      
    alphaobj = (fvalalpha+2/tau*alphaobj)/(2*a+2/tau);
    
    xbar=(xbar*(k-1)+x)/k;
    aaobjbar=(aaobjbar*(k-1)+aaobj)/k;
    bbobjbar=(bbobjbar*(k-1)+bbobj)/k;
    alphaobjbar=(alphaobjbar*(k-1)+alphaobj)/k;
    
end
x=xbar;
aaobj=aaobjbar;
bbobj=bbobjbar;
alphaobj=alphaobjbar;
save('x_best.mat','x_best')
