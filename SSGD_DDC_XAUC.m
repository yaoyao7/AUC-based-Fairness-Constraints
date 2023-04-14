% numiterSGD=300;
numiterSGD=600;

SGDobj=[];
ybar=y;
xbar=x;
aaobjbar=aaobj;
bbobjbar=bbobj;
aacstbar=aacst;
bbcstbar=bbcst;
alphaobjbar = alphaobj;
alphacstbar = alphacst;

bxsum=x*0;
bysum=y*0;
balphaobjsum=alphaobj*0;
balphacstsum=alphacst*0;
axsum=0;
aysum=0;
aalphaobjsum=0;
aalphacstsum=0;
etasum=0;
temp_U=+Inf;
temp_L=-Inf;
errortemp=+Inf;
Inner_L=[];
Inner_U=[];
gradientcst=zeros(d,2);
rng(seed);
for k = 1:numiterSGD
    eta=eta0/sqrt(k);
    tau=tau0/sqrt(k);
    if mod(k,700)==1
        Ax=A*xbar;
        temp_optimalvalue = a*(c^2+sum(Ax(b==1).^2)/numpos+sum(Ax(b==-1).^2)/numneg-2*c*sum(Ax(b==1))/numpos+2*c*sum(Ax(b==-1))/numneg-2*sum(Ax(b==1))*sum(Ax(b==-1))/numpos/numneg);
        temp_constraintvalue = [a*(c^2+sum(Ax(b==1 & sv==1).^2)/numposfemale+sum(Ax(b==-1 & sv==0).^2)/numnegmale+2*c*sum(Ax(b==1 & sv==1))/numposfemale-2*c*sum(Ax(b==-1 & sv==0))/numnegmale-2*sum(Ax(b==1 & sv==1))*sum(Ax(b==-1 & sv==0))/numposfemale/numnegmale)+a*(c^2+sum(Ax(b==1 & sv==0).^2)/numposmale+sum(Ax(b==-1 & sv==1).^2)/numnegfemale+2*c*sum(Ax(b==-1 & sv==1))/numnegfemale-2*c*sum(Ax(b==1 & sv==0))/numposmale-2*sum(Ax(b==1 & sv==0))*sum(Ax(b==-1 & sv==1))/numposmale/numnegfemale);
                                a*(c^2+sum(Ax(b==1 & sv==0).^2)/numposmale+sum(Ax(b==-1 & sv==1).^2)/numnegfemale-2*c*sum(Ax(b==-1 & sv==1))/numnegfemale+2*c*sum(Ax(b==1 & sv==0))/numposmale-2*sum(Ax(b==1 & sv==0))*sum(Ax(b==-1 & sv==1))/numposmale/numnegfemale)+a*(c^2+sum(Ax(b==1 & sv==1).^2)/numposfemale+sum(Ax(b==-1 & sv==0).^2)/numnegmale-2*c*sum(Ax(b==1 & sv==1))/numposfemale+2*c*sum(Ax(b==-1 & sv==0))/numnegmale-2*sum(Ax(b==1 & sv==1))*sum(Ax(b==-1 & sv==0))/numposfemale/numnegmale);
                                ];
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
    fprintf('Iter=%f,k=%f,r0=%f,U=%f,L=%f,obj=%f,infsbl=%f,datapass=%f\n',iter,k,r0,temp_U,temp_L,temp_optimalvalue,temp_infeasibility,total_datapass);
    temp_id = datasample(1:n,batchsize);
    tempAx=A(temp_id,:)*x;
    tempb=b(temp_id);
    tempsv=sv(temp_id);
    
    fval = a*c^2+a*mymean((tempAx(tempb==1)-aaobj).^2)+a*mymean((tempAx(tempb==-1)-bbobj).^2);
    fval = fval-2*a*c*mymean(tempAx(tempb==1))+2*a*c*mymean(tempAx(tempb==-1)) -r0;
    
    gval = zeros(m,1);
    gval(1) = a*c^2+a*mymean((tempAx(tempb==1 & tempsv==1)-aacst(1,1)).^2)+a*mymean((tempAx(tempb==-1 & tempsv==0)-bbcst(1,1)).^2);
    gval(1) = gval(1)-2*a*c*mymean(tempAx(tempb==-1 & tempsv==0))+2*a*c*mymean(tempAx(tempb==1 & tempsv==1));
    gval(1) = gval(1)+a*c^2+a*mymean((tempAx(tempb==1 & tempsv==0)-aacst(1,2)).^2)+a*mymean((tempAx(tempb==-1 & tempsv==1)-bbcst(1,2)).^2);
    gval(1) = gval(1)-2*a*c*mymean(tempAx(tempb==1 & tempsv==0))+2*a*c*mymean(tempAx(tempb==-1 & tempsv==1));
    gval(2) = a*c^2+a*mymean((tempAx(tempb==1 & tempsv==0)-aacst(2,1)).^2)+a*mymean((tempAx(tempb==-1 & tempsv==1)-bbcst(2,1)).^2);
    gval(2) = gval(2)-2*a*c*mymean(tempAx(tempb==-1 & tempsv==1))+2*a*c*mymean(tempAx(tempb==1 & tempsv==1));
    gval(2) = gval(2)+a*c^2+a*mymean((tempAx(tempb==1 & tempsv==1)-aacst(2,2)).^2)+a*mymean((tempAx(tempb==-1 & tempsv==0)-bbcst(2,2)).^2);
    gval(2) = gval(2)-2*a*c*mymean(tempAx(tempb==1 & tempsv==1))+2*a*c*mymean(tempAx(tempb==-1 & tempsv==0));
    
    temp_constraintvalue = gval;
    gval = temp_constraintvalue- constraint_bound;
    grady = [fval;gval]; 
    
    fvalalpha = 2*a*mymean(tempAx(tempb==-1)) - 2*a*mymean(tempAx(tempb==1));
    gvalalpha = zeros(m,2);
    gvalalpha(1,1) = 2*a*mymean(tempAx(tempb==1 & tempsv==1)) - 2*a*mymean(tempAx(tempb==-1 & tempsv==0));
    gvalalpha(1,2) = 2*a*mymean(tempAx(tempb==-1 & tempsv==1)) - 2*a*mymean(tempAx(tempb==1 & tempsv==0));
    gvalalpha(2,1) = 2*a*mymean(tempAx(tempb==1 & tempsv==0)) - 2*a*mymean(tempAx(tempb==-1 & tempsv==1));
    gvalalpha(2,2) = 2*a*mymean(tempAx(tempb==-1 & tempsv==0)) - 2*a*mymean(tempAx(tempb==1 & tempsv==1));
    
    gradientobj=zeros(d,1); %x gradient in obj
    gradientaaobj=0;
    gradientbbobj=0;
    if sum(tempb==1)>0
        gradientobj=gradientobj+y(1)*2*a*A(temp_id(tempb==1),:)'*(tempAx(tempb==1)-aaobj)/sum(tempb==1)-2*a*(y(1)*c+alphaobj)*mean(A(temp_id(tempb==1),:))';
        gradientaaobj=gradientaaobj+y(1)*2*a*mean(aaobj-tempAx(tempb==1));
    end
    if sum(tempb==-1)>0
        gradientobj=gradientobj+y(1)*2*a*A(temp_id(tempb==-1),:)'*(tempAx(tempb==-1)-bbobj)/sum(tempb==-1)+2*a*(y(1)*c+alphaobj)*mean(A(temp_id(tempb==-1),:))';
        gradientbbobj=gradientbbobj+y(1)*2*a*mean(bbobj-tempAx(tempb==-1));
    end    
    
    gradientcst=zeros(d,2); %x gradient in cst
    gradientaacst=zeros(m,2);
    gradientbbcst=zeros(m,2);
    tempid_used=(tempb==1 & tempsv==1);
    if sum(tempid_used)>0
        gradientcst(:,1)=gradientcst(:,1)+y(2)*2*a*A(temp_id(tempid_used),:)'*(tempAx(tempid_used)-aacst(1,1))/sum(tempid_used)+2*a*(y(2)*c+alphacst(1,1))*mean(A(temp_id(tempid_used),:))';
        gradientaacst(1,1)=gradientaacst(1,1)+y(2)*2*a*mean(aacst(1,1)-tempAx(tempid_used));
    end
    tempid_used=(tempb==-1 & tempsv==0);
    if sum(tempid_used)>0
        gradientcst(:,1)=gradientcst(:,1)+y(2)*2*a*A(temp_id(tempid_used),:)'*(tempAx(tempid_used)-bbcst(1,1))/sum(tempid_used)-2*a*(y(2)*c+alphacst(1,1))*mean(A(temp_id(tempid_used),:))';
        gradientbbcst(1,1)=gradientbbcst(1,1)+y(2)*2*a*mean(bbcst(1,1)-tempAx(tempid_used));
    end
    tempid_used=(tempb==1 & tempsv==0);
    if sum(tempid_used)>0
        gradientcst(:,1)=gradientcst(:,1)+y(2)*2*a*A(temp_id(tempid_used),:)'*(tempAx(tempid_used)-aacst(1,2))/sum(tempid_used)-2*a*(y(2)*c+alphacst(1,2))*mean(A(temp_id(tempid_used),:))';
        gradientaacst(1,2)=gradientaacst(1,2)+y(2)*2*a*mean(aacst(1,2)-tempAx(tempid_used));
    end
    tempid_used=(tempb==-1 & tempsv==1);
    if sum(tempid_used)>0
        gradientcst(:,1)=gradientcst(:,1)+y(2)*2*a*A(temp_id(tempid_used),:)'*(tempAx(tempid_used)-bbcst(1,2))/sum(tempid_used)+2*a*(y(2)*c+alphacst(1,2))*mean(A(temp_id(tempid_used),:))';
        gradientbbcst(1,2)=gradientbbcst(1,2)+y(2)*2*a*mean(bbcst(1,2)-tempAx(tempid_used));
    end
    tempid_used=(tempb==1 & tempsv==0);
    if sum(tempid_used)>0
        gradientcst(:,2)=gradientcst(:,2)+y(3)*2*a*A(temp_id(tempid_used),:)'*(tempAx(tempid_used)-aacst(2,1))/sum(tempid_used)+2*a*(y(3)*c+alphacst(2,1))*mean(A(temp_id(tempid_used),:))';
        gradientaacst(2,1)=gradientaacst(2,1)+y(3)*2*a*mean(aacst(2,1)-tempAx(tempid_used));
    end
    tempid_used=(tempb==-1 & tempsv==1);
    if sum(tempid_used)>0
        gradientcst(:,2)=gradientcst(:,2)+y(3)*2*a*A(temp_id(tempid_used),:)'*(tempAx(tempid_used)-bbcst(2,1))/sum(tempid_used)-2*a*(y(3)*c+alphacst(2,1))*mean(A(temp_id(tempid_used),:))';
        gradientbbcst(2,1)=gradientbbcst(2,1)+y(3)*2*a*mean(bbcst(2,1)-tempAx(tempid_used));
    end
    tempid_used=(tempb==1 & tempsv==1);
    if sum(tempid_used)>0
        gradientcst(:,2)=gradientcst(:,2)+y(3)*2*a*A(temp_id(tempid_used),:)'*(tempAx(tempid_used)-aacst(2,2))/sum(tempid_used)-2*a*(y(3)*c+alphacst(2,2))*mean(A(temp_id(tempid_used),:))';
        gradientaacst(2,2)=gradientaacst(2,2)+y(3)*2*a*mean(aacst(2,2)-tempAx(tempid_used));
    end
    tempid_used=(tempb==-1 & tempsv==0);
    if sum(tempid_used)>0
        gradientcst(:,2)=gradientcst(:,2)+y(3)*2*a*A(temp_id(tempid_used),:)'*(tempAx(tempid_used)-bbcst(2,2))/sum(tempid_used)+2*a*(y(3)*c+alphacst(2,2))*mean(A(temp_id(tempid_used),:))';
        gradientbbcst(2,2)=gradientbbcst(2,2)+y(3)*2*a*mean(bbcst(2,2)-tempAx(tempid_used));
    end
    
    gradient=gradientobj+gradientcst(:,1)+gradientcst(:,2); %x gradient all
    
    %The following codes compute online validation temp_U (upper bound)
    errortemp=0;
    etasum=etasum+eta;
    bxsum=bxsum+eta*gradient;
    bysum=bysum+eta*grady;
    balphaobjsum=balphaobjsum+eta*fvalalpha;
    balphacstsum=balphacstsum+eta*gvalalpha;
    axsum=axsum+eta*(grady'*y)-eta*(gradient'*reshape(x,d*(numclass-1),1));
    aalphaobjsum=aalphaobjsum+eta*(fvalalpha'*alphaobj)-eta*(fvalalpha'*alphaobj);
    aalphacstsum=aalphacstsum+eta*sum(sum(gvalalpha.*alphaobj))-eta*sum(sum(gvalalpha.*alphaobj));
    temp_bxsum=reshape(bxsum,d,numclass-1);
    temp=sqrt(diag(temp_bxsum'*temp_bxsum));
    tempid=find(temp~=0);
    temp(tempid)=1./temp(tempid);
    tempx=-temp_bxsum*diag(temp)*regularization_term;
    temp_L=axsum/etasum+trace(tempx'*temp_bxsum)/etasum-errortemp;
    temp=zeros(m+1,1);
    temp(1)=(balphaobjsum/etasum)^2/4/a+bysum(1)/etasum;
    temp(2)=(balphacstsum(1,1)/etasum)^2/4/a+(balphacstsum(1,2)/etasum)^2/4/a+bysum(2)/etasum;
    temp(3)=(balphacstsum(2,1)/etasum)^2/4/a+(balphacstsum(2,2)/etasum)^2/4/a+bysum(3)/etasum;
    temp_U=max(temp);
    
    %The following line use full data to compute temp_U upper bound.
    temp_U=max([temp_optimalvalue-r0;temp_infeasibility]);
    
    Inner_L=[Inner_L;temp_L];
    Inner_U=[Inner_U;temp_U];
    
    % check if the fairness constraints is satisfied
    [~,~,~,AUC] = perfcurve(b,A*x,1);
    % g1: posfemale + negmale
    % g2: posmale + negfemale
    b_g1 = [b(sv==1 & b==1);b(sv==0 & b==-1)];
    A_g1 = [A(sv==1 & b==1,:);A(sv==0 & b==-1,:)];
    b_g2 = [b(sv==0 & b==1);b(sv==1 & b==-1)];
    A_g2 = [A(sv==0 & b==1,:);A(sv==1 & b==-1,:)];
    [~,~,~,AUC_g1] = perfcurve(b_g1,A_g1*x,1);
    [~,~,~,AUC_g2] = perfcurve(b_g2,A_g2*x,1);
    
    b_test_g1 = [b_test(sv_test==1 & b_test==1);b_test(sv_test==0 & b_test==-1)];
    A_test_g1 = [A_test(sv_test==1 & b_test==1,:);A_test(sv_test==0 & b_test==-1,:)];
    b_test_g2 = [b_test(sv_test==0 & b_test==1);b_test(sv_test==1 & b_test==-1)];
    A_test_g2 = [A_test(sv_test==0 & b_test==1,:);A_test(sv_test==1 & b_test==-1,:)];
    [~,~,~,AUC_test] = perfcurve(b_test,A_test*x,1);
    [~,~,~,AUC_test_g1] = perfcurve(b_test_g1,A_test_g1*x,1);
    [~,~,~,AUC_test_g2] = perfcurve(b_test_g2,A_test_g2*x,1);
    
    b_valid_g1 = [b_valid(sv_valid==1 & b_valid==1);b_valid(sv_valid==0 & b_valid==-1)];
    A_valid_g1 = [A_valid(sv_valid==1 & b_valid==1,:);A_valid(sv_valid==0 & b_valid==-1,:)];
    b_valid_g2 = [b_valid(sv_valid==0 & b_valid==1);b_valid(sv_valid==1 & b_valid==-1)];
    A_valid_g2 = [A_valid(sv_valid==0 & b_valid==1,:);A_valid(sv_valid==1 & b_valid==-1,:)];
    [~,~,~,AUC_valid] = perfcurve(b_valid,A_valid*x,1);
    [~,~,~,AUC_valid_g1] = perfcurve(b_valid_g1,A_valid_g1*x,1);
    [~,~,~,AUC_valid_g2] = perfcurve(b_valid_g2,A_valid_g2*x,1);

    if AUC_test > AUC_best
        AUC_best = AUC_test;
        x_best = x;
    end
    
    if abs(abs(AUC_valid_g1-AUC_valid_g2)-fair_con)<0.01        
        train_auc_list=[train_auc_list;AUC];
        train_con1_list=[train_con1_list;AUC_g1];
        train_con2_list=[train_con2_list;AUC_g2];
        
        valid_auc_list=[valid_auc_list;AUC_valid];
        valid_con1_list=[valid_con1_list;AUC_valid_g1];
        valid_con2_list=[valid_con2_list;AUC_valid_g2];
        
        test_auc_list=[test_auc_list;AUC_test];
        test_con1_list=[test_con1_list;AUC_test_g1];
        test_con2_list=[test_con2_list;AUC_test_g2];
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
    aacstold=aacst;
    aacst=aacst-eta*gradientaacst;
    bbcstold=bbcst;
    bbcst=bbcst-eta*gradientbbcst;
    
    yold=y;
    alphaobjold=alphaobj;
    alphacstold=alphacst;
    alphaobjyold=alphaobjy;
    alphacstyold=alphacsty;
    
    tempalphaobjy = (fvalalpha+2/tau*alphaobjy)/(2*a+2/tau);
    pobj= a*tempalphaobjy^2-tempalphaobjy*fvalalpha + 1/tau*(tempalphaobjy-alphaobjy)^2;
    alphaobjy=tempalphaobjy;
    tempalphacsty = (gvalalpha+2/tau*alphacsty)/(2*a+2/tau);
    pcst= a*tempalphacsty.^2-tempalphacsty.*gvalalpha + 1/tau*(tempalphacsty-alphacsty).^2;
    alphacsty=tempalphacsty;
    
    tempy=zeros(m+1,1);
    tempy(1)=y(1)*exp((fval-pobj)/(2*(1+B)^2/tau));
    tempy(2)=y(2)*exp((gval(1)-pcst(1,1)-pcst(1,2))/(2*(1+B)^2/tau));
    tempy(3)=y(3)*exp((gval(2)-pcst(2,1)-pcst(2,2))/(2*(1+B)^2/tau));
    y=tempy/sum(tempy);
    
    alphaobj=alphaobjy*y(1);
    alphacst(1,:)=alphacsty(1,:)*y(2);
    alphacst(2,:)=alphacsty(2,:)*y(3);
    
    xbar=(xbar*(k-1)+x)/k;
    ybar=(ybar*(k-1)+y)/k;
    aaobjbar=(aaobjbar*(k-1)+aaobj)/k;
    bbobjbar=(bbobjbar*(k-1)+bbobj)/k;
    aacstbar=(aacstbar*(k-1)+aacst)/k;
    bbcstbar=(bbcstbar*(k-1)+bbcst)/k;
    alphaobjbar=(alphaobjbar*(k-1)+alphaobj)/k;
    alphacstbar=(alphacstbar*(k-1)+alphacst)/k;
end
x_last=x;
x=xbar;
y=ybar;
aaobj=aaobjbar;
bbobj=bbobjbar;
aacst=aacstbar;
bbcst=bbcstbar;
alphaobj=alphaobjbar;
alphacst=alphacstbar;
alphaobjy = alphaobj/y(1);
alphacsty =  diag(1./y(2:m+1))*alphacst;
