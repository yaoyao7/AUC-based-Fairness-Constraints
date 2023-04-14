[X,Y,T,AUC] = perfcurve(b,A*x,1);
[X,Y,T,AUCmale] = perfcurve(b(sv==0),A(sv==0,:)*x,1);
[X,Y,T,AUCfemale] = perfcurve(b(sv==1),A(sv==1,:)*x,1);