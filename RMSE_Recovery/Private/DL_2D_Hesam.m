
function [Xtrue,As,Bs,t1,t2,RMSE] = DL_2D_Hesam(Y,a,b,iter,numIteration1D,SP,sparsity,A0,B0)

[m,n,T]      = size(Y);

h = m;
% a = 50;
% b = 50;
w = n;
L = T;
sp1d = SP;
inner_iters = iter;
DL_iters = numIteration1D;
tol = 1e-10;

As = zeros(h,a,inner_iters);
Bs = zeros(w,b,inner_iters);
RMSE = zeros(1,inner_iters);
Res = zeros(m,n,T);
t1 = zeros(1,iter);
t2 = zeros(1,iter);
A = A0;
A = normc(A);
B = B0;
B = normc(B);
SM = zeros(a,b,L);
SC = zeros(a,b,L);
M = zeros(h,b,L);
C = zeros(w,a,L);

for inner_iter = 1:inner_iters
    tic
%         fprintf('%.3d',inner_iter);
    [BpinvT,V1] = pinvT_V1(B,tol);        
    for l=1:L
%             M(:,:,l)=Y(:,:,l)*BpinvT + A*SM(:,:,l)*(eye(b)-V1*V1');
        M(:,:,l)=Y(:,:,l)*BpinvT;
    end        
    Mcat = reshape(M,[h,b*L]);

    [A,SMcat] = DL_alg(Mcat,A,DL_iters,sp1d); 
    As(:,:,inner_iter) = A;
    SM = reshape(SMcat,[a,b,L]);
    t11 = toc;
    t1(inner_iter) = t11;
    
    [ApinvT,V1] = pinvT_V1(A,tol);
    for l=1:L
%             C(:,:,l) = Y(:,:,l)'*ApinvT + B*SC(:,:,l)'*(eye(a)-V1*V1');
        C(:,:,l) = Y(:,:,l)'*ApinvT;
    end        
    Ccat = reshape(C,[w,a*L]);        

    [B,SCcat] = DL_alg(Ccat,B,DL_iters,sp1d);   
    Bs(:,:,inner_iter) = B;
    SC = reshape(full(SCcat),[b,a,L]);
    SC = permute(SC,[2 1 3]);
%         fprintf(repmat('\b',1,3));
%         addpoints(hnadl,inner_iter,mean(A_recovery(:,inner_iter)));
%         drawnow

    t22 = toc -t11;
    t2(inner_iter) = t22;
    Xtrue    = zeros(a,b,T);
    C1       = A'*A;
    C2       = B'*B;
    parfor i = 1:T
           Xtrue(:,:,i) = OMP_2D_Sp(Y(:,:,i),A,B,sparsity,C1,C2);  
    end

    for i=1:T
        Res(:,:,i)  = Y(:,:,i)-A*Xtrue(:,:,i)*B';
    end
    error           = sum(Res(:).^2);
    RMSE(inner_iter)       = sqrt(error/(m*n*T)); 
end
end
