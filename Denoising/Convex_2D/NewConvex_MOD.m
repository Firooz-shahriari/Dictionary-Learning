
function [X,A,B,rpA,rpB,RMSE] = NewConvex_MOD(Y,A0,B0,iter,sparsity)

A       = randn(size(A0));
A       = normc(A);
B       = randn(size(B0));
B       = normc(B);
A_pre   = A;
B_pre   = B;
A_old   = A;
B_old   = B;
a       = size(A,2);
b       = size(B,2);
[m,n,T] = size(Y);
lambda  = 1e-4;
X       = zeros(a,b,size(Y,3));
rpA     = zeros(1,iter);
rpB     = zeros(1,iter);
Res     = zeros(m,n,T);
RMSE    = zeros(1,iter);

for it = 1:iter
    
    % Step 1 : Sparse Coding (OMP_2D)
    
    C1 = A'*A;
    C2 = B'*B;
    
    parfor i = 1:T
           temp     = Y(:,:,i) - A*X(:,:,i)*B' + A_old*X(:,:,i)*B_old';
           X(:,:,i) = OMP_2D_Sp(temp,A,B,sparsity,C1,C2);  
    end        
        
    % Step2 : computing dictionary A
    
    sig1  = zeros(m,a);
    sig2  = zeros(a,a);
    for i = 1:T
        temp = B*(X(:,:,i)');
        tmp1 = Y(:,:,i)*temp;
        tmp2 = temp'*temp;
        sig1 = sig1 + tmp1;
        sig2 = sig2 + tmp2;
    end
    A = sig1/(sig2+lambda*eye(size(sig2)));
    A = normc(A);
    
    % Step3 : computing dictionary B
    
    sig3  = zeros(n,b);
    sig4  = zeros(b,b);
    for i = 1:T
        temp = A*X(:,:,i);
        tmp3 = (Y(:,:,i)')*temp;
        tmp4 = temp'*temp;
        sig3 = sig3 + tmp3;
        sig4 = sig4 + tmp4;
    end
    B = sig3/(sig4+lambda*eye(size(sig4)));
    B = normc(B);
    
    A_old = A_pre;
    B_old = B_pre;
    A_pre = A;
    B_pre = B;
    for i=1:T
        Res(:,:,i) = Y(:,:,i)-A*X(:,:,i)*B';
    end
    error       = sum(Res(:).^2);
    RMSE(it)    = sqrt(error/(m*n*T)); 
    [rpA(it),~] = I_findDistanseBetweenDictionaries(A0,A);
    [rpB(it),~] = I_findDistanseBetweenDictionaries(B0,B);
    fprintf('iteration: %d    Recovery_perA: %4.2f    Recovery_perB: %4.2f     RMSE: %4.4f\n',it,rpA(it),rpB(it),RMSE(it))
   
end

