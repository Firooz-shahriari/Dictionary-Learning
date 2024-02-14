
function [S,A,B] = MOD_2DD(X,A0,B0,iter,sparsity)

A       = randn(size(A0));
A       = normc(A);
B       = randn(size(B0));
B       = normc(B);
a       = size(A,2);
b       = size(B,2);
[m,n,T] = size(X);
lambda  = 1e-4;
S       = zeros(a,b,size(X,3));
rpA     = zeros(1,iter);
rpB     = zeros(1,iter);
rpD     = zeros(1,iter);

for it = 1:iter
    
    % Step 1 : Sparse Coding (OMP_2D)
    
    C1       = A'*A;
    C2       = B'*B;   
    parfor i = 1:T
           S(:,:,i) = OMP_2D_Sp(X(:,:,i),A,B,sparsity,C1,C2);  
    end        
    
    if 0 < it < floor(iter/3) || floor(2*iter/3)< it <= iter

        % Step2 : computing dictionary A
        sig1 = zeros(m,a);
        sig2 = zeros(a,a);
        for i = 1:T
            tmp1 = X(:,:,i)*B*(S(:,:,i)');
            tmp2 = S(:,:,i)*(B')*B*(S(:,:,i)');
            sig1 = sig1 + tmp1;
            sig2 = sig2 + tmp2;
        end
        A = sig1/(sig2+lambda*eye(size(sig2)));
        A = normc(A);

        % Step3 : computing dictionary B
        sig3 = zeros(n,b);
        sig4 = zeros(b,b);
        for i = 1:T
            tmp3 = (X(:,:,i)')*A*S(:,:,i);
            tmp4 = (S(:,:,i)')*(A')*A*S(:,:,i);
            sig3 = sig3 + tmp3;
            sig4 = sig4 + tmp4;
        end
        B = sig3/(sig4+lambda*eye(size(sig4)));
        B = normc(B);
    else
        
        % Step3 : computing dictionary B
        sig3 = zeros(n,b);
        sig4 = zeros(b,b);
        for i = 1:T
            tmp3 = (X(:,:,i)')*A*S(:,:,i);
            tmp4 = (S(:,:,i)')*(A')*A*S(:,:,i);
            sig3 = sig3 + tmp3;
            sig4 = sig4 + tmp4;
        end
        B = sig3/(sig4+lambda*eye(size(sig4)));
        B = normc(B);
        
        
        % Step2 : computing dictionary A
        sig1 = zeros(m,a);
        sig2 = zeros(a,a);
        for i = 1:T
            tmp1 = X(:,:,i)*B*(S(:,:,i)');
            tmp2 = S(:,:,i)*(B')*B*(S(:,:,i)');
            sig1 = sig1 + tmp1;
            sig2 = sig2 + tmp2;
        end
        A = sig1/(sig2+lambda*eye(size(sig2)));
        A = normc(A);
        
    end

    [rpA(it),~] = I_findDistanseBetweenDictionaries(A0,A);
    [rpB(it),~] = I_findDistanseBetweenDictionaries(B0,B);
    fprintf('iteration: %d    Recovery_perA: %4.2f    Recovery_perB: %4.2f\n',it,rpA(it),rpB(it))
   
end
end

