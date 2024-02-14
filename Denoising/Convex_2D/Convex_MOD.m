
function [X,A,B] = Convex_MOD(Y,iter,errT)

[m,n,T] = size(Y);
a       = 2*m;
b       = 2*n;
lambda  = 1e-4;
DCT1    = zeros(m,a);
DCT2    = zeros(n,b);

for kk  = 0:1:a-1
    V   = cos((0:1:m-1)'*kk*pi/a);
    if kk>0 ; V = V-mean(V); end
    DCT1(:,kk+1) = V/norm(V);
end
for kk  = 0:1:b-1
    V   = cos((0:1:n-1)'*kk*pi/b);
    if kk>0 ; V = V-mean(V); end
    DCT2(:,kk+1) = V/norm(V);
end

A       = DCT1;
B       = DCT2;
% A       = normc(randn(m,2*m));
% B       = normc(randn(n,2*n));

A_old   = A;
B_old   = B;
A_pre   = A;
B_pre   = B;
X       = zeros(a,b,T);

for it = 1:iter
    
    % Step 1 : Sparse Coding (OMP_2D)
    
    C1       = A'*A;
    C2       = B'*B;  
    
    parfor i = 1:T
           temp     = Y(:,:,i) - (A-A_old)*X(:,:,i)*B_old' - A_old*X(:,:,i)*((B-B_old)');
           X(:,:,i) = OMP_2D_Err(temp,A,B,errT,C1,C2);  
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
    
    A_old    = A_pre;
    B_old    = B_pre;
    A_pre    = A;
    B_pre    = B;
    
%     fprintf('iteration: %d   \n',it)

end

