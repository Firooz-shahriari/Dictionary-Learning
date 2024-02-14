
function [S,A,B] = MOD_2D(X,iter,errT)

[h,w,T] = size(X);
a       = 2*h;
b       = 2*w;

DCT1     = zeros(h,a);
DCT2     = zeros(w,b);

for kk  = 0:1:a-1
    V   = cos((0:1:h-1)'*kk*pi/a);
    if kk>0 ; V = V-mean(V); end
    DCT1(:,kk+1) = V/norm(V);
end
for kk  = 0:1:b-1
    V   = cos((0:1:w-1)'*kk*pi/b);
    if kk>0 ; V = V-mean(V); end
    DCT2(:,kk+1) = V/norm(V);
end

A       = DCT1;
B       = DCT2;
% A       = normc(randn(h,2*h));
% B       = normc(randn(w,2*w));

lambda  = 1e-4;
S       = zeros(a,b,T);

for it = 1:iter
    
    % Step 1 : Sparse Coding (OMP_2D)
    
    C1       = A'*A;
    C2       = B'*B;   
    parfor i = 1:T
           S(:,:,i) = OMP_2D_Err(X(:,:,i),A,B,errT,C1,C2);  
    end        
        
    % Step2 : computing dictionary A
    
    sig1  = zeros(h,a);
    sig2  = zeros(a,a);
    for i = 1:T
        temp = B*(S(:,:,i)');
        tmp1 = X(:,:,i)*temp;
        tmp2 = temp'*temp;
        sig1 = sig1 + tmp1;
        sig2 = sig2 + tmp2;
    end
    A = sig1/(sig2+lambda*eye(size(sig2)));
    A = normc(A);
    
    % Step3 : computing dictionary B
    
    sig3  = zeros(w,b);
    sig4  = zeros(b,b);
    for i = 1:T
        temp = A*S(:,:,i);
        tmp3 = (X(:,:,i)')*temp;
        tmp4 = temp'*temp;
        sig3 = sig3 + tmp3;
        sig4 = sig4 + tmp4;
    end
    B = sig3/(sig4+lambda*eye(size(sig4)));
    B = normc(B);

%     fprintf('iteration: %d   \n',it)
   
end

