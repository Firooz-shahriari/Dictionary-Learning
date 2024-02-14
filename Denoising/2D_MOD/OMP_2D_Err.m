%%%% IN THE NAME OF GOD %%%%

function [X,iter] = OMP_2D_Err(Y,A,B,errT,C1,C2)

[h,a] = size(A);
[w,b] = size(B);
g     = [];
FF    = ones(a,b);
R     = Y;
X     = zeros(a,b);
I     = zeros(1,a*b/4);
J     = zeros(1,a*b/4);
Z0    = A'*Y*B;
k     = 0;

while sum(R(:).^2) >= errT^2 && k < a*b/4
    
    k             = k+1;
    corr          = abs(A'*R*B).*FF;
    [Max,indI]    = max(corr);
    [~,indJ]      = max(Max);
    I(k)          = indI(indJ);
    J(k)          = indJ;      
    FF(I(k),J(k)) = 0;    
    
    H             = C1(I(1:k), I(1:k)) .* C2(J(1:k), J(1:k));
    g(k,1)        = Z0(I(k),J(k));   
    u             = H^(-1) * g;

    tmp    = zeros(h,w);
    for gg = 1:k
       tmp = tmp + u(gg)*(A(:,I(gg))*(B(:,J(gg))'));
    end
    R      = Y - tmp;
end
    iter   = k;   
for kt =1:k
    X(I(kt),J(kt)) = u(kt);
end



