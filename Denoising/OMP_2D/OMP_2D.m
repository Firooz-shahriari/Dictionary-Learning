%%%% IN THE NAME OF GOD %%%%

function X = OMP_2D(Y,A,B,s)

[h,a] = size(A);
[w,b] = size(B);
H     = [];
F     = [];
FF    = ones(a,b);
R     = Y;
X     = zeros(a,b);
ind   = zeros(1,s);
I     = zeros(1,s);
J     = zeros(1,s);

for k = 1:s

    corr         = abs(A'*R*B).*FF;
    tmp          = find (corr == max(max(corr)));
    ind(k)       = tmp(1);
    [I(k),J(k)]  = ind2sub(size(corr),ind(k));
    FF(I(k),J(k))= 0;
    F(end+1)     = sum(sum(Y.*(A(:,I(k))*B(:,J(k))')));

    tmp1         = zeros(1,k);
    tmp2         = zeros(1,k);
    
    for z = 1:k
        tmp1(z)  = sum(sum((A(:,I(z))*B(:,J(z))').*(A(:,I(k))*B(:,J(k))')));
        tmp2(z)  = sum(sum((A(:,I(k))*B(:,J(k))').*(A(:,I(z))*B(:,J(z))')));
    end
    
    H(:,end+1)   = tmp1(1:z-1);
    H(end+1,:)   = tmp2;
    
    u            = H^-1*F';
    tmp          = zeros(h,w);
    
    for g = 1:k
        tmp = tmp + u(g)*(A(:,I(g))*(B(:,J(g))'));
    end
    R = Y - tmp;

end

for k =1:s
    X(I(k),J(k)) = u(k);
end

