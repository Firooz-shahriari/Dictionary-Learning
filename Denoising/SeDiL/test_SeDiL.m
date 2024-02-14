
clc
clear 
close all


m       = 8;
n       = 8;
a       = 16;
b       = 16;
T       = 30000;
k       = 4;
iter    = 100;
A0      = randn(m,a);
B0      = randn(n,b);
S       = zeros(a,b,T);
X       = zeros(m,n);

para          = init_learning_parameters_dict();
para.d_sz     = m;
para.max_iter = 600;
para.mu       = 1e2;                           % Multiplier in log(1+mu*x^2)
para.lambda   = 0.0135; %5e-2;  %1e9;%1e-3;    % Lagrange multiplier
para.kappa    = 0.129 ; %0.129;  %1e-1;    %1e4         % Weighting for Distinctive Terms
para.q        = [0,1];

for i = 1:a
    A0(:,i) = A0(:,i) / norm(A0(:,i));
end

for i = 1:b
    B0(:,i) = B0(:,i) / norm(B0(:,i));
end

para.D{1}     = A0;
para.D{2}     = B0;

for i = 1:T
    tmp          = randperm(a*b);
    tmp          = sort(tmp(1:k));
    tmp2         = S(:,:,i);
    tmp2(tmp)    = randn(1,k);
    S(:,:,i)     = tmp2;
    X(:,:,i)     = A0*S(:,:,i)*(B0');
end

% Displaying results every mod(iter,Learn_para.verbose) iterations
para.verbose  = 20;
max_out_iter  = 3;%10; %3
res = cell(max_out_iter);

para = learn_separable_dictionary(X,para);

A = para.D{1};
B = para.D{2};

counterA = 0;
counterB = 0;

for j = 1:m
    cor  = A(:,j)'*A0;
    maxx = max(abs(cor));
    
    if maxx < 0.99 
        continue
    else
        tmp        = find(cor == maxx);
        A0(:,tmp)  = [];
        counterA   = counterA + 1;
    end 
end

for j = 1:m
    cor  = B(:,j)'*B0;
    maxx = max(abs(cor));
    
    if maxx < 0.99 
        continue
    else
        tmp       = find(cor == maxx);
        B0(:,tmp) = [];
        counterB   = counterB + 1;
    end 
end

recovery_percentageA = counterA*100/m;
recovery_percentageB = counterB*100/m;


