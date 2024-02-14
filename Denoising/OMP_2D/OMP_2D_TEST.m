%%%% IN THE NAME OF GOD %%%%
clc
clear
close all

%%%%%%%%%%%% NOTES: %%%%%%%%%%%%
% Normalizing causees worse results. 
% In Error mode, 2D case needed more time because of the inverse.
% however in Image denoising we need few atoms so 2D case would be better.

h        = 40;
w        = 40;
a        = 100; 
b        = 100;
A        = randn(h,a);                       % first  dictionary
B        = randn(w,b);                       % second dictionary
A        = normc(A);
B        = normc(B);        
k        = 50;                               % Sparsity
sigma    = 0.01;                             % noise variance
S        = zeros(a,b);
tmp      = randperm(a*b);        
tmp      = sort(tmp(1:k));
S(tmp)   = randn(1,k);                       % k_sparse matrix
X        = A*S*B' + sigma*randn(h,w);        % Noisy signal
errT     = 0.3;                              % error Target : practical

tic
S1       = OMP_2D(X,A,B,k);
time     = toc;
SNR1     = 20*log10(norm(S,'fro') / norm(S-S1,'fro'));
fprintf('SNR1 : %4.5f         ',SNR1)
fprintf('Time : %4.5f         my first code\n',time)


%%%%%%%%% This code does not work %%%%%%%%%
%%%%%%%%% beacuse assumes B = A that is not true always 
% C        = A'*A;
% N = zeros(a,b);% matrix for atom norms
% for i=1:a
%     for j=1:b
%         N(i,j) = (C(i,i)*C(j,j))^0.5;
%     end
% end
% tic
% S1       = omp2d(X,A,B',C,N,k);
% time     = toc;
% SNRy     = 20*log10(norm(S,'fro') / norm(S-S1,'fro'));
% fprintf('SNRy : %4.5f         ',SNRy)
% fprintf('Time : %4.5f         yong Fang code\n',time)


tic
S2       = OMP_2D_Sp(X,A,B,k);
time2    = toc;
SNR2     = 20*log10(norm(S,'fro') / norm(S-S2,'fro'));
fprintf('SNR2 : %4.5f         ',SNR2)
fprintf('Time : %4.5f         new code of 2D OMP with Sparsity\n',time2)


tic
[S3,NA]  = OMP_2D_Err(X,A,B,errT);
time3    = toc;
SNR3     = 20*log10(norm(S,'fro') / norm(S-S3,'fro'));
fprintf('SNR3 : %4.5f         ',SNR3)
fprintf('Time : %4.5f         new code of 2D OMP with Error\n',time3)


x = X(:);
tic
D = kron(B,A);
s = omp(D,x,[],k);
time4 = toc;
S4 = reshape(s,a,b);
S4 = full(S4);
SNR4  = 20*log10(norm(S,'fro') / norm(S-S4,'fro'));
fprintf('SNR4 : %4.5f         ',SNR4)
fprintf('Time : %4.5f         omp robeinstein with Sparsity\n',time4)


s2 = omp2(D,x,[],errT);
time5 = toc;
S5 = reshape(s2,a,b);
S5 = full(S5);
SNR5  = 20*log10(norm(S,'fro') / norm(S-S5,'fro'));
fprintf('SNR5 : %4.5f         ',SNR5)
fprintf('Time : %4.5f         omp2 robeinstein with Error\n',time5)


% tic
% s_hat2 = myOMP(x,D,k);
% time3  = toc; 
% s_hat2 = reshape(s_hat2,a,b);
% SNR3   = 20*log10(norm(S,'fro') / norm(S-s_hat2,'fro'));
% fprintf('SNR3 : %4.3f         ',SNR3)
% fprintf('Time : %4.3f\n       ',time3)
