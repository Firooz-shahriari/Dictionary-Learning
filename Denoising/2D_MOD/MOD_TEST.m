
clc
clear
close all
 
m         = 10;
n         = 10;
a         = 20;
b         = 20;
T         = 2000;
k         = 10;
iter      = 100;
num_train = 2;
rpA       = zeros(num_train,iter);
rpB       = zeros(num_train,iter);
rpD       = zeros(num_train,iter);
time      = zeros(num_train,1);
ratioA    = zeros(num_train,1);
ratioB    = zeros(num_train,1);
ratioD    = zeros(num_train,1);


for u = 1:num_train
    
A0     = randn(m,a);
B0     = randn(n,b);
A0     = normc(A0);
B0     = normc(B0);
S0     = zeros(a,b,T);
X      = zeros(m,n,T);
sigman = .0001;
N      = sigman*randn(m,n,T);


for i = 1:T
    tmp          = randperm(a*b);
    tmp          = sort(tmp(1:k));
    tmp2         = S0(:,:,i);
    tmp2(tmp)    = randn(1,k);
    S0(:,:,i)    = tmp2;
    X(:,:,i)     = A0*S0(:,:,i)*(B0') + N(:,:,i);
end

tic
[S,A,B,rpA(u,:),rpB(u,:)]  = MOD_2D(X,A0,B0,iter,k);
time(u)    = toc;


D0         = kron(B0,A0);
D          = kron(B,A);

[ratioA(u),~] = I_findDistanseBetweenDictionaries(A0,A);
[ratioB(u),~] = I_findDistanseBetweenDictionaries(B0,B);
[ratioD(u),~] = I_findDistanseBetweenDictionaries(D0,D);

fprintf('   Recovery_percentageA =  %4.2f\n' , ratioA(u))
fprintf('   Recovery_percentageB =  %4.2f\n' , ratioB(u))
fprintf('   Recovery_percentageD =  %4.2f\n' , ratioD(u))
fprintf('   Elapsed_time =  %8.2f      Trainig_Iteraion:   %d\n\n\n',time(u),u)

end

rpA2 = sum(rpA)/ num_train;
rpB2 = sum(rpB)/ num_train;

itr = 1:iter;
figure(1)
plot(itr,rpA2)
hold on 
plot(itr,rpB2)
xlabel('Iteration')
ylabel('Recovery Per')
legend('RecoveryperA','RecoveryperB')
grid on
title('successful Recovery Percentage wrt Iteration')

total_time = sum(time);
fprintf('     Average Recovery Per A   =  %8.2f\n',rpA2(end))
fprintf('     Average Recovery Per B   =  %8.2f\n',rpB2(end))
fprintf('     Average Recovery Per D   =  %8.2f\n',sum(ratioD)/num_train)
fprintf('     Average Time of Training = %8.2f\n',total_time/num_train)

