
clc
clear
close all
 
h              = 10;
w              = 10;
a              = 20;
b              = 20;
T              = 20000;
k              = 10;
errT           = .4;
iteration      = 70;
numIteration   = 5;
num_train      = 2;
rpA            = zeros(num_train,iteration);
rpB            = zeros(num_train,iteration);
rpD            = zeros(num_train,iteration);
time2          = zeros(num_train,1);


for u = 1:num_train
    
 A0     = randn(h,a);
 B0     = randn(w,b);
 A0     = normc(A0);
 B0     = normc(B0);
 S0     = zeros(a,b,T);
 X      = zeros(h,w,T);
 sigman = sqrt(0.001);
 N      = sigman*randn(h,w,T);


for i = 1:T
    tmp          = randperm(a*b);
    tmp          = sort(tmp(1:k));
    tmp2         = S0(:,:,i);
    tmp2(tmp)    = randn(1,k);
    S0(:,:,i)    = tmp2;
    X(:,:,i)     = A0*S0(:,:,i)*(B0') + N(:,:,i);           % noisy signal 30dB
end


tic
[~,A,B,rpA(u,:),rpB(u,:)]  = DL_2DDD(X,A0,B0,iteration,numIteration,errT);
time2(u)                   = toc;


D0         = kron(B0,A0);
D          = kron(B,A);

[ratioA,~] = I_findDistanseBetweenDictionaries(A0,A);
[ratioB,~] = I_findDistanseBetweenDictionaries(B0,B);
[ratioD,~] = I_findDistanseBetweenDictionaries(D0,D);

fprintf('   Recovery_percentageA =  %8.2f\n' , ratioA)
fprintf('   Recovery_percentageB =  %8.2f\n' , ratioB)
fprintf('   Recovery_percentageD =  %8.2f\n' , ratioD)
fprintf('   Elapsed_time =  %d        num_train : %d\n',time2(u),u)

end
 
if num_train == 1
   rpA2 = rpA/ num_train;
   rpB2 = rpB/ num_train;
   rpD2 = rpD/ num_train;
else
   rpA2 = sum(rpA)/ num_train;
   rpB2 = sum(rpB)/ num_train;
   rpD2 = sum(rpD)/ num_train;
end

itr = 1:iteration;
figure(1)
plot(itr,rpA2)
hold on 
plot(itr,rpB2)
xlabel('Iteration')
ylabel('Recoveryper')
legend('RecoveryperA','RecoveryperB')
grid on
title('successful Recovery percentage wrt Iteration')

total_time = sum(time2);
fprintf('   Total_time =  %d\n',total_time)





