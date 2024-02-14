%%%%                       IN THE NAME OF GOD                          %%%%

%==========================================================================
%%%% This code had run to Compute RMSE and RD in each iteration for all of
%%%% the algorithms. Signals are of size 10x10. 
%%%% When the size of signal grows, the RD Computing is time-consuming, and
%%%% due to the large number of iterations, especially by growing the size 
%%%% of signals, it is better to use main.m code
%%%% and compute the RD in one separate code for especial iterations, for
%%%% example in 1:step:NumIter, which step is eaual to NumIter/NumPoints.
%==========================================================================

clc
clear
close all

%%%% method 1 = 1DMOD
%%%% methid 2 = 1DKSVD
%%%% method 3 = 2DMOD
%%%% method 4 = 2DCMOD1
%%%% methid 5 = 2DCMOD2 Shuffling the steps in each iteraion of 2DCMOD1
%%%% method 6 = DL_2D1  Compute M and C with First  approach (Team  Proposed)
%%%% method 7 = DL_2D2  Compute M and C with Second approach (Hesam Proposed)

%%%% I did some simulations and found that DL2D1 and DL2D2 get the same
%%%% results. So in theese simulation, we see only DL2D1. 

%%%% all initial dictionaries in DL algorithms are produced randomly, 
%%%% not by choosing random training datas as column of initial dictionary.

%%%% 2D : A = mxa       B = nxb    Xi = axb     Yi = mxn 
%%%% 1D : D = mn x ab   y = mnx1   x  = abx1


addpath('Private')
addpath('Algs')

n            = 10;
m            = n;
a            = 2*n;
b            = 2*n;
num_train    = 25;
T            = 500*n;
it_in        = 10;
SNR          = 30;
sigman       = 1/sqrt(10^(SNR/10));
sp1D         = 1;

savedFolder = strcat('SavedMatFiles_N',num2str(n),'_T_', num2str(T) );

avRD      = cell(6,5);
avRMSE    = cell(6,5);
avTIME    = zeros(6,5);

State     = [1     2     3     4     5  ];
sparsity  = [5     7     10    13    15 ];
OuterIter = [200   250   300   350   400];

for state = State
 
s            = sparsity(state);
num_iter     = OuterIter(state);
NumOfPoints  = num_iter;

%===============================================================================================
%===============================================================================================
A0           = zeros(m,a,num_train);          B0         = zeros(n,b,num_train);
D0           = zeros(m*n,a*b,num_train);      X0         = zeros(a,b,T,num_train);
Y2D          = zeros(m,n,T,num_train);        Y1D        = zeros(m*n,T,num_train);
A2DMOD       = zeros(m,a,num_iter,num_train); B2DMOD     = zeros(n,b,num_iter,num_train);
A2DCMOD1     = zeros(m,a,num_iter,num_train); B2DCMOD1   = zeros(n,b,num_iter,num_train);
A2DCMOD2     = zeros(m,a,num_iter,num_train); B2DCMOD2   = zeros(n,b,num_iter,num_train);
Aint         = zeros(m,a,num_train);          Bint       = zeros(n,b,num_train);
RMSE1DMOD    = zeros(num_train,num_iter);     Dint       = zeros(m*n,a*b,num_train);
RMSE2DCMOD1  = zeros(num_train,num_iter);     D1DMOD     = zeros(m*n,a*b,num_iter,num_train);
RMSE1DKSVD   = zeros(num_train,num_iter);     D1DKSVD    = zeros(m*n,a*b,num_iter,num_train);
RMSE2DMOD  = zeros(num_train,num_iter); 
RMSE2DCMOD2  = zeros(num_train,num_iter);     TIME2DMOD  = zeros(num_train,1);
TIME2DCMOD1  = zeros(num_train,1);            TIME1DKSVD = zeros(num_train,1);
TIME2DCMOD2  = zeros(num_train,1);            TIME1DMOD  = zeros(num_train,1);
%===============================================================================================
%===============================================================================================

for u = 1:num_train

fprintf('\t #State = %d \t #Simulation = %d\n', state , u)
  
A0(:,:,u)        = normc(randn(m,a));
B0(:,:,u)        = normc(randn(n,b));
D0(:,:,u)        = normc(kron(B0(:,:,u),A0(:,:,u)));

for i=1:T
   tmp           = randperm(a*b);
   tmp           = sort(tmp(1:s));
   tmp2          = X0(:,:,i,u);
   tmp2(tmp)     = randn(1,s);
   X0(:,:,i,u)   = tmp2;
   Y2D(:,:,i,u)  = A0(:,:,u)*X0(:,:,i,u)*B0(:,:,u)' + sigman*randn(m,n);
end

Y1D(:,:,u)       = reshape(Y2D(:,:,:,u),[m*n,T]);
Aint(:,:,u)      = normc(randn(m,a));
Bint(:,:,u)      = normc(randn(n,b));
Dint(:,:,u)      = normc(kron(Bint(:,:,u),Aint(:,:,u)));

%===============================================================================================
%===============================================================================================
%%%%%%%%%%%    For Ksvd Algorithm (Dr. Rubeinstein Code)      %%%%%%%%%%%%%     
params.cmperr    = 1;            % to compute RMSE
params.lam       = 1e-1;         % regularization parameter of the l_1 norm
params.initx     = eye(a*b,T);   % initial coefficient matrix
params.iternum   = num_iter;     % number of DL iterations
params.Tdata     = s;            % number of non-zeros
params.dictsize  = a*b;          % number of atoms
params.memusage  = 'normal';
params.data      = Y1D(:,:,u);
params.initdict  = Dint(:,:,u);
params.exact     = 1;
%===============================================================================================
%===============================================================================================

tic;[~,D1DMOD(:,:,:,u),RMSE1DMOD(u,:)]                           = MOD(Y1D(:,:,u),a*b,num_iter,s,Dint(:,:,u));t1=toc; TIME1DMOD(u)=t1;     
fprintf('\t\t 1DMOD  \t Finished!\n ')
tic;[~,D1DKSVD(:,:,:,u),RMSE1DKSVD(u,:),~,~]                     = ksvd(params,'');t2=toc; TIME1DKSVD(u)=t2;
fprintf('\t\t 1DKSVD \t Finished!\n ')
tic;[~,A2DMOD(:,:,:,u),B2DMOD(:,:,:,u),RMSE2DMOD(u,:)]           = MOD_2D(Y2D(:,:,:,u),a,b,num_iter,s,Aint(:,:,u),Bint(:,:,u));t3=toc; TIME2DMOD(u)=t3;         
fprintf('\t\t 2DMOD  \t Finished!\n ')
tic;[~,A2DCMOD1(:,:,:,u),B2DCMOD1(:,:,:,u),RMSE2DCMOD1(u,:)]     = CMOD2D1(Y2D(:,:,:,u),a,b,num_iter,s,Aint(:,:,u),Bint(:,:,u));t4=toc; TIME2DCMOD1(u)=t4;
fprintf('\t\t 2DCMOD1\t Finished!\n ')
tic;[~,A2DCMOD2(:,:,:,u),B2DCMOD2(:,:,:,u),RMSE2DCMOD2(u,:)]     = CMOD2D2(Y2D(:,:,:,u),a,b,num_iter,s,Aint(:,:,u),Bint(:,:,u));t5=toc; TIME2DCMOD2(u)=t5;
fprintf('\t\t 2DCMOD2\t Finished!\n ')

fprintf('_________________________________________________\n\n')

end

%===============================================================================================
%===============================================================================================

% Due to the low speed of computing RD, we can compute this metric in some
% specific iterations Not all of them, But for n=10, it is OK to compute here.
    
step           = floor(num_iter/NumOfPoints);
points         = 1:step:num_iter;
RD1DMOD        = zeros(num_train,length(points));
RD1DKSVD       = zeros(num_train,length(points));
RD2DMOD        = zeros(num_train,length(points));
RD2DCMOD1      = zeros(num_train,length(points));
RD2DCMOD2      = zeros(num_train,length(points));
counter        = 0;

for uu = 1:num_train
    counter = 0;
    for pot = 1:step:num_iter
        counter                = counter + 1;
        RD1DMOD(uu,counter)    = I_findDistanseBetweenDictionaries(D0(:,:,uu),D1DMOD(:,:,pot,uu));
        RD1DKSVD(uu,counter)   = I_findDistanseBetweenDictionaries(D0(:,:,uu),D1DKSVD(:,:,pot,uu));
        RD2DMOD(uu,counter)    = I_findDistanseBetweenDictionaries(D0(:,:,uu),kron(B2DMOD(:,:,pot,uu),A2DMOD(:,:,pot,uu)));
        RD2DCMOD1(uu,counter)  = I_findDistanseBetweenDictionaries(D0(:,:,uu),kron(B2DCMOD1(:,:,pot,uu),A2DCMOD1(:,:,pot,uu)));
        RD2DCMOD2(uu,counter)  = I_findDistanseBetweenDictionaries(D0(:,:,uu),kron(B2DCMOD2(:,:,pot,uu),A2DCMOD2(:,:,pot,uu)));
    end
end


%===============================================================================================
%===============================================================================================

% Averaging on the number of simulations(num_train)

avRD1DMOD      = mean(RD1DMOD);
avRD1DKSVD     = mean(RD1DKSVD);
avRD2DMOD      = mean(RD2DMOD);
avRD2DCMOD1    = mean(RD2DCMOD1);
avRD2DCMOD2    = mean(RD2DCMOD2);

avRMSE1DMOD    = mean(RMSE1DMOD);
avRMSE1DKSVD   = mean(RMSE1DKSVD);
avRMSE2DMOD    = mean(RMSE2DMOD);
avRMSE2DCMOD1  = mean(RMSE2DCMOD1);
avRMSE2DCMOD2  = mean(RMSE2DCMOD2);

avTIME1DMODiter     = (sum(TIME1DMOD)/num_train)/num_iter;
avTIME1DKSVDiter    = (sum(TIME1DKSVD)/num_train)/num_iter;
avTIME2DMODiter     = (sum(TIME2DMOD)/num_train)/num_iter;
avTIME2DCMOD1iter   = (sum(TIME2DCMOD1)/num_train)/num_iter;
avTIME2DCMOD2iter   = (sum(TIME2DCMOD2)/num_train)/num_iter;

%===============================================================================================
%===============================================================================================
avRMSE{1,state} = avRMSE1DMOD;
avRMSE{2,state} = avRMSE1DKSVD;
avRMSE{3,state} = avRMSE2DMOD;
avRMSE{4,state} = avRMSE2DCMOD1;
avRMSE{5,state} = avRMSE2DCMOD2;

avRD{1,state} = avRD1DMOD;
avRD{2,state} = avRD1DKSVD;
avRD{3,state} = avRD2DMOD;
avRD{4,state} = avRD2DCMOD1;
avRD{5,state} = avRD2DCMOD2;

avTIME(1,state) = avTIME1DMODiter;
avTIME(2,state) = avTIME1DKSVDiter;
avTIME(3,state) = avTIME2DMODiter;
avTIME(4,state) = avTIME2DCMOD1iter;
avTIME(5,state) = avTIME2DCMOD2iter;
%===============================================================================================
%===============================================================================================
%%%%%%%%%%%%%%%%%%%%%%% TO AVOID BIG SAVED FILES %%%%%%%%%%%%%%%%%%%%%%%%%%
clear 't1' 't2' 't3' 't4' 't5' 'u' 'uu' 'tmp' 'tmp2'
clear 'X0' 'Y1D' 'Y2D'  %% --> These variables need huge memory.
%===============================================================================================
%===============================================================================================
% Save the Mat File
if(~exist(savedFolder,'dir'))
    mkdir(savedFolder)
end
save([savedFolder '\State_' num2str(state) '_n_' num2str(n) '_s_' num2str(s) '_Iterations_' num2str(num_iter) '_NumTrain_' num2str(T) ] , '-v7.3' , '-nocompression')
%===============================================================================================
%===============================================================================================

end

clearvars -except avRD avRMSE avTIME savedFolder 
save([savedFolder '\Output'])


