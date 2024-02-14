%%%%%%%%%%                 IN THE NAME OF GOD                     %%%%%%%%%
%=========================================================================%
%%%%%%%%%%%%%%%%%%%%%%%         Outputs:          %%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%  PSNROut & Time 3D Matrices that
%%%%%  Row      : Noise-Variance
%%%%%  Column   : simulation
%%%%%  Fiber    : Algorithm
%=========================================================================%
%%%%%%%%%%%%%%%%%%%%%%%    Algorithms_Number:     %%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% Algorithm1:     ODCT
%%%%% Algorithm2:     2DMOD
%%%%% Algorithm3:     KSVD
%%%%% Algorithm4:     SeDiL
%%%%% Algorithm5:     2DCMOD
%%%%% Algorithm6:     GlobalKSVD
%%%%% Algorithm7:     GlobalSeDiL
%=========================================================================%
%%%%%%%%%%%%%%%%%%%%%%%       Initializaion         %%%%%%%%%%%%%%%%%%%%%%%
clc
clear
close all
warning off 

addpath('.\2Dto1D')
addpath('.\DenoiseImage_2Domp')
addpath('.\KSVD_1D')
addpath('.\2D_MOD')
addpath('.\SeDiL')
addpath('.\SeDiL\tensor_toolbox_2.5')
addpath('.\SeDiL\utilis')
addpath('.\SL0_2D_1D')
addpath('.\OMP_2D')
addpath('.\Convex_2D')
savedFolder = 'SavedMatFilesNew';


State       = [1   2   3   4   5   6   7   8   9   10  11  12  13  14  15  16  17  18  19  20  21  22  23  24];
BlockSize   = [8   8   8   8   8   8   12  12  12  12  12  12  16  16  16  16  16  16  12  12  12  12  12  12];
MaxTrain    = [4e4 4e4 4e4 4e4 4e4 4e4 4e4 4e4 4e4 4e4 4e4 4e4 4e4 4e4 4e4 4e4 4e4 4e4 2e4 2e4 2e4 2e4 2e4 2e4];
Itr_2DMOD   = [10  20  30  40  50  100 10  20  30  40  50  100 10  20  30  40  50  100 10  20  30  40  50  100];
Itr_2DCMOD  = [10  20  30  40  50  100 10  20  30  40  50  100 10  20  30  40  50  100 10  20  30  40  50  100];
Itr_KSVD    = [10  20  30  40  50  100 10  20  30  40  50  100 10  20  30  40  50  100 10  20  30  40  50  100];
Itr_SeDiL   = [10  20  30  40  50  100 10  20  30  40  50  100 10  20  30  40  50  100 10  20  30  40  50  100];

maxBlocksToConsider   = 5e11;
NumAlg                = 5;                % Number of Algorithms
num_simul             = 5;                % Number of simulations
Sigma                 = [10,20,30,50];    % Noise variance
RR                    = 4;                % redundancy factor

for state = State

maxNumBlocksToTrainOn = MaxTrain(state);  % Number of Training Signals
blockSize             = BlockSize(state); % Patch-Size
K                     = RR*blockSize^2;   % number of atoms in 1D the dictionary

itr_2DMOD   = Itr_2DMOD(state);
itr_2DCMOD  = Itr_2DCMOD(state);
itr_KSVD    = Itr_KSVD(state);
itr_SeDiL   = Itr_SeDiL(state);

pathForImages = '';
imageName     = 'boat.png';
% imageName     = 'house.png';
% imageName     = 'peppers.png';

[IMin0,~]  = imread(strcat([pathForImages,imageName]));
IMin0      = im2double(IMin0);
if (length(size(IMin0))>2)
    IMin0  = rgb2gray(IMin0);
end
if (max(IMin0(:))<2)
    IMin0  = IMin0*255;
end

PSNROut = zeros(size(Sigma,2),num_simul,NumAlg);
Time    = zeros(size(Sigma,2),num_simul,NumAlg);


%%%%%%%%%%%%%%%%%%%%%%%         Denoising           %%%%%%%%%%%%%%%%%%%%%%%

for numsig  = 1:size(Sigma,2)
sigma       = Sigma(numsig);
IMin        = IMin0+sigma*randn(size(IMin0));
    
for numsim  = 1:num_simul    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% DCT
tic
[IoutDCT,~]              = denoiseImageDCT(IMin, sigma, K,'maxNumBlocksToTrainOn',maxNumBlocksToTrainOn,'maxBlocksToConsider',maxBlocksToConsider,'blockSize',blockSize,'displayFlag',0);
Time(numsig,numsim,1)    = toc;
PSNROut(numsig,numsim,1) = 20*log10(255/sqrt(mean((IoutDCT(:)-IMin0(:)).^2)));
fprintf('\n State:%2.0d, Sigma:%2.0f, Simulation:%2.0d   DCT     ==> Done ',state,sigma,numsim)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% MOD-2D
tic
[IoutMOD_2D,~]           = denoiseImageMODDL(IMin, sigma,itr_2DMOD,'maxNumBlocksToTrainOn',maxNumBlocksToTrainOn,'maxBlocksToConsider',maxBlocksToConsider,'blockSize',blockSize,'displayFlag',0);
Time(numsig,numsim,2)    = toc;
PSNROut(numsig,numsim,2) = 20*log10(255/sqrt(mean((IoutMOD_2D(:)-IMin0(:)).^2)));
fprintf('\n State:%2.0d, Sigma:%2.0f, Simulation:%2.0d   2DMOD   ==> Done ',state,sigma,numsim)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% KSVD 
tic
[Ioutksvd,~]             = denoiseImageKSVD(IMin, sigma,K,itr_KSVD,'maxNumBlocksToTrainOn',maxNumBlocksToTrainOn,'maxBlocksToConsider',maxBlocksToConsider,'blockSize',blockSize,'displayFlag',0,'waitBarOn',0);
Time(numsig,numsim,3)    = toc;
PSNROut(numsig,numsim,3) = 20*log10(255/sqrt(mean((Ioutksvd(:)-IMin0(:)).^2)));
fprintf('\n State:%2.0d, Sigma:%2.0f, Simulation:%2.0d   KSVD    ==> Done ',state,sigma,numsim)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SeDiL
tic 
[IoutSeDil,~]            = denoiseImageSeDiL(IMin, sigma,itr_SeDiL,'maxNumBlocksToTrainOn',maxNumBlocksToTrainOn,'maxBlocksToConsider',maxBlocksToConsider,'blockSize',blockSize,'displayFlag',0);
Time(numsig,numsim,4)    = toc;
PSNROut(numsig,numsim,4) = 20*log10(255/sqrt(mean((IoutSeDil(:)-IMin0(:)).^2)));
fprintf('\n State:%2.0d, Sigma:%2.0f, Simulation:%2.0d   SeDiL   ==> Done ',state,sigma,numsim)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Convex MOD
tic 
[IoutCMOD,~]             = denoiseImageConvex(IMin, sigma,itr_2DCMOD,'maxNumBlocksToTrainOn',maxNumBlocksToTrainOn,'maxBlocksToConsider',maxBlocksToConsider,'blockSize',blockSize,'displayFlag',0);
Time(numsig,numsim,5)    = toc;  
PSNROut(numsig,numsim,5) = 20*log10(255/sqrt(mean((IoutCMOD(:)-IMin0(:)).^2)));
fprintf('\n State:%2.0d, Sigma:%2.0f, Simulation:%2.0d   2DCMOD  ==> Done ',state,sigma,numsim)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Global KSVD (8x8 patches)
% tic
% [IoutGlobal,~]           = denoiseImageGlobal(IMin, 30,K);
% Time(numsig,numsim,6)    = toc;
% PSNROut(numsig,numsim,6) = 20*log10(255/sqrt(mean((IoutGlobal(:)-IMin0(:)).^2)));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Global SeDiL (8x8 patches otherwise learn it) 
% tic 
% [IoutSeDil,~]            = denoiseImageGlobalSeDiL(IMin, sigma,K,'maxNumBlocksToTrainOn',maxNumBlocksToTrainOn,'maxBlocksToConsider',maxBlocksToConsider,'blockSize',blockSize);
% Time(numsig,numsim,7)    = toc;
% PSNROut(numsig,numsim,7) = 20*log10(255/sqrt(mean((IoutSeDil(:)-IMin0(:)).^2)));
end
fprintf('========================================================================================================')
fprintf('========================================================================================================')
end

clear IoutSeDil Ioutksvd IoutGlobal IoutDCT IoutCMOD IoutMOD_2D IMin IMin0 
clear numsim numsig sigma pp pathForImages K 

if(~exist(savedFolder,'dir'))
    mkdir(savedFolder)
end
save([savedFolder '\State_' num2str(state)])

fprintf('========================================================================================================')
fprintf('========================================================================================================')
fprintf('========================================================================================================')
end
