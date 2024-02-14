%%%%%%%%%%                 IN THE NAME OF GOD                     %%%%%%%%%
%=========================================================================%
%%%%%%%%%%%%%%%%%%%%%%%         Outputs:          %%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%  PSNROut & Time ===> Cells that each row show a different
%%%%%  Noise-variance and each column a different algorithm.
%=========================================================================%
%%%%%%%%%%%%%%%%%%%%%%%    Algorithms_Number:     %%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% Algorithm1:     ODCT
%%%%% Algorithm2:     2DMOD
%%%%% Algorithm3:     KSVD
%%%%% Algorithm4:     SeDiL
%%%%% Algorithm5:     2dCMOD
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
savedFolder = 'SavedMatFiles';


State       = [1   2   3   4   5   6   7   8   9     10  11  12  13    14  15  16  17    18 ];
BlockSize   = [8   8   8   8   8   8   8   8   8     13  13  13  13    16  16  16  16    16 ];
MaxTrain    = [1e4 1e4 1e4 2e4 2e4 2e4 4e4 4e4 4e4   4e4 4e4 4e4 4e4   4e4 4e4 4e4 4e4   6e4];
Itr_2DMOD   = [10  20  30  10  20  30  10  20  30    10  20  30  50    10  20  30  50    50 ];
Itr_2DCMOD  = [10  20  30  10  20  30  10  20  30    10  20  30  50    10  20  30  50    50 ];
Itr_KSVD    = [10  20  30  10  20  30  10  20  30    10  20  30  50    10  20  30  50    50 ];
Itr_SeDiL   = [10  50  100 10  50  100 50  100 150   50  100 150 1     50  100 150 1     1  ];

maxBlocksToConsider   = 5e7;
num_simul             = 1;
Sigma                 = [10,20,30,50];    % Noise variance
RR                    = 4;                % redundancy factor

for state = 17

maxNumBlocksToTrainOn = MaxTrain(state);
blockSize             = BlockSize(state);
K                     = RR*blockSize^2;          % number of atoms in 1D the dictionary

itr_2DMOD   = Itr_2DMOD(state);
itr_2DCMOD  = Itr_2DCMOD(state);
itr_KSVD    = Itr_KSVD(state);
% itr_SeDiL   = itr_SeDiL(state);

pathForImages = '';
imageName     = 'boat.png';
% imageName     = 'house.png';
% imageName     = 'peppers.png';

[IMin0,~] = imread(strcat([pathForImages,imageName]));
IMin0      = im2double(IMin0);
if (length(size(IMin0))>2)
    IMin0  = rgb2gray(IMin0);
end
if (max(IMin0(:))<2)
    IMin0  = IMin0*255;
end

PSNROut = cell(size(Sigma,2),7);
Time    = cell(size(Sigma,2),7);


%%%%%%%%%%%%%%%%%%%%%%%         Denoising           %%%%%%%%%%%%%%%%%%%%%%%

for numsig  = 1:size(Sigma,2)
sigma     = Sigma(numsig);
IMin      = IMin0+sigma*randn(size(IMin0));
PSNROut1  = zeros(num_simul,1); t1 = zeros(num_simul,1);
PSNROut2  = zeros(num_simul,1); t2 = zeros(num_simul,1);
PSNROut3  = zeros(num_simul,1); t3 = zeros(num_simul,1);
PSNROut4  = zeros(num_simul,1); t4 = zeros(num_simul,1);
PSNROut5  = zeros(num_simul,1); t5 = zeros(num_simul,1);
PSNROut6  = zeros(num_simul,1); t6 = zeros(num_simul,1);
PSNROut7  = zeros(num_simul,1); t7 = zeros(num_simul,1);
    
for numsim  = 1:num_simul    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% DCT
tic
[IoutDCT,~]       = denoiseImageDCT(IMin, sigma, K,'maxNumBlocksToTrainOn',maxNumBlocksToTrainOn,'maxBlocksToConsider',maxBlocksToConsider,'blockSize',blockSize);
t1(numsim)        = toc;
PSNROut1(numsim)  = 20*log10(255/sqrt(mean((IoutDCT(:)-IMin0(:)).^2)));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% MOD-2D
tic
[IoutMOD_2D,~]    = denoiseImageMODDL(IMin, sigma,itr_2DMOD,'maxNumBlocksToTrainOn',maxNumBlocksToTrainOn,'maxBlocksToConsider',maxBlocksToConsider,'blockSize',blockSize);
t2(numsim)        = toc;
PSNROut2(numsim)  = 20*log10(255/sqrt(mean((IoutMOD_2D(:)-IMin0(:)).^2)));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% KSVD 
tic
[Ioutksvd,~]      = denoiseImageKSVD(IMin, sigma,K,itr_KSVD,'maxNumBlocksToTrainOn',maxNumBlocksToTrainOn,'maxBlocksToConsider',maxBlocksToConsider,'blockSize',blockSize);
t3(numsim)        = toc;
PSNROut3(numsim)  = 20*log10(255/sqrt(mean((Ioutksvd(:)-IMin0(:)).^2)));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SeDiL
tic 
[IoutSeDil,~]     = denoiseImageSeDiL(IMin, sigma,itr_SeDiL,'maxNumBlocksToTrainOn',maxNumBlocksToTrainOn,'maxBlocksToConsider',maxBlocksToConsider,'blockSize',blockSize);
t4(numsim)        = toc;
PSNROut4(numsim)  = 20*log10(255/sqrt(mean((IoutSeDil(:)-IMin0(:)).^2)));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Convex MOD
tic 
[IoutCMOD,~]      = denoiseImageConvex(IMin, sigma,itr_2DCMOD,'maxNumBlocksToTrainOn',maxNumBlocksToTrainOn,'maxBlocksToConsider',maxBlocksToConsider,'blockSize',blockSize);
t5(numsim)        = toc;  
PSNROut5(numsim)  = 20*log10(255/sqrt(mean((IoutCMOD(:)-IMin0(:)).^2)));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Global KSVD (8x8 patches)
% tic
% [IoutGlobal,~]    = denoiseImageGlobal(IMin, 30,K);
% t6(numsim)        = toc;
% PSNROut6(numsim)  = 20*log10(255/sqrt(mean((IoutGlobal(:)-IMin0(:)).^2)));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Global SeDiL (8x8 patches otherwise learn it) 
% tic 
% [IoutSeDil,~]     = denoiseImageGlobalSeDiL(IMin, sigma,K,'maxNumBlocksToTrainOn',maxNumBlocksToTrainOn,'maxBlocksToConsider',maxBlocksToConsider,'blockSize',blockSize);
% t7(numsim)        = toc;
% PSNROut7(numsim)  = 20*log10(255/sqrt(mean((IoutSeDil(:)-IMin0(:)).^2)));
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Save the results in Cell
PSNROut{numsig,1} = PSNROut1;   Time{numsig,1} = t1;
PSNROut{numsig,2} = PSNROut2;   Time{numsig,2} = t2;
PSNROut{numsig,3} = PSNROut3;   Time{numsig,3} = t3;
PSNROut{numsig,4} = PSNROut4;   Time{numsig,4} = t4;
PSNROut{numsig,5} = PSNROut5;   Time{numsig,5} = t5;
PSNROut{numsig,6} = PSNROut6;   Time{numsig,6} = t6;
PSNROut{numsig,7} = PSNROut7;   Time{numsig,7} = t7;
end

clear t1 t2 t3 t4 t5 t6 t7
clear PSNROut1 PSNROut2 PSNROut3 PSNROut4 PSNROut5 PSNROut6 PSNROut7
clear IoutSeDil Ioutksvd IoutGlobal IoutDCT IoutCMOD IoutMOD_2D IMin IMin0 
clear numsim numsig sigma pp pathForImages K 


if(~exist(savedFolder,'dir'))
    mkdir(savedFolder)
end
save([savedFolder '\State_' num2str(state)])

end
