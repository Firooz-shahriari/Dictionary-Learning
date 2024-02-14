%(c) Simon Hawe, Lehrstuhl fuer Datenverarbeitung Technische Universitaet
%Muenchen, 2012. Contact: simon.hawe@tum.de

clc
clear 
close all

addpath('./utilis/');
rng(1239876,'v5uniform');
% Path to tensor toolbox
addpath('./tensor_toolbox_2.5/');
% Currently possible selections
Operators    = {'TV', 'SVD', 'SVDN','DCT','RAND','ELAD','LOAD','NONE'};
Applications = {'Upsampling', 'Inpainting_mask', 'Inpainting_rnd', 'Denoising','CS'};

%IName = 'couple.png';
%IName = 'boat.png';
%IName = 'barbara.png';
IName = 'bridge.bmp';

% Width of square image patch
% First width, second height
%P_sz            = [64,64];
P_sz            = [8,8];

Nbr_of_atoms    = round(sqrt(4)*P_sz);
%Nbr_of_atoms    = round(1*P_sz);
%Nbr_of_atoms    = round(P_sz);
%Nbr_of_atoms    = 128;

S                   = [];
IdxSet              = {};
no_sep              = 0;

for j = 1:1
    
    % Get the images from which we learn
    clear Images;
    % Number of training patches
    n_patches    = 30000*1;
    Images{1}    = './training/';
    Images{2}    = IName;
    
    % S containes the entire trainings set
    [S] = complete_training_set(Images, n_patches, P_sz);
    
    % Normalize the patches and subtract the mean value
    ms = mean(S);
    S  = bsxfun(@minus,S,ms);
    S  = bsxfun(@times,S,1./(sqrt(sum(S.^2))));
    n_patches    = size(S,2);
    Learn_para   = init_learning_parameters_dict();
    if exist('LLGG','var')
        Learn_para.logger = LLGG;
    end
    
    Learn_para.d_sz = P_sz(1);
    
    if no_sep == 1
        P_sz            = prod(P_sz);
        Nbr_of_atoms    = prod(Nbr_of_atoms);
        Learn_para.d_sz = sqrt(P_sz(1));
    end
    
    % Normalizing the columns of the inital kernel
    rng(0,'v4')
    
    for i=1:numel(Nbr_of_atoms)
        D = randn(P_sz(i),Nbr_of_atoms(i));
        [U,SS,V]=svd(D,0);
        D = bsxfun(@minus,D,mean(D));
        Learn_para.D{i} = bsxfun(@times,D,1./sqrt(sum(D.^2,1)));
    end
    
    Learn_para.max_iter = 300;
    Learn_para.mu       = 1e2;    % Multiplier in log(1+mu*x^2)
    Learn_para.lambda   = 1e3; %0.335;%5e-2;  %1e9;%1e-3;    % Lagrange multiplier
    Learn_para.kappa    = 0;%0.129;%1e-1;    %1e4    % Weighting for Distinctive Terms
    Learn_para.q        = [0,1];
    
    % Displaying results every mod(iter,Learn_para.verbose) iterations
    Learn_para.verbose  = 20;
    max_out_iter = 10;%3
    res = cell(max_out_iter);
    tic
    
    lambda_end = 7e-3;%5e4;
    shrink1  = (lambda_end/Learn_para.lambda)^(1/(max_out_iter-1));
    shrink2  = (1e-2/Learn_para.kappa)^(1/(max_out_iter-1));
    
    S = reshape(S,[P_sz,n_patches]);
    
    Learn_para = learn_separable_dictionary(S, Learn_para);
    LP = Learn_para;
    LP.X=[];
    save(sprintf('DICT'),'-struct','LP');
    Learn_para.lambda = Learn_para.lambda*shrink1;
    Learn_para.X=[];
    
    toc
    LLGG = Learn_para.logger;
end

