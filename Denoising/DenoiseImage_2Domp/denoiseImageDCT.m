function [IOut,output] = denoiseImageDCT(Image,sigma,K,varargin)
%==========================================================================
%   P E R F O R M   D E N O I S I N G   U S I N G   O V E R C O M P L E T E 
%                        D C T    D I C T I O N A R Y
%==========================================================================
% function IOut = denoiseImageDCT(Image,sigma,bb,K)
% denoise an image by sparsely representing each block with the
% overcomplete DCT Dictionary, and averaging the represented parts.
% Detailed description can be found in "Image Denoising Via Sparse and Redundant
% representations over Learned Dictionaries", (appeared in the 
% IEEE Trans. on Image Processing, Vol. 15, no. 12, December 2006).
% ===================================================================
% INPUT ARGUMENTS : Image - the noisy image (gray-level scale)
%                   sigma - the s.d. of the noise (assume to be white Gaussian).
%                   K - the number of atoms in the representing dictionary.
%    Optional argumeters:              
%                  'blockSize' - the size of the blocks the algorithm
%                       works. All blocks are squares, therefore the given
%                       parameter should be one number (width or height).
%                       Default value: 8.
%                  'errorFactor' - a factor that multiplies sigma in order
%                       to set the allowed representation error. In the
%                       experiments presented in the paper, it was set to 1.15
%                       (which is also the default value here).
%                  'maxBlocksToConsider' - maximal number of blocks that
%                       can be processed. This number is dependent on the memory
%                       capabilities of the machine, and performances’
%                       considerations. If the number of available blocks in the
%                       image is larger than 'maxBlocksToConsider', the sliding
%                       distance between the blocks increases. The default value
%                       is: 250000.
%                  'slidingFactor' - the sliding distance between processed
%                       blocks. Default value is 1. However, if the image is
%                       large, this number increases automatically (because of
%                       memory requirements). Larger values result faster
%                       performances (because of fewer processed blocks).
%                  'waitBarOn' - can be set to either 1 or 0. If
%                       waitBarOn==1 a waitbar, presenting the progress of the
%                       algorithm will be displayed.
% OUTPUT ARGUMENTS : IOut - a 2-dimensional array in the same size of the
%                   input image, that contains the cleaned image.
%                    output - a struct that contains that following field:
%                       D - the dictionary used for denoising
% =========================================================================


reduceDC  = 1;
[NN1,NN2] = size(Image);
C = 1.15;
maxBlocksToConsider = 260000;
slidingDis = 1;
bb = 8;
for argI = 1:2:length(varargin)
    if (strcmp(varargin{argI}, 'slidingFactor'))
        slidingDis = varargin{argI+1};
    end
    if (strcmp(varargin{argI}, 'errorFactor'))
        C = varargin{argI+1};
    end
    if (strcmp(varargin{argI}, 'maxBlocksToConsider'))
        maxBlocksToConsider = varargin{argI+1};
    end
    if (strcmp(varargin{argI}, 'blockSize'))
        bb = varargin{argI+1};
    end

end
errT  = C*sigma;
errT2 = bb*errT;
% Create an initial dictionary from the DCT frame
Pn  = ceil(sqrt(K));
DCT = zeros(bb,Pn);
for k=0:1:Pn-1
    V=cos([0:1:bb-1]'*k*pi/Pn);
    if k>0, V  =V-mean(V); end
    DCT(:,k+1) =V/norm(V);
end
A        = DCT;
B        = DCT;
DCT      = kron(DCT,DCT);
output.D = DCT;

while (prod(floor((size(Image)-bb)/slidingDis)+1)>maxBlocksToConsider)
    slidingDis = slidingDis+1;
end
[blocks,idx] = my_im2col(Image,[bb,bb],slidingDis);

%%%%%%%%%%%%%%%%%%%%%%%% If use omp2 function(1D)  %%%%%%%%%%%%%%%%%%%%%%%%
% go with jumps of 30000
% for jj = 1:30000:size(blocks,2)
% 
%     jumpSize = min(jj+30000-1,size(blocks,2));
%     if (reduceDC)
%         vecOfMeans = mean(blocks(:,jj:jumpSize));
%         blocks(:,jj:jumpSize) = blocks(:,jj:jumpSize) - repmat(vecOfMeans,size(blocks,1),1);
%     end
%     
%     Coefs = omp2(DCT,blocks(:,jj:jumpSize),[],errT2);
%      
%      if reduceDC ==1
%          blocks(:,jj:jumpSize)= Dictionary*Coefs + ones(size(blocks,1),1) * vecOfMeans;
%      else
%          blocks(:,jj:jumpSize)= Dictionary*Coefs ;
%      end
%  end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%% If use OMP_2D_Err(2D) %%%%%%%%%%%%%%%%%%%%%%%%%%%
[tmp1,tmp2] = size(blocks);
C1          = A'*A;
C2          = B'*B;
block       = zeros(tmp1,tmp2);
parfor jj=1:tmp2
    if (reduceDC)
    vecOfMeans  = mean(blocks(:,jj));
    block(:,jj) = blocks(:,jj) - repmat(vecOfMeans,tmp1,1);
    end
    Coefs = OMP_2D_Err(reshape(block(:,jj),[bb,bb]),A,B,errT2,C1,C2);
    if reduceDC ==1
     ttt = A*Coefs*B' + ones(bb,bb) * vecOfMeans;
     block(:,jj)= ttt(:);
    else
     ttt = A*Coefs*B';
     block(:,jj) = ttt(:);
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

count       = 1;
Weight      = zeros(NN1,NN2);
IMout       = zeros(NN1,NN2);
[rows,cols] = ind2sub(size(Image)-bb+1,idx);

for i = 1:length(cols)
    col     = cols(i); row = rows(i);        
    Block   = reshape(block(:,count),[bb,bb]);
    IMout(row:row+bb-1,col:col+bb-1)  = IMout(row:row+bb-1,col:col+bb-1)+Block;
    Weight(row:row+bb-1,col:col+bb-1) = Weight(row:row+bb-1,col:col+bb-1)+ones(bb);
    count                             = count+1;
end

IOut = (Image+0.034*sigma*IMout)./(1+0.034*sigma*Weight);  % lambda = 30/sigma 
