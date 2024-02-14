
function [IOut,output] = denoiseImageMODDL(Image,sigma,iteration,varargin)

% first, train a dictionary on the noisy image

reduceDC               = 1;
[NN1,NN2]              = size(Image);
C                      = 1.15;
maxBlocksToConsider    = 260000;
slidingDis             = 1;
bb                     = 8;
maxNumBlocksToTrainOn  = 20000;
displayFlag            = 1;

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
    if (strcmp(varargin{argI}, 'maxNumBlocksToTrainOn'))
        maxNumBlocksToTrainOn = varargin{argI+1};
    end
    if (strcmp(varargin{argI}, 'displayFlag'))
        displayFlag = varargin{argI+1};
    end
end

% first, train a dictionary on blocks from the noisy image

if(prod([NN1,NN2]-bb+1)> maxNumBlocksToTrainOn)
    randPermutation    =  randperm(prod([NN1,NN2]-bb+1));
    selectedBlocks     = randPermutation(1:maxNumBlocksToTrainOn);

    blkMatrix = zeros(bb^2,maxNumBlocksToTrainOn);
    for i = 1:maxNumBlocksToTrainOn
        [row,col]      = ind2sub(size(Image)-bb+1,selectedBlocks(i));
        currBlock      = Image(row:row+bb-1,col:col+bb-1);
        blkMatrix(:,i) = currBlock(:);
    end
else
    blkMatrix = im2col(Image,[bb,bb],'sliding');
end

inp    = zeros(bb,bb,size(blkMatrix,2));
for ii = 1:size(blkMatrix,2)
   inp(:,:,ii) = reshape(blkMatrix(:,ii),[bb,bb]);  
end

errT1        = sigma*C;
errT2        = bb*errT1;

[~,A,B]      = MOD_2D(inp,iteration,errT2);

output.A    = A;
output.B    = B;
output.D    = kron(B,A);
Dictionary  = output.D;

if (displayFlag)
    disp('finished Trainning dictionary');
end

% denoise the image using the resulted dictionary

while (prod(floor((size(Image)-bb)/slidingDis)+1)>maxBlocksToConsider)
    slidingDis = slidingDis+1;
end
[blocks,idx]   = my_im2col(Image,[bb,bb],slidingDis);

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
%    % Coefs = OMPerr(Dictionary,blocks(:,jj:jumpSize),errT1);
%     Coefs = omp2(Dictionary,blocks(:,jj:jumpSize),[],errT2);
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

IOut = (Image+0.034*sigma*IMout)./(1+0.034*sigma*Weight);     % lambda = 30/sigma 

