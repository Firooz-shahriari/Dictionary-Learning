
clc
clear

pathForImages    = '';
imageName        = 'lena.png';
[Image,pp]       = imread(strcat([pathForImages,imageName]));

bb               = 8;
K                = 4*bb^2;
[NN1,NN2]        = size(Image);
Pn               = ceil(sqrt(K));
DCT              = zeros(bb,Pn); 
NumberOfBlocks   = 5; 

if(prod([NN1,NN2]-bb+1)> NumberOfBlocks)
    randPermutation     =  randperm(prod([NN1,NN2]-bb+1));
    selectedBlocks      =  randPermutation(1:NumberOfBlocks);
    blkMatrix           = zeros(bb^2,NumberOfBlocks);
    for i = 1:NumberOfBlocks
        [row,col]       = ind2sub(size(Image)-bb+1,selectedBlocks(i));
        currBlock       = Image(row:row+bb-1,col:col+bb-1);
        blkMatrix(:,i)  = currBlock(:);
    end
else
    blkMatrix           = im2col(Image,[bb,bb],'sliding');
end


for k = 0:1:Pn-1
    V = cos([0:1:bb-1]'*k*pi/Pn);
    if k>0, V  = V-mean(V); end
    DCT(:,k+1) = V/norm(V);
end

DCT    = kron(DCT,DCT);
DCT    = normc(DCT);

errT1  = 10;
errT2  = 80;
sp1    = OMPerr(DCT,blkMatrix,errT1);
sp2    = omp2(DCT,blkMatrix,[],errT2);



%      Result : For having the same results we should have:
%    
%               errT2 = errT1 * size(data,1) 





















