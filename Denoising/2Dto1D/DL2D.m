function  [Sn,A,B] = DL2D(Y,iteration,numIteration,errT)

[h,w,L] = size(Y);
a       = 2*h;   
b       = 2*w;

DCT    = zeros(h,a);
for kk = 0:1:a-1
    V  = cos((0:1:size(Y,1)-1)'*kk*pi/a);
    if kk>0 ; V = V-mean(V); end
    DCT(:,kk+1) = V/norm(V);
end

A       = DCT;
B       = DCT;
A       = normc(A);
B       = normc(B);




M=zeros(h,b,L);
C=zeros(h,b,L);
% ratio1=zeros(1,iteration);
% ratio2=zeros(1,iteration);

for itr=1:iteration

Bpinv = pinv(B);


for j1=1:L
              M(:,:,j1)=Y(:,:,j1)*Bpinv';
end

Mcat = reshape(M,[h,b*L]);
K = a;                               % number of atoms
                

param.numIteration=numIteration;              % number of iterations in DL




%%%% %%%% Using MOD algorithm for DL without mex code for sparse coding
 

%  params.data = Mcat;
%  params.Edata = errT;
%  params.dictsize = K;
%  params.iternum  = param.numIteration;
%  params.initdict = A;
%  % params.trud=A0;
%  params.memusage = 'high';
%  params.innit=3;
%  params.dsp=1;
%  params.comperrdata=0;
%  [D1,~] = MOD_DL(params,'');
 

%%%% Using ksvd algorithm for DL without mex code for sparse coding

 param.K = K;
 param.numIteration = numIteration ;
 param.errorFlag = 1;
 param.errorGoal = errT;
 param.preserveDCAtom = 0;
 param.initialDictionary = A;
 param.InitializationMethod = 'GivenMatrix';
 displayFlag = 1;
 param.displayProgress = displayFlag;
 [D1,~]  = KSVD(Mcat,param);


A=D1;
A=normc(A);


Apinv = pinv(A);

for j1=1:L
              C(:,:,j1)=Y(:,:,j1)'*Apinv';
end

Ccat = reshape(C,[w,a*L]);



 
% params.data = Ccat;
% params.Edata = errT;
% params.dictsize = b;
% params.iternum = param.numIteration;
% params.initdict=B1;
% % params.trud=B0;
% params.memusage = 'high';
% params.innit=3;
% params.dsp=1;
% params.comperrdata=0;
% [D2,g2] =MOD_DL(params,'');


param.K = K;
param.numIteration = 2 ;

param.errorFlag = 1;
param.errorGoal = errT;
param.preserveDCAtom = 0;


param.initialDictionary = B;
param.InitializationMethod = 'GivenMatrix';


displayFlag = 1;
param.displayProgress = displayFlag;


[D2,out2]  = KSVD(Ccat,param);
g2 = out2.CoefMatrix;

G2=eye(size(g2,1))*g2;
Sn = reshape(G2,[b,a,L]);

B=normc(D2); 

end

end
