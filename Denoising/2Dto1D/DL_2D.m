
function [Sn,A,B] = DL_2D(Y,iteration,numIteration,sparsity)

[h,w,L] = size(Y);
a       = 2*h;   
b       = 2*w;

DCT     = zeros(h,a);
for kk  = 0:1:a-1
    V   = cos((0:1:size(Y,1)-1)'*kk*pi/a);
    if kk>0 ; V = V-mean(V); end
    DCT(:,kk+1) = V/norm(V);
end

A           = DCT;
B           = DCT;
A           = normc(A);
B           = normc(B);
M           = zeros(h,b,L);
C           = zeros(h,b,L);

for itr=1:iteration

Bpinv = pinv(B);

for j1=1:L
         M(:,:,j1)  = Y(:,:,j1)*Bpinv';
end

Mcat                = reshape(M,[h,b*L]);
K                   = a;                   % number of atoms
param.numIteration  = numIteration;        % number of iterations in DL
params.data         = Mcat;
% params.Edata        = errTM;
params.Tdata        = sparsity;
params.dictsize     = K;
params.iternum      = param.numIteration;
params.initdict     = A;
% params.trud         = A;
params.memusage     = 'high';
params.innit        = 3;
params.dsp          = 1;
params.comperrdata  = 0;
[D1,g1,~]            = MOD_DL(params,'');
A                   = D1;
A                   = normc(A);
Apinv               = pinv(A);

for j1=1:L
          C(:,:,j1) = Y(:,:,j1)'*Apinv';
end

Ccat                = reshape(C,[w,a*L]);
params.data         = Ccat;
% params.Edata        = errTC;
 params.Tdata        = sparsity;
params.dictsize     = b;
params.iternum      = param.numIteration;
params.initdict     = B;
% params.trud         = B;
params.memusage     = 'high';
params.innit        = 3;
params.dsp          = 1;
params.comperrdata  = 0;
[D2,g2,~]           = MOD_DL(params,'');
G2                  = eye(size(g2,1))*g2;
Sn                  = reshape(G2,[b,a,L]);
B                   = normc(D2);

fprintf('Iteraton:  %d\n',itr)

end
end


