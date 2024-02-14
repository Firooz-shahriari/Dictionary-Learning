
function [Sn,A,B,rpA,rpB] = DL_2DDD(Y,A0,B0,iteration,numIteration,errT)

A           = randn(size(A0));
B           = randn(size(B0));
A           = normc(A);
B           = normc(B);
[h,a]       = size(A0);
[w,b]       = size(B0);
L           = size(Y,3);


% D0          = kron(B0,A0);
% Sn          = zeros(size(S));
% ctc         = 0.0001;
% tol         = 1e-10;
% M1          = M;

M           = zeros(h,b,L);
C           = zeros(h,b,L);
rpA         = zeros(1,iteration);
rpB         = zeros(1,iteration);


for itr=1:iteration

Bpinv = pinv(B);

for j1=1:L
         M(:,:,j1)  = Y(:,:,j1)*Bpinv';
end

Mcat                = reshape(M,[h,b*L]);
K                   = a;                   % number of atoms
param.numIteration  = numIteration;        % number of iterations in DL
params.data         = Mcat;
params.Edata        = errT;
params.dictsize     = K;
params.iternum      = param.numIteration;
params.initdict     = A;
params.trud         = A0;
params.memusage     = 'high';
params.innit        = 3;
params.dsp          = 1;
params.comperrdata  = 0;
[D1,~,~,~,rat1]     = MOD_DL(params,'');
rpA(itr)            = rat1(param.numIteration);
A                   = D1;
A                   = normc(A);
Apinv               = pinv(A);

for j1=1:L
          C(:,:,j1) = Y(:,:,j1)'*Apinv';
end

Ccat                = reshape(C,[w,a*L]);
params.data         = Ccat;
params.Edata        = errT;
params.dictsize     = b;
params.iternum      = param.numIteration;
params.initdict     = B;
params.trud         = B0;
params.memusage     = 'high';
params.innit        = 3;
params.dsp          = 1;
params.comperrdata  = 0;
[D2,g2,~,~,rat2]    = MOD_DL(params,'');
G2                  = eye(size(g2,1))*g2;
Sn                  = reshape(G2,[b,a,L]);
rpB(itr)            = rat2(param.numIteration);
B                   = normc(D2);

end
end


