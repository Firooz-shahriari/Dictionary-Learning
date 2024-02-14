%%%%                      IN THE NAME OF GOD                           %%%%

function [X,D,RMSE] = MOD(Y,num_atom,iter,sparsity,D0)

% ind  = randperm(size(Y,2));
% D0   = normc(Y(:,ind(1:num_atom)));
% D0   = normc(randn(size(Y,1),num_atom));
X    = zeros(size(D0,2),size(Y,2));
D    = zeros(size(Y,1),num_atom);
RMSE = zeros(1,iter);
lam  = 0.005;

for it = 1:iter
    parfor i=1:size(X,2)
       X(:,i) = myOMP(Y(:,i),D0,sparsity);
    end
    
    RMSE(it)  = norm(Y-D0*X,'fro')/sqrt(numel(Y));
    tmp       = X*X';
    sss       = size(tmp,1);
    D0        = (Y*X')/(lam*eye(sss)+(tmp));
    D0        = normc(D0);
    D(:,:,it) = D0;
end
end