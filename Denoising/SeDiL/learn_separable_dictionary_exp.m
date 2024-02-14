% (c) Simon Hawe, Lehrstuhl fuer Datenverarbeitung Technische Universitaet
% Muenchen, 2012. Contact: simon.hawe@tum.de
% Function finds the best analysis operator D that lies on the oblique
% manifold via a riemannian conjugate gradient method. The cost function to
% be minimized is f(D) = ||Y||_p - gamma*log(det(D'*D))
% D is cell array that contains all seperated Matrices and Y is a
% Tensor of patches
function para = learn_separable_dictionary_exp(S, para, FixX)

if nargin == 2
    FixX = 0;
else
    FixX = 1;
end

S       = tensor(S);
D       = para.D;
mu 		= para.mu;
q 		= [para.q];
Sp_type = para.Sp_type;
lambda  = para.lambda;
% Linesearch parameters
alpha 	= 1e-2;
beta  	=  .9;
Qk      =   0;
Ck      =   0;
nuk     =  .0;
N       =  length(double(S));

%kappa       = para.kappa/2;

%% Initialize all required cells
TDim    = numel(D);
% Helper Variable for computing log det penalty
I       = cell(TDim,2);
% Contains the gradient
Grad    = cell(TDim+1,1);
dx      = cell(TDim+1,1);
Norm_dx = cell(TDim,1);
sel     = cell(TDim,1);
tau_dx  = cell(TDim,1);
D_c     = cell(TDim,1);

max_linesearch = 1000;

I(:,1)    = cellfun(@(D)eye(length(D)) + ones(length(D)),D,'UniformOutput',false);
I(:,2)    = cellfun(@(I)I==2,I(:,1),'UniformOutput',false);
tau_dx(:) = {0};
kappa     = cellfun(@(D)para.kappa*2/((size(D,2)^2-size(D,2))),para.D,'UniformOutput',false);
dx{end}   = 0;

if isempty(para.X)
    % n-mode notation | Matrix Vector
    % Initialize via S x1 D1' x2 D2' ... xn Dn' | D'*S
    %X = double(ttm(S,cellfun(@pinv,D,'UniformOutput',false),(1:TDim)));
    X = double(ttm(S,cellfun(@transpose,D,'UniformOutput',false),(1:TDim)));
    %X(abs(X)<mean(abs(X(:))))=0;
    sz_x = size(X);
    X = reshape(X,prod(sz_x(1:end-1)),[]);
    [~,INDX]=sort(abs(X),'descend');
    INDX  = INDX(30:end,:);
    INDX = bsxfun(@plus,INDX,(0:size(INDX,2)-1)*size(X,1));
    %X(INDX(:))=0;
    %     X = ones(size(X));
    %     %X = bsxfun(@times,X,Xm./sqrt(sum(X.^2)));
    %     X = bsxfun(@times,X,Xm./sqrt(sum(X.^2)));
    %
    %  X(abs(X)<0.2)=1e-4;
    X = reshape(X,sz_x);
    %X = randn(size(X));
    %     %X=bsxfun(@times,X,sqrt(sum(double(S).^2))./sqrt(sum(X.^2)));
    %     %X=bsxfun(@times,X,1./sqrt(sum(X.^2)));
    %     % X = double(ttm(S,cellfun(@pinv,D,'UniformOutput',false),(1:TDim)));
    %     %X( bsxfun(@le,abs(X),mean(abs(X))))=0;
    %     if ToZeros ~=0
    %         X(1:ToZeros:end)=0;
    %     end
    %X = X/norm(X(:))*q1;
else
    X = para.X;
    %[~,INDX]=sort(abs(X),'descend');
    %INDX  = INDX(end-140:end,:);
    %INDX = bsxfun(@plus,INDX,(0:size(INDX,2)-1)*size(X,1));
    %X(INDX(:))=0;
    %X=bsxfun(@times,X,1./sqrt(sum(X.^2)));
end

DX_S    = double(ttm(tensor(X),D,(1:TDim))-S);

lambda2 = 1/N;
lambda  = lambda/N * numel(double(S))/numel(double(X));

% n-mode notation | Matrix Vector
% X x1 D1 x2 D2 ... xn Dn - S | D*X - S
% n-mode notation | Matrix Vector
% 1/2||X x1 D1 x2 D2 ... xn Dn - S||_F^2 | 1/2*||D*X - S||_F^2
f0_input    = sum(double(DX_S(:)).^2);

%||X||_p^p any sparsyfing function determinde by Sp_type
if FixX
    fsp = 0;
else
    [fsp,q_w] = Sparsifying_functions(Sp_type, 'Evaluate', double(X), q, mu);
end
f0          = lambda2*f0_input + lambda*fsp;
sp_begin = fsp;
for i=1:TDim
    if kappa{i} ~= 0
        f0   =  f0 -  kappa{i}*sum(sum(log(I{i,1}-(D{i}'*D{i}).^2)));
    end
end


ob_begin = f0;
for k = 1:para.max_iter
    if k==1 || mod(k,para.verbose) == 0
        draw_atoms(D,[para.d_sz,para.d_sz],para.d_sz*2);
    end
    
    if FixX
        Grad{end} = 0;
    else
        Grad{end} = 2*lambda2*double(ttm(tensor(DX_S),D,(1:TDim),'t')) + lambda*(2*Sparsifying_functions(Sp_type, 'Derivative', double(X), q, mu, [], q_w));
    end
    if k ~= 1
        yk          = Grad{end} - g0{end};
        denom       = dx{end}(:)'*yk(:);
        nomHS       = Grad{end}(:)'*yk(:);
        nomDY       = norm(Grad{end}(:))^2;
    end
    
    for i=1:TDim
        if TDim == 1
            Grad{i}  =  2*lambda2*double(ttt(tensor(DX_S),tensor(X),2,2));
        else
            ind      = 1:(TDim);
            ind(i)   = [];
            Grad{i}  =  2*lambda2*double(ttt(tensor(DX_S),ttm(tensor(X),D(ind),ind),[ind,TDim+1],[ind,TDim+1]));
        end
        
        if kappa{i} ~= 0
            Mat         = D{i}'*D{i};
            Mat(I{i,2}) = 0;
            Grad{i}     = Grad{i}  + 4*kappa{i}*(D{i}*(Mat./(I{i,1}-Mat.^2)));
        end
        
        % Projection of the gradient onto the tangent space via
        % dO = dO - O*ddiag(O'*dO);
        Grad{i} = Grad{i} - bsxfun(@times,D{i},sum(D{i}.*Grad{i}));
        
        if numel(Grad{i}(isnan(Grad{i})))
            Grad{i}(isnan(Grad{i})) = 1e10;
        end
        
        if k ~= 1
            % Previous step transported along the step to current tangent
            % space
            tau_dx{i} = parallel_transport([], D_old{i}, dx{i}, t_prev, Norm_dx{i});
            tau_g     = parallel_transport(g0{i}, D_old{i}, dx{i}, t_prev, Norm_dx{i});
            yk        = Grad{i} - tau_g;
            denom     = denom +  tau_dx{i}(:)'*yk(:);
            nomHS     = nomHS +  Grad{i}(:)'*yk(:);
            nomDY     = nomDY +  norm(Grad{i}(:))^2;
        end
    end
    
    % CG Update direction computation
    val = 0;
    if k == 1
        cg_beta = 0;
        t_init = 0;
    else
        cg_beta     = (nomHS)/denom;
        cg_beta_dy  = (nomDY)/denom;
        cg_beta     = max(0,min(cg_beta_dy, cg_beta));
    end
    
    for i=1:TDim
        dx{i}       = -Grad{i} + cg_beta*tau_dx{i};
        val         = val + Grad{i}(:)'*dx{i}(:);
        Norm_dx{i}  = sqrt(sum(dx{i}.^2));
        if k == 1 || ~ mod(k,100)
            t_init = t_init + sum(Norm_dx{i}.^2);
            %t_init = max(t_init,max(Norm_dx{i}));
        end
        sel{i}      = Norm_dx{i} > 0;
    end
    if ~FixX
        dx{end}       = -Grad{end} + cg_beta*dx{end};
        val           = val + Grad{end}(:)'*dx{end}(:);
    end
    if k == 1 %|| ~ mod(k,100)
        %         c_check = X./dx{end};
        %         c_check = abs(median(c_check(c_check<0)));
        %         if isempty(c_check)
        %             c_check  = 1;
        %         end
        %         c_check(isinf(c_check))=1;
        %         t_init=(2*pi*max(1,round(c_check/(2*pi))))/t_init;
        if ~FixX
            t_init = 1/sqrt(t_init+sum(dx{end}(:).^2));
        else
            t_init = 1/sqrt(t_init);
        end
        t = t_init;
        fprintf('TINIT: %f\n',t_init)
    end
    
    ls_iter = 0;
    Ck      = (nuk*Qk*Ck+f0)/(nuk*Qk+1);
    Qk      = nuk*Qk+1;
    t       = t/beta;
    
    % Linesearch
    while (ls_iter == 0) || ... % Quasi Do while loop
            (f0 > Ck + alpha * t * val) && ... % Check Wolfe Condition
            (ls_iter < max_linesearch) % Check Maximum number of Iterations
        % Update the step size
        t  = t*beta;
        % Store the step length as it is required for the parallel
        % transport and the retraction
        t_prev    = t;
        
        %% Performing the linsearch
        f0         = 0;
        for i = 1:TDim
            D_c{i} = exp_mapping(D{i}, dx{i}, t, Norm_dx{i},sel{i});
            % Evalute the Objective Function at the taken step
            if  kappa{i} ~= 0
                f0         = f0 -  kappa{i}*sum(sum(log(I{i,1}-(D_c{i}'*D_c{i}).^2)));
            end
            
        end
        f_atom_sim = f0;
        if ~FixX
            % Update sparse coefficients
            X0 = X + t*dx{end};
            % Evaluate sparsity
            [fsp,q_w] = Sparsifying_functions(Sp_type, 'Evaluate', X0, q, mu);
            % Compute new difference
            DX_S = double(ttm(tensor(X0),D_c,(1:TDim))-S);
        else
            DX_S = double(ttm(tensor(X),D_c,(1:TDim))-S);
        end
        
        % Fidelity term
        f0_fid  = sum(DX_S(:).^2);
        f0      = lambda2*f0_fid + lambda*fsp  + f0;
        
        % Increase the number of linesearch iterates as we only
        % allow a certain amount of steps
        ls_iter = ls_iter + 1;
    end
    
    if ~mod(k,para.verbose)
        for i=1:TDim
            mc = sort(abs(D_c{i}'*D_c{i}));
            %fprintf('Change of D%d:\t\t %f\t\t~ Condition Number:\t\t %.2f\n',i,norm(D{i}-D_c{i},'fro'),cond(D_c{i}));
            fprintf('Mutual Coherence of D%d:\t\t %f\t\t~ Mean:\t\t %f\n',i,max(mc(end-1,:)),mean(mc(:)));
        end
        
        fprintf('Current Objective:\t %e \t~ Objective at start:\t %e\n',f0,ob_begin);
        fprintf('Current Sparsity:\t %e \t~ Input Sparsity:\t\t %e\t ~ Atom sim %e\n', 1*fsp, 1*sp_begin, f_atom_sim)
        fprintf('Current Fidelity:\t %e \t~ Input Fidelity:\t\t %e\t ~ Cg Setp %e\n',1*f0_fid, 1*f0_input,cg_beta);
        fprintf('Stepsize:\t\t %e\n',t)
        if ~FixX
            fprintf('Objective %e Numel Zero %e, %e\n', f0,numel(X(abs(X0)<1e-6)),numel(X))
        end
        fprintf('LAMBDA: %f.4\n',lambda)
    end
    end_iter = 0;
    for i=1:TDim
        if norm(D{i}-D_c{i},'fro') < 1e-10
            end_iter = end_iter + 1;
        else
            break;
        end
    end
    
    if  end_iter == TDim
        fprintf('************ NO PROGRESS **********\n');
        break;
    end
    
    if ls_iter == max_linesearch
        fprintf('************ FINISHED **********\n');
        break;
    else
        D_old   = D;
        g0      = Grad;
        if ~FixX
            X       = X0;
        end
        D       = D_c;
        t       = t/beta^2;
    end
    para.D = D;
    
    if ~mod(k,para.verbose)
        fprintf('************ NEXT ITERATION %d **********\n',k);
    end
end
para.X = X;

end





