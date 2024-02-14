function [Zr, t] = omp2d(Y, A, A_t, C, N, k, delta)
%% 2D orthogonal matching pursuit
% Zr = omp2(Y, A, A_t, C, N, k)
% z: vectorized recovered spikes
% Y: (m x m) sample matrix
% A: (m x n) sampling matrix, A = Phi * Psi
% A_t: A'
% C: (n x n) matrix for correlations between columns of A, i.e. C(i,j) = A_t(i,:) * A(:,j)
% N: (n x n) matrix for atom norms, i.e. N(i,j) = (C(i,i)*C(j,j))^0.5
% k: sparsity level
% delta: threshold to cease the iteration

n = size(A,2);
Z0 = A_t * Y * A;       % 
R = Y;                  % residue matrix
F = ones(n);            % flag matrix
Lambda = [];            % coordinates of selected atom
g = [];                 %

if nargin==6
    for t=1:k    
        % find the most significant atom and record its coordinates
        P = abs(A_t * R * A ./N) .* F;  % projection
        [x,I] = max(P); [x,j] = max(x); i = I(j);
        Lambda(t,1:2) = [i,j];    % record the coordinates
        F(i,j) = 0;             % clear the flag of the (i,j)-th atom
        % construct H and g, and calculate the optimal vector z
        H = C(Lambda((1:t),1), Lambda((1:t),1)) .* C(Lambda((1:t),2), Lambda((1:t),2));
        g(t,1) = Z0(Lambda(t,1), Lambda(t,2));   
        z = H^(-1) * g(1:t);
        % update residue 
        R = Y;
        for i=1:t
            R =  R - z(i) * A(:, Lambda(i,1)) * A_t(Lambda(i,2), :);
        end
    end
else
    t = 0;
    while mse(R)>delta    
        t = t+1;
%         fprintf('Iteration %d\n', t);        
        % find the most significant atom and record its coordinates
        P = abs(A_t * R * A ./N) .* F;  % projection
        [x,I] = max(P); [x,j] = max(x); i = I(j);
        Lambda(t,1:2) = [i,j];    % record the coordinates
        F(i,j) = 0;             % clear the flag of the (i,j)-th atom
        % construct H and g, and calculate the optimal vector z
        H = C(Lambda((1:t),1), Lambda((1:t),1)) .* C(Lambda((1:t),2), Lambda((1:t),2));
        g(t,1) = Z0(Lambda(t,1), Lambda(t,2));   
        z = H^(-1) * g(1:t);
        % update residue 
        R = Y;
        for i=1:t
            R =  R - z(i) * A(:, Lambda(i,1)) * A_t(Lambda(i,2), :);
        end
    end    
end
Zr = sparse(Lambda(1:t,1), Lambda(1:t,2), z, n, n);