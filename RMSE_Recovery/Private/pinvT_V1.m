function [BpinvT,V1] = pinvT_V1(B,tol)
    [U,E,V] = svd(B,'econ');
    E_diag = diag(E);
    ind = E_diag > tol;
    E_inv_diag = E_diag;
    E_inv_diag(ind) = 1./E_diag(ind);
    BpinvT = U*diag(E_inv_diag)*V';
    V1 = V(:,ind);
end