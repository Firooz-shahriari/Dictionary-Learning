%(c) Simon Hawe, Lehrstuhl fuer Datenverarbeitung Technische Universitaet
%Muenchen, 2012. Contact: simon.hawe@tum.de
function para = init_learning_parameters_dict()
    para.q        = 2;      % All are important
    para.mu       = 100;    % Multiplier in log(1+mu*x^2)
    para.lambda   = 1e5;    % Lagrange multiplier
    para.kappa    = 1e6;    % Weighting for Distinctive Terms
    para.rho      = 0;      % Sparsity on Dictionaries
    para.spdw     = 0;      % Weighting in log sparsity function
    para.max_iter = 300;    % 
    % LogAbs, LogSquare, PNormAbs, PNormSquare, AtanAbs, AtanSquare
    para.Sp_type  = 'LogAbs'; 
    para.D        = cell(0);
    para.verbose  = 1;
    para.logger   = [];
    para.X        = [];
    para.sgd      = [];
end