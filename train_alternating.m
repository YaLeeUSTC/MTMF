function [W_t, w0, D_t, costfunc] = train_alternating(trainx,trainy, gamma, beta, Dini_t, Dini_0, iterations,...
    method,kernel_method,f_method,Dmin_method)

%This code is for the paper: On Better Exploring and Exploiting Task Relationships in Multitask Learning: Joint Model and Feature Learning.
%which is based on the code https://ttic.uchicago.edu/~argyriou/code/index.html

%trainx: cell, each element is one data matrix for one specific task. 
%trainy: cell, corresponding label for the data matrix.

num_data = size(trainx{1},2);    
dim = size(trainx{1},1);     
T = size(trainx, 2);   

feat = 1; independent = 2; diagonal = 3;  %

if (max(max(abs(Dini_t-Dini_t'))) > eps)
    error('D_t should be symmetric');
end

if (min(eig(Dini_t)) < -eps)
    error('D_t should be positive semidefinite');
end

if (abs(trace(Dini_t)-1) > 200*eps)
    error('D_t should have trace  1');
end


D_t = Dini_t;    
D_0 = Dini_0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                            %
%               Feature Learning                             %
%                                                            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if (method == feat)
    costfunc = [];
    
    % Compute f(D)^{-1/2} for the next step
    [U,S,dummy] = svd(D_t);               % svd seems more robust than eig
    fS = feval(f_method,diag(S));       
    temp = sqrt(fS);                    
    tempi = find(temp > eps);           
    temp(tempi) = 1./temp(tempi);
    fDt_isqrt = U * diag(temp) * U';    
                                        
         
                                        
    %************************change the data structure of trainx£¬ trainy for memory consideration***********************
    tempY = trainy;
    TrainX = [];
    trainy = [];
    for j = 1:T
        TrainX = blkdiag(TrainX, trainx{j});
        trainy = [trainy; tempY{j}];
    end
    
    
    %******************************Main difference from MTL******************************************************************
    for iter = 1:iterations
        % Use variable transform to solve the regularization problem for
        % fixed D
        D0 = [];
        D_sqrt = [];
        I0 = eye(dim);
        for j = 1:T
            D_sqrt = blkdiag(D_sqrt, fDt_isqrt);
            D0 = [D0; I0];
        end
        M = [1/sqrt(gamma)*D_sqrt sqrt(dim/(T*beta))*D0];
        
        new_trainx = TrainX'*M;
        [W_t, w0, costf, err, reg] = train_kernel(new_trainx, trainy, T, dim, gamma, beta, kernel_method);
        W_t = D_sqrt*W_t*1/sqrt(gamma);
        W_t = reshape(W_t, dim, T);
        w0 = sqrt(dim/(T*beta))*w0;
        % W = fD_isqrt * W;        
        costfunc = [costfunc; iter, costf, err, reg];
        
        
        % Update D_t
        [U,S,V] = svd(W_t);
        if (dim > T)
            S = [S, zeros(dim,dim-T)];
        end
        Smin_t = feval(Dmin_method, diag(S));
        D_t = U * diag(Smin_t) * U';   
                
        % Compute f(D)^{-1/2} for the next step
        fS = feval(f_method, Smin_t);
        temp = sqrt(fS);
        tempi = find(temp > eps);
        temp(tempi) = 1./temp(tempi);
        fDt_isqrt = U * diag(temp) * U';
        
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                          %
%           Independent Regularizations                    %
%                                                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if (method == independent)
    [W,costfunc,err,reg] = train_kernel(trainx,trainy,task_indexes,gamma,kernel_method);
    D = [];
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                          %
%           Variable selection                             %
%                                                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if (method == diagonal)
    if (norm(D-diag(diag(D))) > eps)
        error('D should be diagonal');
    end
    costfunc = [];

    % Compute f(D)^{-1/2} for the next step
    fS = feval(f_method,diag(D));
    temp = sqrt(fS);
    tempi = find(temp > eps);
    temp(tempi) = 1./temp(tempi);
    fD_isqrt = diag(temp);
    
    for iter = 1:iterations
        new_trainx = fD_isqrt * trainx;
        [W,costf,err,reg] = train_kernel(new_trainx,trainy,task_indexes,gamma,kernel_method);
        W = fD_isqrt * W;
        
        costfunc = [costfunc; iter, costf, err, reg];

        % Update D
        Smin = feval(Dmin_method, sqrt(sum(W.^2,2)));
        D = diag(Smin);
        
        % Compute f(D)^{-1/2} for the next step
        fS = feval(f_method,Smin);
        temp = sqrt(fS);
        tempi = find(temp > eps);
        temp(tempi) = 1./temp(tempi);
        fD_isqrt = diag(temp);
    end
end


