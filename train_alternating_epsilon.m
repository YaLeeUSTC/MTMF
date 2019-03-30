function [W_t, W_0, D_t, D_0,costfunc,mineps] = train_alternating_epsilon(trainx,trainy,task_indexes,gamma,Dini_t, Dini_0, iterations,...
    method,kernel_method,f_method,Dmin_method,epsilon_init)

if (epsilon_init < eps)
    % Run without epsilon
    [W_t, W_0, D_t, D_0, costfunc] = train_alternating(trainx,trainy,task_indexes,gamma,Dini_t, Dini_0, iterations,...
        method,kernel_method,f_method,Dmin_method);
    mineps = 0;
    return;
end

mincost = inf;
epsilon = epsilon_init;

i = 1;
while (epsilon > eps)
    Dmin_e_method = @(b) (feval(Dmin_method, sqrt(b.^2+epsilon)));
    [We,De,costfunc_e] = train_alternating(trainx,trainy,task_indexes,gamma,Dini_t, Dini_0, iterations,...
        method,kernel_method,f_method,Dmin_e_method);
    s = svd(De);
    costfunc_e(:,[2,4]) = costfunc_e(:,[2,4]) + gamma * epsilon * sum(feval(f_method,s));

    curcost = costfunc_e(size(costfunc_e,1),2);
    if (curcost < mincost)
        mincost = curcost;
        mineps = epsilon;
        W = We;
        D = De;
    end
    costfunc{i} = costfunc_e;
    i = i+1;
    epsilon = epsilon / 10;
end
