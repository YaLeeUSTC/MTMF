function [W_t, w0, costfunc,err,reg] = train_kernel(trainx, trainy, T, dim, gamma, beta, kernel_method)


costfunc = 0;
err = 0;
reg = 0;

K = trainx'*trainx;
trainy = trainx'*trainy;    %modified 
[a, costfunct, errt, regt] = feval(kernel_method, K, trainy, 1);
W_t = a(1:T*dim);
w0 = a(T*dim+1:(T+1)*dim);

% for t = 1:T
%     % get the data for this task
%     x = trainx(: , task_indexes(t):task_indexes(t+1)-1);
%     y = trainy(task_indexes(t):task_indexes(t+1)-1);
%     K = x'*x;
%     [a, costfunct, errt, regt] = feval(kernel_method,K,y,gamma);
%     W(:,t) = x*a;
% 
%     costfunc = costfunc + costfunct;
%     err = err + errt;
%     reg = reg + regt;
% end
