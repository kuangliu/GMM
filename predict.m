function C = predict(X, model)
%% Predict label for data.
%    MODEL: GMM model
%    X:     [N, D]

%  returns: 
%    C: cluster label [N, 1]

[~, resp] = e_step(X, model);
[~, C] = max(resp, [], 2);

