function model = m_step(X, resp, model)
% update Weights
weights = sum(resp, 1);     % 1 * K   weights = sum(resp)
model.Weights = weights / sum(weights);

K = size(weights, 2);
D = size(X, 2);

%     % update Mu
%     respX = resp' * X;      % K * D
%     [K, D] = size(respX);
%     model.Mu = bsxfun(@rdivide, respX, weights');

% update Mu & Sigma
min_cov = 1e-3;
for k = 1:K
    post = resp(:, k);
    model.Mu(k, :) = post' * X / weights(k);
    
    mu = model.Mu(k, :);
    diff = bsxfun(@minus, X, mu);
    diff = bsxfun(@times, sqrt(post), diff);
    model.Sigma(:, :, k) = diff' * diff / weights(k) + min_cov * eye(D);
end





