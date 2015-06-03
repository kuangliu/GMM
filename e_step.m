function [logpx, resp] = e_step(X, model)
% Expectation step: calculating responsibility by log form
%   p(k|x) = p(x|k)p(k)/p(x)
%   => logp(k|x) = logp(x|k) + logp(k) - logp(x)

pxk = log_gaussian_density(X, model);   % logp(x|k) = logN(x|mu,sigma)
lpr = bsxfun(@plus, pxk, log(model.Weights));    % lpr = logp(x|k) + logp(k)
%% 1. My method works as the same
logpx = log(sum(exp(lpr), 2));          % logp(x) = logsum(p(x|k)p(k)) = logsumexp(lpr)
resp = bsxfun(@minus, lpr, logpx);      % responsibility = logp(k|x) = logp(x|k)+logp(k)-logp(x)
resp = exp(resp);                       % calculate p(k|x)

%% 2. Method from Matlab2015
% maxll = max(lpr, [], 2);
% % minus maxll to avoid underflow
% resp = exp(bsxfun(@minus, lpr, maxll));
% density = sum(resp, 2);
% % normalize posteriors
% resp = bsxfun(@rdivide, resp, density);
% logpx = log(density) + maxll;


function log_prob = log_gaussian_density(X, model)
% Log probability for full covariance matrices.
% logN(x|mu,sigma) = -0.5*(x-mu)*sigma^-1*(x-mu) -0.5*D*log(2*pi) - 0.5*logdet(Sigma)
% actually is log density instead of log probility.

Mu = model.Mu;
Sigma = model.Sigma;

[N, D]=size(X);
K = size(Mu, 1);

log_prob = zeros(N, K);
min_cov = 1e-7;
for k = 1:K
    % logDetSigma = log(det(Sigma(:,:,k)));
    % Maybe "Cholesky decomposition" is more stable.
    % Try 'cholcov' specially for covariance matrix.
    L = chol(Sigma(:,:,k) + min_cov*eye(D));     % sigma = L' * L
    diagL = diag(L);
    log_det_sigma = 2*sum(log(diagL));
    
    X_mu = bsxfun(@minus, X, Mu(k,:));    % x-mu
    X_mu_l = X_mu / L;   % (x-mu)*sigma^-1*(x-mu) = ((x-mu)/L)^2
    quadform = sum(X_mu_l.^2, 2);
    
    log_prob(:, k) = -0.5 * (quadform + D*log(2*pi) + log_det_sigma);
end


