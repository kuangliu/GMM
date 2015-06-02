% Gaussian Mixture Models via EM algorithm
%   N = sample num
%   D = dimension
%   K = component num
%   X = [N x D]
%
%   Model: return GMM model including following parameters
%           Mu = [K x D]
%           Sigma = [D x D x K]
%           Weights = [1 x K]
%
%   Reference: https://github.com/Mrliukuang/scikit-learn/blob/master/sklearn/mixture/gmm.py
%              fitgmdist()


clear; clc; close all;

MU1 = [1 2];
SIGMA1 = [2, 0; 
          0, 0.5];

MU2 = [-3 -5];
SIGMA2 = [1, 0; 
          0, 1];

MU3 = [3 -3];
SIGMA3 = [0.7, 0; 
          0,   2.1];

X = [mvnrnd(MU1,SIGMA1,500);
     mvnrnd(MU2,SIGMA2,100);
     mvnrnd(MU3,SIGMA3,1000)];
scatter(X(:,1), X(:,2), 10, '.')
hold on

K = 3;
[N, D] = size(X);
model = [];


%% Initialization step
[~, model.Mu] = kmeans(X, K);
model.Weights = ones(1, K) ./ K;

min_covar = 1e-3;   %  Floor on the diagonal of the covariance matrix to prevent overfitting.  Defaults to 1e-3.
cv = cov(X) + min_covar * eye(D);
model.Sigma = repmat(cv, [1, 1, K]);


%% EM algorithm
n_iter = 100;
model.Converged = false;
tol = 1e-6;   % Tolerance. EM iterations will stop when average gain in log-likelihood meet this tolerance.
pre_likelihood = -Inf;

for i = 1:n_iter
    % E step: calculating responsibility in log form
    [logpx, resp] = e_step(X, model);
    likelihood = mean(logpx);
    
    % Convergence check
    delta = likelihood - pre_likelihood;
    if delta >= 0 && delta < tol * abs(likelihood)    % use relative difference
        model.Converged = true;
        break
    end
    pre_likelihood = likelihood;
    
    % M step: update params
    model = m_step(X, resp);
    
    fprintf('iter %d: p(x) = %f, delta = %f\n', i, likelihood, delta)
end

















