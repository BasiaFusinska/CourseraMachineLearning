function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%




pos = y==1;
neg = y==0;

size_p = size(X(pos,:));
size_n = size(X(neg,:));

h_theta1 = -ones(1, size_p(1))* log(sigmoid(X(pos,:) * theta));
h_theta2 = -ones(1, size_n(1))* log(1 - sigmoid(X(neg,:) * theta));

theta_r = theta;
theta_r(1) = 0;
regularized = lambda/2 * (theta_r'*theta_r);

J = 1/m * ((h_theta1 + h_theta2) + regularized);

Diff = sigmoid(X * theta) - y;
grad = 1/m * ((Diff' * X)' + lambda*theta_r);


% =============================================================

grad = grad(:);

end
