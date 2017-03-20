function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
%J = 0;
%grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

pos = y==1;
neg = y==0;

size_p = size(X(pos,:));
size_n = size(X(neg,:));

h_theta1 = -ones(1, size_p(1))* log(sigmoid(X(pos,:) * theta));
h_theta2 = -ones(1, size_n(1))* log(1 - sigmoid(X(neg,:) * theta));

J = 1/m * (h_theta1 + h_theta2);

Diff = sigmoid(X * theta) - y;
grad = 1/m * (Diff' * X)';

% =============================================================

end
