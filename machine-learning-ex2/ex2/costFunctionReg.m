function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
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


h = sigmoid(X*theta)

J = 1/m*sum((-1)*y.*log(h) - (1-y).*log(1-h)) + (lambda/(2*m))*sum(theta(2:size(theta)).^2)
grad = 1/m*sum((h-y).*X)' + (lambda/m)*[0; theta(2:size(theta))]

% The cost function for logistic regression with regularization has
% an extra regularization term (the lambda/2*m... term). Lambda is
% called the regularization parameter and its function here is to
% penalize theta parameters that get too large.
%
% The kind of regularization used here is called L2 regularization
% where the parameters are penalized in proportion to their squares.
% It makes sense to use this here because logistic regression is
% a generalized linear model (i.e. it depends on a linear function
% of theta and x).
%
% With regularization, the solution (decision boundary) obtained
% has higher bias but lower variance.
%
% (Note: read up on regularization and regularized logistic regression.)


% =============================================================

end
