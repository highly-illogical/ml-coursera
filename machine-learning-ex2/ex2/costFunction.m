function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
% Note: grad should have the same dimensions as theta
%


h = sigmoid(X*theta)

J = 1/m*sum((-1)*y.*log(h) - (1-y).*log(1-h))
grad = 1/m*sum((h-y).*X)'

% Theta (a vector) determines the decision boundary for the classification
% problem. It needs to be optimized such that the decision boundary it
% represents classifies the points on either side of it correctly.
%
% We apply the sigmoid function to the decision boundary function so that 
% for points which are even a little way away from it on either side, 
% the hypothesis function will yield 0 or 1 (which corresponds with the 
% training examples we have).
%
% For gradient descent, we need a cost function that does not have lots
% of local minima. The linear regression cost function will not work
% in this case. Instead, we construct a function which is defined as
% -log h when y = 1 and -log(1-h) when y = 0. This function is 0 when
% h(x) = y (the classification is correct) and infinity when |h(x)-y| = 1
% (the classification is incorrect).
%
% Thus, if more points are classified incorrectly, and farther from
% the decision boundary, this will cause the cost to go up, while 
% classifying points correctly will not add to the cost.
%
% This is a convex function so the minimum can be found by gradient descent.
% According to the course, it can be derived using maximum likelihood estimation
% but the math is beyond the scope of the course.


% =============================================================

end
