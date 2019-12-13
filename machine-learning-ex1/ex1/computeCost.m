function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

J = (1/(2*m))*(sum((X*theta-y).^2));

% Explanation: This is the mean squared error (actually, half the MSE)
% for the line specified by the vector theta. It is calculated
% as the sum of the squares of the differences between the observed
% value (y) and the predicted value (X*theta), divided by the number
% of data points (m).
% This method of estimating/predicting unknown values is called 
% ordinary least squares in statistics.

% =========================================================================

end
