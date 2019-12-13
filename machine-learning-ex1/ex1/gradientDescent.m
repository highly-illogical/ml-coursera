function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

	t = theta
	theta(1) = theta(1) - (alpha/m)*sum((X*t-y).*X(:,1))
	theta(2) = theta(2) - (alpha/m)*sum((X*t-y).*X(:,2))

    % Explanation: A temp variable has to be used to store theta
    % because the update step for every entry of theta requires
    % the values of all the other entries from the previous iteration.

    % Here, 1/m*(the quantity in brackets) is the partial derivative of
    % the cost function w.r.t each parameter in theta. This is the
    % theta-component of the gradient of the scalar field represented
    % by J (the component of the gradient in any direction tells us
    % how the function changes in that direction.)

    % We subtract the gradient (multiplied by the learning rate) because
    % we want to go in the direction that the function decreases. If the
    % gradient is positive, it means the function increases with theta,
    % so if we want to make the function decrease, we have to decrease
    % theta. So we want a negative quantity (and vice versa for a
    % negative gradient.)

    % Also, it makes sense for our steps to be proportional to the gradient
    % because the gradient is large at large distances from the minimum,
    % where we can take larger steps without worrying about overshooting,
    % and smaller at small distances, where we need small steps in order
    % to locate the minimum precisely. (At the minimum, the gradient is zero.)




    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
