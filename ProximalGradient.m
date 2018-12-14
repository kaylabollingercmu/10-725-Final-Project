function [c] = ProximalGradient(A,v,tau,alph,MaxIt,tol)

% 10-725 Final Project, Kayla Bollinger and Landon Settle

%% Problem Description
% INPUT VALUES:
%   A = mK*N matrix
%   v = mK*1 vector
%   sigma = constrain parameter
%   tau = step size
%   alpha = scale factor for objective function
%   MaxIt = maximum number of iterations allowed
%   tol = tolerance
%
% OUTPUT VALUES:
%   cstar = min_{c}  ||c||_1 + alpha*||Ac-v||_2^2

%% Initialize Variables
n = size(A,2);
ite = 0;
c = zeros(n,1);
error = 1;

%% Main Proximal Gradient Descent Loop
while error > tol && MaxIt > ite
    ite = ite + 1;
    
    cold = c;

    gradF = (2*alph)*A'*(A*c-v);
    u = c - tau*gradF;
    
    c = sign(u).*(max(abs(u)-tau, 0)); % soft thresh
    
    error = norm(c-cold,2);    
end

end