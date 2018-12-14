%% Douglas Rachford
function [c] = DouglasRachford(A,v,sigma,tau,lambda,MaxIt,tol)

% 10-725 Final Project, Kayla Bollinger and Landon Settle

%% Problem Description
% INPUT VALUES:
%   A = mK*N matrix
%   v = mK*1 vector
%   sigma = error tolerance between AC and V
%   tau = step sizes
%   lambda = algorithm parameter
%   MaxIt = maximum number of iterations allowed
%   tol = tolerance
%
% OTHER VALUES:
% L = Cholesky decomp of I + A^T A
%
% OUTPUT VALUES:
%   (cstar,~) = min_{c,w}  ||c||_1 + I_{B_sigma(v)}(w) + I_k(c,w)

%% Initialize Variables
[m,n] = size(A);
L = chol(eye(n) + A'*A,'lower');
c = zeros(n,1); c1 = zeros(n,1);
w = zeros(m,1); w1 = zeros(m,1);
lambda2 = lambda/2;
lambda2f1 = 1-lambda2;

ite = 0;
error = 1;

%% Main Douglas-Rachford Loops
while error > tol && MaxIt > ite
    ite = ite + 1;
    
    cold = c1;
    
    [p,q] = rproxF1(c1,w1);
    [p,q] = rproxF2(p,q);
    c1 = lambda2f1*c1 + lambda2*p;
    w1 = lambda2f1*w1 + lambda2*q;

    error = norm(c1-cold);
end
[c,~] = proxF1(c1,w1);

%% Functions
function [c1,w1] = proxF1(c,w)
c1 = max(0,abs(c)-tau).*sign(c);
w1 = min(sigma/norm(w-v,2),1)*(w-v) + v;
end

function [c1,w1] = proxF2(c,w)
%c1 = bsub(L',fsub(L,c + A'*w)); % uses our code
c1 = L'\(L\(c+A'*w)); % faster
w1 = A*c1;
end

function [c1,w1] = rproxF1(c,w)
[c1,w1] = proxF1(c,w);
c1 = 2*c1 - c;w1 = 2*w1 - w;
end

function [c1,w1] = rproxF2(c,w)
[c1,w1] = proxF2(c,w);
c1 = 2*c1 - c;w1 = 2*w1 - w;
end

end