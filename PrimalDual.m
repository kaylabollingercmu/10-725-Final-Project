function [c] = PrimalDual(A,v,sigma,tau1,tau2,theta,MaxIt,tol)

% 10-725 Final Project, Kayla Bollinger and Landon Settle

%% Problem Description
% INPUT VALUES:
%   A = mK*N matrix
%   v = mK*1 vector
%   sigma = error tolerance between AC and V
%   taui = step sizes
%   theta = relaxation parameter
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
cy = zeros(n,1); cx = zeros(n,1); cxb = zeros(n,1);
wy = zeros(m,1); wx = zeros(m,1); wxb = zeros(m,1);

ite = 0;
error = 1;

%% Main Primal-Dual Loop
while error > tol && MaxIt > ite
    ite = ite + 1;
    
    cxold = cx;
    wxold = wx;
    
    % dual step
    [p,q] = proxF2(cy/tau1 + cxb,wy/tau1 + wxb);
    cy = cy + tau1*cx - tau1*p; % applying Moreau's Identity
    wy = wy + tau1*wx - tau1*q; % applying Moreau's Identity
    
    % primal step
    [cx,wx] = proxF1(cx-tau2*cy,wx - tau2*wy);
    
    % relaxation step
    cxb = cx + theta*(cx - cxold);
    wxb = wx + theta*(wx - wxold);

    error = norm(cx-cxold) + norm(wx - wxold);
end
c = cx;

%% Functions
function [c1,w1] = proxF1(c,w)
c1 = max(0,abs(c)-tau1).*sign(c);
w1 = min(sigma/norm(w-v,2),1)*(w-v) + v;
end

function [c1,w1] = proxF2(c,w)
%c1 = bsub(L',fsub(L,c + A'*w)); % uses our code
c1 = L'\(L\(c+A'*w)); % faster
w1 = A*c1;
end

end