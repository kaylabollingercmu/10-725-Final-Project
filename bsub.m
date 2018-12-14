function x = bsub(R,b)
% solve Rx = b where R is an 
% nxn upper triangular matrix

n = size(R,1);
x = zeros(n,1);

x(n) = b(n)/R(n,n);
for i = n-1:-1:1
    x(i) = (b(i) - R(i,i+1:end)*x(i+1:end))/R(i,i);
end