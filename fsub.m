function x = fsub(L,b)
% solve Lx = b where L is an 
% nxn lower triangular matrix

n = size(L,1);
x = zeros(n,1);

x(1) = b(1)/L(1,1);
for i = 2:n
    x(i) = (b(i) - L(i,1:i-1)*x(1:i-1))/L(i,i);
end