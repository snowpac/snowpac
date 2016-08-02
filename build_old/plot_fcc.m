clc
clear

load surrogate_data_o.dat
sd = surrogate_data_o;
n = sqrt(length(sd(:,1)));
x = reshape(sd(:,1), n,n);
y = reshape(sd(:,2), n,n);
f = reshape(sd(:,3), n,n);
% c1 = reshape(sd(:,4), n,n);
% c2 = reshape(sd(:,5), n,n);

f0 = f;
for i = 1 : 1 : numel(f)
    f0(i) = myfunc([x(i), y(i)]);
end

r = 1;
nodes = 0.5+[0 0; ...
    1 0; ...
    0 1; ...
    -1 0; ...
    0 -1]*r;

figure(1); clf;
%countour(x,y,f, 50, 'linewidth', 2);
surf(x,y,f);
hold on
%contour(x,y,f0, 50, 'linewidth', 2);
surf(x,y,f0);
for i = 1 : 1 : length(nodes(:,1))
    plot(nodes(i,1), nodes(i,2), 'r*');
end
% contour(x,y,c1-1.0, [0 0], 'k', 'linewidth', 2);
% contour(x,y,c1-0.0, [0 0], 'k:', 'linewidth', 2);
% contour(x,y,c2-1.0, [0 0], 'k', 'linewidth', 2);
% contour(x,y,c2-0.0, [0 0], 'k:', 'linewidth', 2);
ti = 0: 0.01:2*pi+0.01;
xo = [0.5 0.5];
plot(xo(1), xo(2), 'r.', 'markersize', 20);
plot(xo(1)+r*cos(ti), xo(2)+r*sin(ti), 'g-');
daspect([1 1 1])
colorbar
xlabel('x');
ylabel('y');