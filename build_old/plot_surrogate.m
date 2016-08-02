clc
clear

load surrogate_data_o.dat
sd = surrogate_data_o;
nb_nodes = sd(1,3);
n = sqrt(length(sd(nb_nodes+3:end,1)));
x = reshape(sd(nb_nodes+3:end,1), n,n);
y = reshape(sd(nb_nodes+3:end,2), n,n);
f = reshape(sd(nb_nodes+3:end,3), n,n);
xbest = sd(2,1:2);
xtrial = sd(1,1:2);
r = sd(2,3);
nodes = sd(3:nb_nodes+3,1:2);
% c1 = reshape(sd(:,4), n,n);
% c2 = reshape(sd(:,5), n,n);

figure(1); clf;
contour(x,y,f, 200);
hold on
% contour(x,y,c1-1.0, [0 0], 'k', 'linewidth', 2);
% contour(x,y,c1-0.0, [0 0], 'k:', 'linewidth', 2);
% contour(x,y,c2-1.0, [0 0], 'k', 'linewidth', 2);
% contour(x,y,c2-0.0, [0 0], 'k:', 'linewidth', 2);
plot(xtrial(1), xtrial(2), 'r*');
for i = 1 : 1 : length(nodes(:,1))
    plot(nodes(i,1), nodes(i,2), '.', 'markersize', 20, 'color', [0 0.5 0.5]);
end
ti = 0: 0.01:2*pi+0.01;
xo = xbest;
plot(xo(1), xo(2), 'r.', 'markersize', 20);
plot(xo(1)+r*cos(ti), xo(2)+r*sin(ti), 'g-');
daspect([1 1 1])
colorbar
xlabel('x');
ylabel('y');