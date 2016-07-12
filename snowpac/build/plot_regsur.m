clc
clear

load val_data.dat

load surrogate_data_o.dat
sd = surrogate_data_o;
n = sqrt(length(sd(:,1)));
x = reshape(sd(1:end,1), n,n);
y = reshape(sd(1:end,2), n,n);
f{1} = reshape(sd(1:end,3), n,n);
fv{1} = reshape(sd(1:end,6), n,n);
f{2} = reshape(sd(1:end,4), n,n);
fv{2} = reshape(sd(1:end,7), n,n);
f{3} = reshape(sd(1:end,5), n,n);
fv{3} = reshape(sd(1:end,8), n,n);

k=1;

figure(1); clf;
contour(x,y,f{1}, 200);
hold on
contour(x,y,f{2}, -1:0.1:0, 'k', 'linewidth', 2);
contour(x,y,f{3}, -1:0.1:0, 'k', 'linewidth', 2);
% surf(x,y,f{k} + fv{k})
% contour(x,y,c1-1.0, [0 0], 'k', 'linewidth', 2);
% contour(x,y,c1-0.0, [0 0], 'k:', 'linewidth', 2);
% contour(x,y,c2-1.0, [0 0], 'k', 'linewidth', 2);
% contour(x,y,c2-0.0, [0 0], 'k:', 'linewidth', 2);
% ti = 0: 0.01:2*pi+0.01;
% xo = [0.5 0.5];
% r = sqrt(0.5);
% plot(xo(1), xo(2), 'r.', 'markersize', 20);
% plot(xo(1)+r*cos(ti), xo(2)+r*sin(ti), 'g-');
% daspect([1 1 1])
% colorbar
% xlabel('x');
% ylabel('y');

% hold on
% for i = 1 : 1 : length(val_data(:,1))
%     i
%     plot3(val_data(i,1), val_data(i,2), val_data(i,5+k), 'r*');
% end
