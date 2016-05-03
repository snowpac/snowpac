clc
clear

load points.dat

figure(1); clf;
plot(points(end-2,1), points(end-2,2), 'r*');    
hold on
plot(points(end-3,1), points(end-3,2), 'g*');    
for i = 1 : 1 : length(points(:,1))-4
    plot(points(i,1), points(i,2), 'b*');    
end
plot(points(end-1, 1), points(end-1,2), 'bo')
ti = 0 : 0.01 : 2*pi+0.01;
r = points(end,1);
plot(points(end-1, 1) + r*cos(ti), points(end-1,2) + r*sin(ti), 'g-')

daspect([1 1 1])