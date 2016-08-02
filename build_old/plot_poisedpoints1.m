clc
clear


points = [0,0;...
    0.9 0.01; ...
    0.9 0.005; ...
    -0.9 0.1; ...
    -0.9 -0.3];

x0 = [0,0];

xnew = [0.0102301, 0.0994754; ...
    0.0843297, -0.0537448];

index = [1 3 4];

figure(1); clf;
for i = 1 : 1 : length(points(:,1))
    plot(points(i,1), points(i,2), 'b*');
    hold on;
    if ~isempty(find(i == index,1))
        plot(points(i,1), points(i,2), 'bo');
    end    
end
for i = 1 : 1 : length(xnew(:,1))
    plot(xnew(i,1), xnew(i,2), 'ro');
end
ti = 0 : 0.01 : 2*pi+0.01;
r = .1;
plot(x0(1) + r*cos(ti), x0(2) + r*sin(ti), 'g-')

daspect([1 1 1])