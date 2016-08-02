clear
clc

load gp_data_o.dat
sd = gp_data_o;
n = sqrt(length(sd(:,1)));
x = reshape(sd(1:end,1), n,n);
y = reshape(sd(1:end,2), n,n);
mean = reshape(sd(1:end,3), n,n);
var = reshape(sd(1:end,4), n,n);

figure(1);clf;
surf(x,y,mean);
hold on
surf(x,y, mean + 2*sqrt(var));
surf(x,y, mean - 2*sqrt(var));
% xlim([-.2, .2]);
% ylim([-.2, .2]);