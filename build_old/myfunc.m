function y = myfunc ( x )

y = 1;

y = y + sum(x) + sum(x.*x);

for i = 1 : 1 : length(x)
    for j = i+1 : 1 : length(x)
        y = y + x(i)*x(j);
    end
end

end