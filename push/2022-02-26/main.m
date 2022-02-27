%% 初始化

clc;
clear;
close;

%% 数据处理

data = xlsread('./data.xlsx');
m = size(data, 1); % 数据数量
x = [ones(m, 1) log(data(:, 2 : 5))];
y = log(data(:, 6)); % 对变量取对数
n = 2^(size(x, 2) - 1) - 1; % 模型数量
p = zeros(n, size(x, 2)); % 模型的系数

%% 线性回归

p(1, :) = [1 1 0 0 0] .* x \ y;
p(2, :) = [1 0 1 0 0] .* x \ y;
p(3, :) = [1 0 0 1 0] .* x \ y;
p(4, :) = [1 0 0 0 1] .* x \ y;
p(5, :) = [1 1 1 0 0] .* x \ y;
p(6, :) = [1 1 0 1 0] .* x \ y;
p(7, :) = [1 1 0 0 1] .* x \ y;
p(8, :) = [1 0 1 1 0] .* x \ y;
p(9, :) = [1 0 1 0 1] .* x \ y;
p(10, :) = [1 0 0 1 1] .* x \ y;
p(11, :) = [1 1 1 1 0] .* x \ y;
p(12, :) = [1 1 1 0 1] .* x \ y;
p(13, :) = [1 1 0 1 1] .* x \ y;
p(14, :) = [1 0 1 1 1] .* x \ y;
p(15, :) = [1 1 1 1 1] .* x \ y;

%% 模型平均

omega = 1/n * ones(1, n); % 权重
e1 = e_omega(omega, p, x, y, m); % 均方误差
e2 = entropy(omega); % 熵
fun = @(omega) e_omega(omega, p, x, y, m) + entropy(omega); % 目标函数
[omega min] = fmincon(fun, 1/n * ones(1, n), [], [], ones(1, n), 1, zeros(1, n), ones(1, n)); % 最优化权重
omega
min

%% 均方误差

function e = e_omega(omega, p, x, y, m)
    e = sum((y - sum(omega * p .* x, 2)).^2) / m;
end