%% 初始化

clc;
clear;
close;

%% 数据产生

x = [ones(10, 1), rand(10, 4)];
y = 1 * x(:, 1) + 2 * x(:, 2) + 3 * x(:, 3) + 2 * x(:, 4) - 1 * x(:, 5) + normrnd(0, 1);
m = size(x, 1); % 数据数量
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
e1 = e_omega(omega, p, x, y); % 均方误差
e2 = entropy(omega); % 熵
c = var(x, y) / log(m); % 常数
fun = @(omega) e_omega(omega, p, x, y) + c * entropy(omega); % 目标函数
omega = fmincon(fun, 1/n * ones(1, n), [], [], ones(1, n), 1, zeros(1, n), ones(1, n)); % 最优化权重
p_omega = omega * p
[omega * p e_pomega(p_omega, x, y)]

%% 均方误差

function e = e_pomega(p_omega, x, y)
    m = size(x, 1);
    e = sum((y - sum(p_omega .* x, 2)).^2) / m;
end

%% 带权重的均方误差

function e = e_omega(omega, p, x, y)
    m = size(x, 1);
    e = sum((y - sum(omega * p .* x, 2)).^2) / m;
end

%% 方差

function d = var(x, y)
    m = size(x, 1);
    p = x \ y;
    d = sum((y - x * p).^2) / m;
end

%% 熵

function y = entropy(omega)
    n = size(omega, 2);
    y = 0;
    for i = 1 : n
        if 0 < omega(i)
            y = y - omega(i) * log(omega(i));
        end
    end
end