%% 初始化

clc;
clear;
close;

%% 输入与预处理

r = [0 3 3 5 7 5; 
    3 0 5 5 5 13; 
    3 5 0 3 3 5
    5 5 3 0 7 7
    7 5 3 7 0 7
    5 13 5 7 7 0]; % 输入
n = size(r, 1); % 数据量
r = (r - 3) ./ (19 - 3); % 标准化
for i = 1 : n
    r(i, i) = 1; % 对角线上为1
end

%% 平方法合成传递闭包

newr = zeros(n, n);
for k = 1 : log2(n) + 1 % 至多需要计算log2(n) + 1次
    for i = 1 : n
        for j = 1 : n
            newr(i, j) = max(min(r(i, :), r(:, j)')); % 模糊关系矩阵的乘法
        end
    end
    r = newr; % 更新关系矩阵
end

%% 求lambda-截矩阵

list = unique(sort(r)); % 不同的lambda的取值
lambda = list(2); % 可以调整lambda值
R = zeros(n, n);
for i = 1 : n
    for j = 1 : n
        if r(i, j) >= lambda
            R(i, j) = 1;
        else
            R(i, j) = 0;
        end
    end
end

%% 利用以上结果聚类
    
g = graph(R, 'omitselfloops');
plot(g);