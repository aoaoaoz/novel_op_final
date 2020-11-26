%parameters
anchNum = 5; nodeNum = 200; range = 100;
tau = 0.1; Generation = 1000; N = 100;
rate = 0.1; genRate = 10;
% Generate data
%����nodeCoorֻ������ʱʹ��
[anchCoor,disMat, nodeCoor] = genData(anchNum, nodeNum, range, tau);
%LocalSearch
result = zeros(nodeNum, 2);
for i = 1:nodeNum
    func = @(x)(sum((vecnorm((anchCoor-x)')./disMat(i+anchNum, 1:5)-1).^2));
    result(i, 1:2) = fminunc(func, rand(1, 2));
end
% �ֲ��������
evaluation(nodeCoor, result)
% ��ʼ����Ⱥ
x0 = zeros(N, nodeNum, 2);
for i = 1:N
    for j = 1:nodeNum
        x0(i, j, 1) = result(j, 1) * (1 + randn * genRate);
        x0(i, j, 2) = result(j, 2) * (1 + randn * genRate);
    end
end
% ��������
for g = 1:Generation
    x1 = 
end