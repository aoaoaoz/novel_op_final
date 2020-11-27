clear, close all;
%parameters
anchNum = 5; nodeNum = 500; range = 100;
tau = 0.1; Generation = 3000; N = 200;
mutationRate = 0.1; initRate = tau/2; F = 0.3;
alpha = 0.1;
% Generate data
%����nodeCoorֻ������ʱʹ��
[anchCoor,disMat, nodeCoor] = genData(anchNum, nodeNum, range, tau);
% load("anchCoor1126.mat");
% load("disMat1126.mat");
% load("nodeCoor1126.mat");
figure(1);title('��ʵֵ');
scatter(anchCoor(:, 1), anchCoor(:, 2), '*');hold on;grid on;
scatter(nodeCoor(:, 1), nodeCoor(:, 2), '+');
%LocalSearch
result = zeros(nodeNum, 2);
for i = 1:nodeNum
    func = @(x)(sum((vecnorm((anchCoor-x)')./disMat(i+anchNum, 1:5)-1).^2));
    result(i, 1:2) = fminunc(func, rand(1, 2));
end
result = min(100, max(0, result));
% �ֲ��������
figure(2);title('�ֲ��������');
scatter(anchCoor(:, 1), anchCoor(:, 2), '*');hold on;grid on;
scatter(result(:, 1), result(:, 2), '+');
evaluation(nodeCoor, result)
% ��ʼ����Ⱥ
x0 = zeros(N, nodeNum, 2);
for i = 1:N
    for j = 1:nodeNum
        x0(i, j, 1) = result(j, 1) * (1 + randn * (randi(2)*2-3) * initRate);
        x0(i, j, 2) = result(j, 2) * (1 + randn * (randi(2)*2-3) * initRate);
    end
end
x0 = min(100, max(0, x0));
ind = 1;
for i = 2:N
    if evaluation(nodeCoor, squeeze(x0(ind, :, :))) > evaluation(nodeCoor, squeeze(x0(i, :, :)))
        ind = i;
    end
end
figure(3);title('��ʼ��');
scatter(anchCoor(:, 1), anchCoor(:, 2), '*');hold on;grid on;
scatter(x0(ind, :, 1), x0(ind, :, 2), '+');

% ��������(�������㷨)
Indices = zeros(1, Generation);

best = 1;
for i = 2:N
    if fitness(disMat, anchNum, anchCoor, nodeNum, squeeze(x0(best, :, :)), alpha) > fitness(disMat, anchNum, anchCoor, nodeNum, squeeze(x0(i, :, :)), alpha)
        best = i;
    end
end
for g = 1:Generation
    if mod(g, 10) == 0
        disp(g)
    end
    % ��һ��
    x1 = x0;
    % ��һ��������Ӧ��ֵ
    mi = Inf;
    scoreSum = 0;
    nextBestInd = 1;
    nextBestScore = Inf;
    for i = 1:N
        % ͻ��
        r1 = randi(N);r2 = randi(N);r3 = randi(N);
        %v = squeeze( x0(r1, :, :) + F * ( x0(r2, :, :) - x0(r3, :, :) ) );
        v = squeeze( x0(best, :, :) + F * ( x0(r2, :, :) - x0(r3, :, :) ) );
        v = min(100, max(v, 0));
        u = squeeze( x0(i, :, :) );
        Jr1 = randi(nodeNum);
        Jr2 = randi(2);
        for j = 1:nodeNum
            rcj = rand;
            if rcj <= mutationRate || (Jr1 == j && Jr2 == 1)
                u(j, 1) = v(j, 1);
            end
            rcj = rand;
            if rcj <= mutationRate || (Jr1 == j && Jr2 == 2)
                u(j, 2) = v(j, 2);
            end
        end
        % ѡ��
        curScore = fitness(disMat, anchNum, anchCoor, nodeNum, u, alpha);
        if curScore < fitness(disMat, anchNum, anchCoor, nodeNum, squeeze(x0(i, :, :)), alpha)
            x1(i, :, :) = u;
        end
        if curScore < nextBestScore
            nextBestScore = curScore;
            nextBestInd = i;
        end
        % ����
        score = evaluation(nodeCoor, squeeze(x1(i, :, :)));
        mi = min(mi, score);
        scoreSum = scoreSum + score;
    end
    Indices(g) = mi;
    x0 = x1;
    best = nextBestInd;
end
X = 1:Generation;
%�������ս��
ind = 1;
for i = 2:N
    if evaluation(nodeCoor, squeeze(x0(ind, :, :))) > evaluation(nodeCoor, squeeze(x0(i, :, :)))
        ind = i;
    end
end
figure(4);title('���ս��');
scatter(anchCoor(:, 1), anchCoor(:, 2), '*');hold on;grid on;
scatter(x0(ind, :, 1), x0(ind, :, 2), '+');

figure(5);title('ָ��');
plot(1:Generation, Indices);