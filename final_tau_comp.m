clear, close all;
%parameters
anchNum = 5; nodeNum = 40; range = 100;
tau = 0.1; Generation = 600; N = 80;
mutationRate = 0.2; initRate = tau/2; F = 0.3;
alpha = 0.1;Count = 10;
taus = [0.05,0.1,0.15,0.2,0.3];
bestResurt = zeros(length(taus), Count);
options = optimset('Display','off');
for tau_id = 1:length(taus)
    tau = taus(tau_id);
    disp(tau_id);
    for time = 1:Count
        % Generate data
        %其中nodeCoor只在评价时使用
        [anchCoor,disMat, nodeCoor] = genData(anchNum, nodeNum, range, tau);
        %LocalSearch
        result = zeros(nodeNum, 2);
        for i = 1:nodeNum
            func = @(x)(sum((vecnorm((anchCoor-x)')./disMat(i+anchNum, 1:5)-1).^2));
            result(i, 1:2) = fminunc(func, rand(1, 2), options);
        end
        result = min(100, max(0, result));
        % 初始化种群
        x0 = zeros(N, nodeNum, 2);
        for i = 1:N
            for j = 1:nodeNum
                x0(i, j, 1) = result(j, 1) * (1 + randn * (randi(2)*2-3) * initRate);
                x0(i, j, 2) = result(j, 2) * (1 + randn * (randi(2)*2-3) * initRate);
            end
        end
        x0 = min(100, max(0, x0));
        % 迭代更新(经典差分算法)
        Indices = zeros(1, Generation);

        best = 1;
        for i = 2:N
            if fitness(disMat, anchNum, anchCoor, nodeNum, squeeze(x0(best, :, :)), alpha) > fitness(disMat, anchNum, anchCoor, nodeNum, squeeze(x0(i, :, :)), alpha)
                best = i;
            end
        end
        for g = 1:Generation
            % 下一代
            x1 = x0;
            % 下一代最优适应度值
            mi = Inf;
            scoreSum = 0;
            nextBestInd = 1;
            nextBestScore = Inf;
            for i = 1:N
                % 突变
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
                % 选择
                curScore = fitness(disMat, anchNum, anchCoor, nodeNum, u, alpha);
                if curScore < fitness(disMat, anchNum, anchCoor, nodeNum, squeeze(x0(i, :, :)), alpha)
                    x1(i, :, :) = u;
                end
                if curScore < nextBestScore
                    nextBestScore = curScore;
                    nextBestInd = i;
                end
                % 评估
                score = evaluation(nodeCoor, squeeze(x1(i, :, :)));
                mi = min(mi, score);
                scoreSum = scoreSum + score;
            end
            Indices(g) = mi;
            x0 = x1;
            best = nextBestInd;
        end
        X = 1:Generation;
        %绘制最终结果
        ind = 1;
        for i = 2:N
            if evaluation(nodeCoor, squeeze(x0(ind, :, :))) > evaluation(nodeCoor, squeeze(x0(i, :, :)))
                ind = i;
            end
        end
        bestResurt(tau_id, time) = evaluation(nodeCoor, squeeze(x0(ind, :, :)));
    end
end
for i = 1:length(taus)
    mean(bestResurt(i, :))
    std(bestResurt(i, :))
end