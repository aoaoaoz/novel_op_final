function score = fitness(disMat, anchNum, anchCoor, nodeNum, candidate, alpha)
%FITNESS 此处显示有关此函数的摘要
%   此处显示详细说明
node2Node = 0;
node2Anch = 0;
for i = 1:nodeNum
    for j = i+1:nodeNum
        node2Node = node2Node + (norm( candidate(i, :)-candidate(j, :) ) / disMat(i+anchNum, j+anchNum) - 1)^2;
    end
end

for i = 1:nodeNum
    for j = 1:anchNum
        node2Anch = node2Anch + (norm( candidate(i, :)-anchCoor(j, :) ) / disMat(i+anchNum, j) - 1)^2;
    end
end
score = alpha * node2Node + (1-alpha) * node2Anch;
end

