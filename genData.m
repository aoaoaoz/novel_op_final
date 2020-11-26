function [anchCoor,disMat, nodeCoor] = genData(anchNum, nodeNum, range, tau)
%GENDATA 生成数据,anchNum固定为5
%   此处显示详细说明
anchCoor = [0.2*range, 0.2*range; 0.8*range, 0.2*range; 0.2*range, 0.8*range; 0.8*range, 0.8*range; 0.5*range, 0.5*range];
disMat = zeros(anchNum + nodeNum, anchNum + nodeNum);
nodeCoor = rand(nodeNum, 2) * range;
totCoor = [anchCoor; nodeCoor];
for i = 1:anchNum + nodeNum
    for j = i+1:anchNum + nodeNum
        dij = sqrt((totCoor(i, 1)-totCoor(j, 1))^2 + (totCoor(i, 2)-totCoor(j, 2))^2);
        disMat(j,i) = dij * (1+randn*tau);
        disMat(i,j) = disMat(j,i);
    end
end
end