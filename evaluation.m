function score = evaluation(groundTruth,result)
%EVALUATION 此处显示有关此函数的摘要
%   此处显示详细说明
%vecnorm((groundTruth-result)')
    score = sum(vecnorm((groundTruth-result)'));
end

