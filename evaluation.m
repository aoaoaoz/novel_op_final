function score = evaluation(groundTruth,result)
%EVALUATION �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
%vecnorm((groundTruth-result)')
    score = sum(vecnorm((groundTruth-result)'));
end

