function F = fusionnet(net1,net2,method,X,actFun,para,l1,l2,scale)
%FUSION 此处显示有关此函数的摘要
%   此处显示详细说明
len1 = length(net1);
len2 = length(net2);
InputDataLayer =  X;
for i=1:l1
    tempH_test1=net1{i}.W * InputDataLayer;
    %F1 = ActivationFunc(tempH_test1,actFun,para);
    if i<len1-1
        F1 = mapminmax('apply',tempH_test1,net1{i}.nl);
    elseif i == len1-1
        tempH_test1  = tempH_test1*scale;
        F1 = ActivationFunc(tempH_test1,actFun,para);
    else
        F1 = tempH_test1;
    end
    InputDataLayer = F1;
    clear tempH_test1;
end
InputDataLayer = X;
for i=1:l2
    tempH_test2=net2{i}.W * InputDataLayer;
    if i~=len2
        tempH_test2 = ActivationFunc(tempH_test2,actFun,para);%'prelu'
    end
    F2 = tempH_test2;
    InputDataLayer = F2;
    clear tempH_test2;
end
F = [F1;F2];
end

