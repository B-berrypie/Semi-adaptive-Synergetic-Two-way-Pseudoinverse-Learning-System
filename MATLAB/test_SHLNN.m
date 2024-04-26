function [pre,acc,err]=test_SHLNN(net,X,Y,s)
%TEST_SHLNN Summary of this function goes here
%   X = d*N
%   Y = m*N
%   N is the sample number
%   m is the class number
TOP = 1;
N = size(X,2);
H = net.WI*X;%p*N
H = H * net.scale;
HO = ActivationFunc(H,net.actFun,net.para);
O = net.WO*HO;%m*N
[~, label_index_expected]=max(Y);
if TOP == 1
    [~,label_index_actual]=max(O);
    MissClassification=length(find(label_index_actual~=label_index_expected));
else
    MissClassification = 0;
    for i=1:size(Y,2)
        [~,pos] = sort(O,'descend');
        if ~ismember(label_index_expected(i),pos(1:TOP,i))
            MissClassification = MissClassification + 1;
        end
    end
end
acc=1-MissClassification/size(Y,2);
err = sum(sum((O-Y).^2))/N;
pre = O;
end

