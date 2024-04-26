function [net,acc,err,pre] = train_SHLNN(X,Y,p,actFun,para)
%   X = d*N
%   Y = m*N
%   N is the sample number
%   m is the class number
%   p is the hidden neural node number
s = .8;

lambda = 1e-12;%mnist
% lambda = 2^-30;%fmnist
N = size(X,2);
d = size(X,1);
tic
%%%%%%%%%%%%%%%%%
% R = eye(p,N);%p*N
% WI = R/X;%p*d
%%%%%%%%%%%%%%%%%
WI = (rand(p,d)*2-1);
if p >= d
    WI = orth(WI);
else
    WI = orth(WI')';
end
%%%%%%%%%%%%%%%%%%%%%
H = WI*X;%p*N
l = max(max(H));l = s/l;
H = H * l;
HO = ActivationFunc(H,actFun,para);

WO = Y*HO'/(HO*HO'+lambda*eye(p));%m*p
%WO = Y*((HO'*HO+lambda*eye(N))\HO');
%beta = (T3'  *  T3+eye(size(T3',1)) * (C)) \ ( T3'  *  train_y);
O = WO*HO;%m*N
Training_time = toc;
%fprintf('======Classification training time: %.4f =======\n',trainingTime);
[~, label_index_expected]=max(Y);
[~,label_index_actual]=max(O);
MissClassification=length(find(label_index_actual~=label_index_expected));
acc=1-MissClassification/size(Y,2);
err = sum(sum((O-Y).^2))/N;
net.WI = WI;
net.WO = WO;
net.actFun = actFun;
net.para = para;
pre = O;
net.scale = l;
end

