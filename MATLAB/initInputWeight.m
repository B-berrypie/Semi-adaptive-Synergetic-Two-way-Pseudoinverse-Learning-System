function IW = initInputWeight(TrainX,trainY,hn,method,actFun)
%INITINPUTWEIGHT 
% Initialize the weight for the encoder, where hn is the number of hidden layer neurons
% Methods ae was traditional pilae and l2 regularization was used
c = size(trainY,1);

lambda = 1e-3;%mnist
% lambda = 1e-11;%fmnist
switch lower(method)
    case {'ae'}
        W = rand(size(TrainX,1),hn)*2-1;
        %expctH = pinv(W)*TrainX;
        %expctH = W\TrainX;
        expctH = W'/(W*W' + eye(size(W,1))*lambda)*TrainX;
        %expctH = TrainX'* W/(W'*W+ eye(size(W,2))*lambda);
        %de_expctH = DeactivationFunc(expctH,actFun);
        de_expctH = expctH;
        %IW = de_expctH*pinv(TrainX);
        IW = de_expctH*TrainX'/(TrainX*TrainX' + eye(size(TrainX,1))*lambda);
        %IW = de_expctH/TrainX;
    case {'sa'}
        W = 2*rand(hn,size(TrainX,1))-1;
        H = W * TrainX;
        H = mapminmax(H')';
        IW  =  calculateWeights4AE(H',TrainX',lambda,50);
end
end

