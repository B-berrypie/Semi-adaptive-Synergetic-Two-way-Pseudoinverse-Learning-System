function net = finetunning(pilae,TrainX,trainY)
%FINETUNNING 
% Backward learning propagates the label information back to the first hidden layer and updates the weights
actFun  =  'tan';%mnist
actFun  =  'prelu';%
lambda = 1e-1;%mnist
% lambda = 2^-30;%fmnist
net = {};
layers = length(pilae);
deactH = {};
deactH{layers} = trainY;

for l = 1:layers-1
    %tempH = pinv(pilae{l}.W) * deactH{l};
    %tempH = pilae{l}.W \ deactH{l};
    
    tempH = pilae{layers-l+1}.W'/(pilae{layers-l+1}.W*pilae{layers-l+1}.W' + eye(size(pilae{layers-l+1}.W,1))*lambda)* deactH{layers-l+1};
    %tempH = (pilae{l}.W'*pilae{l}.W + eye(size(pilae{l}.W,2))*lambda)\pilae{l}.W'* deactH{l};
    deactH{layers-l} = DeactivationFunc(tempH,actFun);
    % deactH{l-1} = tempH;
end

for i = 1:layers
    net{i}.W = deactH{i} * TrainX'/(TrainX*TrainX' + eye(size(TrainX,1))*lambda);
    %net{i}.W = deactH{i} * ((TrainX'*TrainX + eye(size(TrainX,2))*lambda)\TrainX');
    if i~=layers
        TrainX = ActivationFunc(net{i}.W * TrainX,actFun);
    else
        TrainX  = net{i}.W * TrainX;
    end
end
end

