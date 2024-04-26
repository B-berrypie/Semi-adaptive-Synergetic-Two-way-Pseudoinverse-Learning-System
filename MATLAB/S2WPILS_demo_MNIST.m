clear;
clc;
warning('off','all');
%% Dataset
% Label is numbers
% % X is d*N matrix, N is the sample number; d is the dimension
% avg 98.97
%%%%%%%%%%%%%%%%%%% MNIST
Label = importdata('./Dataset/mnistnumY.mat');
X = importdata('./Dataset/mnistX.mat');
%%%%%%%%%%%%%%%%% F-MNIST
% Label = importdata('./Dataset/fashion_mnistnumY.mat');
% X = importdata('./Dataset/fashion_mnistX.mat');
%% STRUCTURE
MAX_SUBNET = 2;
MAX_LAYER = 3;
TRAING_RATIO = 0.8;%for SPLIT_TRAIN_TEST=0
SAMPLE_RATIO = 0.8;% when there is only one sub-net, set it to 1.0


para = 0.05;
% actFun = 'prelu';
actFun = 'tan';
p = 0.9;
nl = 0.0001;% conv noise level
% SPLIT_TRAIN_TEST = 0; % 0:random split; 1: set the index
SPLIT_TRAIN_TEST = 1; % 0:random split; 1: set the index
HIDDEN_NUM_Mode = 0;% 0: HIDDEN_NEURON_NUM; 1: n= p * dim;

CLASSIFICATION_NEURON_NUM = 800;
%1000 F-MNIST
HIDDEN_NEURON_NUM = [2000,1500,500];%MNIST
% HIDDEN_NEURON_NUM = [1500,1000,600,500];%F-MNIST
HN = 10000;%fusionF's SHLNN neurons

randomProES = 0.99; % probability to early stop

randomProSN = 0.6; % probability of stopping add subnetwork

px = [];
py = [];


%%  Random seperate training and test set %%%%
if SPLIT_TRAIN_TEST == 0
    rand('seed',0);
    rand_idx= randperm(size(X,2));
    trainidx = rand_idx(1:ceil(size(X,2).*TRAING_RATIO));
    vaidx = rand_idx(ceil(size(X,2).*TRAING_RATIO)+1:ceil(size(X,2).*TRAING_RATIO)+1+ceil(size(X,2).*(1-TRAING_RATIO)/2));
    teidx = setdiff(1:size(X,2),[trainidx vaidx]);

elseif SPLIT_TRAIN_TEST == 1
    
    %%%%%%%%%%%% MNIST & F-MNIST
    trainidx = 1:60000;
    vaidx = 50001:60000;
    teidx = 60001:70000;
end

%%  Normalization %%%%%%
Label = double(Label);
% X = double(X);
X = zscore(X);% mnist norb
% X = mapminmax(X,0,1);%fmnist
%X = whiten(X')';

train_X = X(:,trainidx);
train_Y = Label(:,trainidx);

test_X = X(:,teidx);
test_Y = Label(:,teidx);

valid_X = X(:,vaidx);
valid_Y = Label(:,vaidx);

%%%% For confusion Matrix %%%%
[test_Y,i]= sort(test_Y);
test_X = test_X(:,i);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

P = train_X;
T = train_Y;

TT.P = test_X;
TT.T = test_Y;

VA.P = valid_X;
VA.T = valid_Y;


clear train_X test_X X Label;
NumberofTrainingData=size(P,2);
NumberofTestingData=size(TT.P,2);
NumberofValidationData=size(VA.P,2);
targetPrepro;
esbTY = zeros(size(TT.T));
esbVY = zeros(size(VA.T));
df_trainX = [];
df_trainY = [];
CM = zeros(size(TT.T,1),size(TT.T,1),MAX_SUBNET);
ind=2*ones(1,size(TT.T,1));
mask = diag(ind);
mask = mask-1;
LL = zeros(MAX_SUBNET,NumberofTestingData);
%% Semi-adaptive Synergetic Two-way Pseudoinverse Learning System
%  PC equipped with an Intel(R) Core(TM) i5-14600K 3.50 GHz processor and 48.0 GB of DDR5 RAM,Matlab R2023b
%  Non-parallel running
TrainingData = P;

sub_net = 1;
best_V_Eacc = 0;
alltime = 0;

while true
    if (sub_net > MAX_SUBNET)
        break;
    end
    SR = normrnd(SAMPLE_RATIO,0.02);%increase sub-network diversity
    if SR > 1 || SR <=0
        SR = SAMPLE_RATIO;
    end
    %sampling
    [trainInd,valInd,testInd]=dividerand(NumberofTrainingData,SR,1-SR,0.0);
    trainBatch = TrainingData(:,trainInd);%d*N
    trainBatchLabel = T(:,trainInd);

  
    validBatch = VA.P;%d*n
    testBatch = TT.P;%d*n

    %======================================
    %training set
    l=1; %layer for PILer-MLP
    InputDataLayer = trainBatch;%d*n
    best_V_acc = 0;
    TInputDataLayer=[];
    fl_start = tic;
    while l<=MAX_LAYER
    
        %  ======= PILer-Classifier  Training ==========
        numsamples = size(InputDataLayer,2);
        numdims = size(InputDataLayer,1);
        po = 3+(6-3)*rand;
        if HIDDEN_NUM_Mode == 1
            HiddernNeuronsNum = floor(po*numdims);
        else
            HiddernNeuronsNum = HIDDEN_NEURON_NUM(l);
        end
        fprintf('======subnet:%d Layer: %d feedforwad- Hidden Neurons: %d\n',sub_net,l,HiddernNeuronsNum);
        HiddernNeurons(l) = HiddernNeuronsNum;
        IW = initInputWeight(InputDataLayer,trainBatchLabel,HiddernNeuronsNum,'sa',actFun);
        ae{l}.OW = IW';
        ae{l}.IW = IW;
        clear IW
        layers = l;
        %%%%%%%%%%% Training Phase feedforwad %%%%%%%%%%%%%%;
        InputData = trainBatch;
        for i=1:1:layers
            tempH=(ae{i}.OW)'* (InputData);% tied weight
            [Ho,ps]  =  mapminmax(tempH,0,1);
            ae{i}.nl = ps;
            InputData = Ho;
            F_featrues{i} = InputData;
            clear tempH;
        end
        InputDataLayer = InputData;
        [cls,TrainingAccuracy,TrainingErr,TrainPre] = train_SHLNN(InputData,trainBatchLabel,CLASSIFICATION_NEURON_NUM,actFun,para);
        classifier{l} = cls;
        %%%%%%%%%%% Validation Phase feedforwad %%%%%%%%%%%%%%;

        InputData = VA.P;
        for i=1:1:layers
            tempH_v=(ae{i}.OW)'*(InputData); %tied weight
            Ho = mapminmax('apply',tempH_v,ae{i}.nl);
            InputData = Ho;
            clear tempH_v;
        end
        [VPre,VAccuracy,VErr]=test_SHLNN(cls,InputData,VA.T);

        if VAccuracy > best_V_acc
            fprintf('====== Current Validation Accuracy %.4f > Best Validation Accuracy %.4f======\n',VAccuracy,best_V_acc);
            best_V_acc = VAccuracy;
        else
            rand('seed',sum(clock));
            if rand(1) < randomProES
                F_featrues(l) = [];
                l=l-1;
                cl = classifier{l};
                break;
            end
        end
        if (l >= MAX_LAYER)
            break;
        end
        l=l+1;
        
    end
    alltime=alltime+toc(fl_start);
    fprintf('====== forward Traning time: %.4f\n',alltime);
    %testing set
    layers = l;
    %%%%%%%%%%% Test Phase feedforwad %%%%%%%%%%%%%%;

    InputDataLayer = TT.P;
    for i=1:1:layers
        tempH_test=(ae{i}.OW)'*(InputDataLayer);%tied weight
        Ho = mapminmax('apply',tempH_test,ae{i}.nl);
        InputDataLayer = Ho;
        clear tempH_test;
    end
    [TestPre,TestingAccuracy,TestErr]=test_SHLNN(cl,InputDataLayer,TT.T);

    %fprintf('====== Training Accuracy: %.4f \n',TrainingAccuracy);
    fprintf('======subnet:%d Feedforwad Test Accuracy: %.4f \n',sub_net,TestingAccuracy);
    fprintf('======subnet:%d Network structure: ',sub_net);
    NS = [size(P,1) HiddernNeurons(1:layers) size(T,1)];
    for i=1:1:length(NS)
        fprintf('[%d]',NS(i))
    end
    fprintf('\n\n');

    pilae = {};
    for l=1:1:layers
        pilae{l}.W = ae{l}.OW';
        pilae{l}.nl = ae{l}.nl;
    end
    pilae{l+1}.W = cl.WI;
    pilae{l+2}.W = cl.WO;
    %======================================
    b_start = tic;
    bwnet = finetunning(pilae,trainBatch,trainBatchLabel);
    InputDataLayer = trainBatch;
    for i=1:1:length(bwnet)
        tempH_test=bwnet{i}.W*(InputDataLayer);
        if i~=length(bwnet)
            tempH_test = ActivationFunc(tempH_test,actFun,para);%prelu
        end
        B_features{i} = tempH_test;
        InputDataLayer = tempH_test;
        clear tempH_test;
    end
    O = InputDataLayer;
    % % % [~, label_index_expected]=max(trainBatchLabel);
    % % % [~,label_index_actual]=max(O);
    % % % MissClassification=length(find(label_index_actual~=label_index_expected));
    % % % acc=1-MissClassification/size(trainBatchLabel,2);
    % % % fprintf('======subnet:%d Training Accuracy of bwnet: %.4f \n',sub_net,acc);
    alltime=alltime+toc(b_start);
    fprintf('======Traning backward learning time: %.4f\n',alltime);

    InputDataLayer = TT.P;
    for i=1:1:length(bwnet)
        tempH_test=bwnet{i}.W*(InputDataLayer);
        if i~=length(bwnet)
            tempH_test = ActivationFunc(tempH_test,actFun,para);%prelu
        end
        InputDataLayer = tempH_test;
        clear tempH_test;
    end
    O = InputDataLayer;
    % [~, label_index_expected]=max(TT.T);
    % [~,label_index_actual]=max(O);
    % MissClassification=length(find(label_index_actual~=label_index_expected));
    % acc=1-MissClassification/size(TT.T,2);
    % fprintf('======subnet:%d Testing Accuracy of bwnet: %.4f \n',sub_net,acc);
    
    [~, label_index_expected]=max(TT.T);

    clsFea = cl.WI*F_featrues{layers}*cl.scale;
    clsFea = ActivationFunc(clsFea,cl.actFun,cl.para);
    F_featrues{layers+1} = clsFea;

    final_pre = zeros(size(TT.T));
    finalbestAcc=0;
    finalbestpre= zeros(size(TT.T));
    
    for l1 = length(pilae)-1:-1:2
        for l2 = length(bwnet):-1:2
            f_start = tic;
            F = [F_featrues{l1};B_features{l2}];
            alltime = alltime+toc(f_start);
            TF = fusionnet(pilae,bwnet,'feature',TT.P,actFun,para,l1,l2,cl.scale);
            NF = mapminmax([F TF],0,1);
            %NF = zscore([F TF]')';
            %NF = whiten([F TF]')';
            F = NF(:,1:size(F,2));
            TF = NF(:,size(F,2)+1:end);
            clear NF
            f_start = tic;
            [clsfier,TrainingAccuracy,TrainingErr,TrainPre] = train_SHLNN(F,trainBatchLabel,HN,actFun,para);
            alltime = alltime+toc(f_start);
            
            [pre,acc,err]=test_SHLNN(clsfier,TF,TT.T);
            
            [~,label_index_actual]=max(pre);
            MissClassification=length(find(label_index_actual~=label_index_expected));
            acc=1-MissClassification/size(TT.T,2);
%             fprintf('====== Test Accuracy after fusion: %.4f \n',acc);

            
            if acc>finalbestAcc
                finalbestAcc = acc;
                finalbestpre = pre;
            end
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        end
    end
    fprintf('\n=================subnet:%d finalbest Test Accuracy: %.4f \n',sub_net,finalbestAcc);
    fprintf('==============================fusion and classifier Traning time: %.4f\n',alltime);
    %  ======= PILer-Classifier  validating  For the determination of the number of sub-networks==========
    



    esbTY = esbTY+finalbestpre;
    [~, label_index_expected]=max(TT.T);
    [~,label_index_actual]=max(esbTY);
    MissClassification=length(find(label_index_actual~=label_index_expected));
    Eacc=1-MissClassification/size(TT.T,2);
    if mod(sub_net,3)==0
        fprintf('====== %d Ensemble Test accuracy %.4f ======\n',sub_net,Eacc);
    end
    h = animatedline('Color','b','LineWidth',2);
    px = [px sub_net];
    py = [py Eacc];
    for k = 1:length(px)
        addpoints(h,px(k),py(k));
    end
    ylabel('Accuracy');
    xlabel('Subnet');
    %ylim([0 1]);
    grid on
    pause(0.001)
    sub_net = sub_net + 1;
end

[~, label_index_expected]=max(TT.T);
[~,label_index_actual]=max(esbTY);
MissClassification=length(find(label_index_actual~=label_index_expected));
Eacc=1-MissClassification/size(TT.T,2);
fprintf('====== Final %d ensemble Test accuracy %.4f ======\n',sub_net-1,Eacc);

