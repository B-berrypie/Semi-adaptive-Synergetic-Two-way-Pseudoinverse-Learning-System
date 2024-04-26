function H = DeactivationFunc( tempH, ActivationFunction,p)
%ACTIVATIONFUNC Summary of this function goes here
%   Detailed explanation goes here
switch lower(ActivationFunction)
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid

    case {'sin','sine'}
        %%%%%%%% Sine

    case {'hardlim'}
        %%%%%%%% Hard Limit

    case {'tribas'}
        %%%%%%%% Triangular basis function

    case {'radbas'}
        %%%%%%%% Radial basis function

    case {'gau'}

    case {'relu'}
        %%%%%%%% ReLU
        idx = find(tempH(:)<0);
        tempH(idx)=0;
        H = tempH;
    case {'srelu'}
        idx = find(tempH(:)<p);
        tempH(idx)=0;
        H = tempH;
    case {'tan'}
        H = 0.5.*log((1.+ tempH)./(1.- tempH));
        %H = tempH;
    case {'prelu'}
        alpha = 0.1;
        idx = find(tempH(:)<0);
        tempH(idx)=tempH(idx)./alpha;
        H = tempH;
    case {'mor'}
        
end
end

