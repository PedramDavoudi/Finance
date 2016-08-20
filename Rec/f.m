function y=f(w1)
% check the weights illegal usage

global r1 a b c al bt Tau

% Extract number of Obsevation
[n,k] =size(r1);
% Check length of w1
if length(w1)<k
    w1(end+1:k,1)=zeros(k-length(w1),1);
end
w1(k+1:end)=[];


%
lambda=w1(k);


% build final Weight
w1(k,1)=1-sum(w1(1:k-1));
% check
if any(w1>1) || any(w1<0)
    A=inf;
    rB=-inf;
    y=inf;
    return;
end

% Calc Portfo
r=r1*w1;

% Expected return
if isnan(Tau)
    rB=mean(r);
else
    rB=Tau;
end
rBar=repmat(rB,n,1);
%
LPM=max(rBar-r,0);
UMP=max(r-rBar,0);
pminus=sum(LPM>0)/n;
Pplus=1-pminus;
A= a*pminus*sum(LPM.^al)-b*c*Pplus*sum(UMP.^bt);

%ObjectiveFunction
y=A-2*rB;%lambda
end
