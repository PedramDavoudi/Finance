function F2ndDraft_Cleared
% al & bt must be grater than one.. according to paper
global r1 a b c al bt Tau
% Inputs
a=0.1;
b=0.2;
c=0.1;
al=1.5;
bt=3;
Tau=nan; % the treshold valu, nan= mean of data
nn=200; % the number of samples to plot the graphs
% sample generation data
n=10000; % number of Series
k=2; % number of assets
% Generate series
% for the next Version we do not generate the data
mu=1.5*ones(k,1)+rand(k,1);
Sigma=diag((ones(k,1)+rand(k,1)).^2);
r1=mvnrnd(mu,Sigma,n);%MU,SIGMA,n
f((1/k).*ones(k,1))

[n,k] =size(r1);
% Optimization
% we use global search instead of local ones
options = optimoptions(@fmincon,'Algorithm','interior-point'); % this interior-point algorithm result was so good in the case of two asset model
% (Other available algorithms: 'active-set', 'sqp', 'trust-region-reflective')

problem = createOptimProblem('fmincon','objective',...
    @f,'x0',(1/k).*ones(k,1),'lb',[zeros(k-1,1);0],'ub',[ones(k-1,1);0],'options',options);
% x0 is the stating point, lb the lower bound and up the upper bound in this case nothing has to be changed by user
% the lastinput is the shadow price or lagrange multiplier
gs = GlobalSearch;
disp('Solving started');
% xfinal is the optimum weght for asset one
% XfX is the Valu of objective function
[xfinal, XfX] = run(gs,problem);
xfinal(k)=1-sum(xfinal(1:k-1));

% display the Results
for i=1:k
disp(['Optimum Asset #' num2str(i) ' Weight:' ,num2str(round(xfinal(i),3))]);
end
disp(['Optimum Value of Objective Function:' ,num2str(round(XfX,3))]);

% plot Grapgh


x=rand(k,nn);
% normal to one each column
x=x./repmat(sum(x,1),k,1);
[y,A,r]=f(x(:,1));

for i=2:nn
    [y0,A0,r0]=f(x(:,i));
    y=[y,y0];
    A=[A,A0];
    r=[r,r0];
end
for i=1:k
    figure();
    [sortedX,I] = sort(x(i,:));
    plot(sortedX,y(I),'b',sortedX,A(I),'g--',sortedX,r(I),'c -  ');
    hold on
    plot(xfinal(i), XfX,'r*')
    title('Portfo Analyze');
    xlabel(['Weight of Asset #' num2str(i)]);
    legend({'Portfo Value Function','Portfo ALPM','Portfo Return','Optimum Point'})
    hold off
end
% efficient frontier
figure();
[sortedr,I] = sort(r);
plot(A(I),sortedr);
hold on
[~,A0,r0]=f(xfinal);
plot(A0, r0,'r*')
title('Efficient Frontier');
ylabel('Portfo Expected Return');
xlabel('Portfo ALPM');
legend({'R/R ','Optimum Point'})
hold off
end
%$$$$$$$$$$$$$$$$$$$$$$$######
%####################### Objective Function
function [y,A,rB]=f(w1)
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


% biuld final Weight
w1(k,1)=1-sum(w1(1:k-1));
% check
if any(w1>1) || any(w1<0)
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
