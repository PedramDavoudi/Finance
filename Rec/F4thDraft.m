function F4thDraft
% al & bt must be grater than one.. according to paper
global r1 a b c al bt Tau
% Inputs
a=0.1;
b=0.2;
c=0.1;
al=1.5;
bt=3;
Tau=nan; % the treshold valu, nan= mean of data
nn=20000; % the number of samples to plot the graphs
% sample generation data
% n=10000; % number of Series
% k=2; % number of assets
% % Generate series
% for the next Version we do not generate the data
% mu=1.5*ones(k,1)+rand(k,1);
% Sigma=diag((ones(k,1)+rand(k,1)).^2);
% r1=mvnrnd(mu,Sigma,n);%MU,SIGMA,n
% f((1/k).*ones(k,1))
r1=xlsread('Input\data.xlsx');
[~,k] =size(r1);
% Optimization
% we use global search instead of local ones
options = optimoptions(@fmincon,'Algorithm','sqp'); % this interior-point algorithm result was so good in the case of two asset model
% (Other available algorithms: 'active-set', 'sqp', 'trust-region-reflective', 'interior-point')
% First satring values
x0=(1/k).*ones(k,1);
lb=[zeros(k-1,1);0];
ub=[ones(k-1,1);0];
for j=1:3
    problem = createOptimProblem('fmincon','objective',...
        @f,'x0',x0,'lb',lb,'ub',ub,'options',options);
    % x0 is the stating point, lb the lower bound and up the upper bound in this case nothing has to be changed by user
    % the lastinput is the shadow price or lagrange multiplier
    gs = GlobalSearch;
    disp(['Solving itration #' num2str(j)]);
    % xfinal is the optimum weght for asset one
    % XfX is the Valu of objective function
    [xfinal, XfX] = run(gs,problem);
    xfinal(k)=1-sum(xfinal(1:k-1));
    % simulating for the next starting point
    
    ASigma=xfinal./4;%repmat(0.02,k,1)
    ASigma(xfinal>0.5)=(1-xfinal(xfinal>0.5))./4;
    Sigma=diag(ASigma.^2);
    x=mvnrnd(xfinal,Sigma,nn).';%MU,SIGMA,n
    [y,A,r]=f(xfinal);% Add the final point to Sapmle collection
    x=[xfinal,x];
    for i=2:nn+1
        [y0,A0,r0]=f(x(:,i));
        y=[y,y0];
        A=[A,A0];
        r=[r,r0];
    end
    XfX1=min(y);
    if XfX<=XfX1
        break;
    end
    [~,I]=find(y==XfX1);
    
    x0=x(:,I(1));
    lb=x0-ASigma;
    ub=x0+ASigma;
    lb(end)=0;ub(end)=0;
end

% display the Results
disp('Optimization Result listed below:');
for i=1:k
    disp(['Optimum Asset #' num2str(i) ' Weight:' ,num2str(100*round(xfinal(i),3))]);
end
disp(['Optimum Value of Objective Function:' ,num2str(round(XfX,3))]);

% plot Grapgh

% Gathering Random Samples
%{
x= randn(k,nn);%*R
x=repmat(-min(x,[],2),1,nn)+x; % non zero
% normal to one each column
x=x./repmat(sum(x,1),k,1);
x=x./repmat(mean(x,2),1,nn); % recenter
x=x.*repmat(xfinal,1,nn);% recenter
%}
%{
ASigma=xfinal./4;%repmat(0.02,k,1)
Sigma=diag(ASigma.^2);
x=mvnrnd(xfinal,Sigma,nn).';%MU,SIGMA,n
[y,A,r]=f(xfinal);% Add the final point to Sapmle collection
x=[xfinal,x];
for i=2:nn+1
    [y0,A0,r0]=f(x(:,i));
    y=[y,y0];
    A=[A,A0];
    r=[r,r0];
end
%}
for i=1:k
    figure();
    [sortedX,I] = sort(x(i,:));
    [x0y,y0]=Xfine(x(i,:),y);
    [x0A,A0]=Xfine(x(i,:),A);
%     [x0r,r0]=Xfine(x(i,:),r);
    hold on
%     plot(sortedX,y(I),'b . ',sortedX,A(I),'g . ',sortedX,r(I),'c -  ');
plot(x0y,y0,'b . ',x0A,A0,'g . ',sortedX,r(I),'c -  ');
    plot(xfinal(i), XfX,'r O')
    title('Portfo Analyze');
    xlabel(['Weight of Asset #' num2str(i)]);
    legend({'Portfo Value Function','Portfo ALPM','Portfo Return','Optimum Point'})
    hold off
end
% efficient frontier
[r,A]=Xfine(r,A);
% [sortedr,I] = sort(r);
figure();
hold on
[~,A0,r0]=f(xfinal);
plot(A,r,'b . ',A0, r0,'r O');
title('Efficient Frontier');
ylabel('Portfo Expected Return');
xlabel('Portfo ALPM');
legend({'R/R','Optimum Point'})
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
%%% Curve refinry
function [x,y]=Xfine(x,y)
%
[x,I] =sort(x);
y=y(I);
return;
%}
lb=min(x(isfinite(x)));
ub=max(x(isfinite(x)));
Stp=(ub-lb)/100;
x0=[];
y0=[];
for i=lb:Stp:ub
    m0=mean(x(x>=i & x<i+Stp));
    if ~isnan(m0)
    x0=[x0; m0];
    y0=[y0;min(y(x>=i & x<i+Stp))];
    end
end

x=x0;
y=y0;%(I);
end