function F1stDraft_Cleared
% al & bt must be grater than one.. according to paper
global r1 r2 a b c al bt
% Inputs
a=0.1;
b=0.2;
c=0.1;
al=1.5;
bt=2;

n=10000; % number of Series

% Generate series
% for the next Version we do not generate the data
r1=mvnrnd(0,0.5,n);%MU,SIGMA,n
r2=mvnrnd(1,0.5,n);



% Optimization
% we use global search instead of local ones
options = optimoptions(@fmincon,'Algorithm','interior-point'); % this interior-point algorithm result was so good in the case of two asset model


problem = createOptimProblem('fmincon','objective',...
    @f,'x0',0.5,'lb',0,'ub',1,'options',options); 
% x0 is the stating point, lb the lower bound and up the upper bound in this case nothing has to be changed by user

gs = GlobalSearch;
disp('Solving started');
% xfinal is the optimum weght for asset one
% XfX is the Valu of objective function
[xfinal, XfX] = run(gs,problem);
% display the Results
disp(['Optimum First Assets Weight:' ,num2str(round(xfinal,3))]);
disp(['Optimum Value of Objective Function:' ,num2str(round(XfX,3))]);

% plot Grapgh
figure();
x=0:0.001:1;
[y,A,r]=f(0);
for i=2:length(x)
    [y0,A0,r0]=f(x(i));
    y=[y,y0];
    A=[A,A0];
    r=[r,r0];
end
plot(x,y,'b',x,A,'g--',x,r,'c -  ');
hold on
plot(xfinal, XfX,'r*')
title('Portfo Analyze');
xlabel('Weight of First Asset');
legend({'Portfo Value Function','Portfo ALPM','Portfo Return','Optimum Point'})
hold off
% efficient frontier
figure();
plot(A,r);
hold on
title('Efficient Frontier');
ylabel('Portfo Expected Return');
xlabel('Portfo ALPM');
hold off
end
%$$$$$$$$$$$$$$$$$$$$$$$######
%#######################
function [y,A,rB]=f(w1)
% check the weights illegal usage 
if (w1>1) || (w1<0)
   y=inf; 
   return;
end
global r1 r2 a b c al bt
% biuld final Weight
w2=1-w1;
% Extract number of Obsevation
n=size(r1,1);
% Calc Portfo
r=w1*r1+w2*r2;
% Expected return
rB=mean(r);
rBar=repmat(rB,n,1);
%
LPM=max(rBar-r,0);
UMP=max(r-rBar,0);
pminus=sum(LPM>0)/n;
Pplus=1-pminus;
A= a*sum(LPM.^al)-b*c*sum(UMP.^bt);

%ObjectiveFunction
y=A-rB;%lambda
end
