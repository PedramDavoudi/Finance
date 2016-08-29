function F1stDraft
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
% StartingPoint=1;

%options = optimoptions('fminunc','Display','final','Algorithm','quasi-newton', 'OptimalityTolerance',10^-20,'MaxFunctionEvaluations',1000);

%fh2 = objective with no gradient or Hessian
%[xfinal,fval,exitflag,output2] = fminunc(@f,StartingPoint,options);
%}
% options = optimoptions(@fmincon,'Algorithm','interior-point');
% 
% problem = createOptimProblem('fmincon','objective',...
%     fh2,'x0',StartingPoint,'lb',init.Min_Par_Calib,'ub',init.Max_Par_Calib,'options',options); %
% gs = GlobalSearch;
% disp('Solving started');
% [xfinal, XfX] = run(gs,problem);
%{
gradf = jacobian(f,x).'; % column gradf
%V=solve(gradf);%
hessf = jacobian(gradf,x);
fh = matlabFunction(f,gradf,hessf,'vars',{x});
options = optimoptions('fminunc', 'OptimalityTolerance',10^-20, 'MaxFunctionEvaluations',10^15,'StepTolerance', 10^-20,...
    'SpecifyObjectiveGradient', true, ...
    'HessianFcn', 'objective', ...
    'Algorithm','trust-region', ...
    'Display','final');
[xfinal,fval,exitflag,output] = fminunc(fh,StartinPoint,options);
%}
options = optimoptions(@fmincon,'Algorithm','interior-point');

problem = createOptimProblem('fmincon','objective',...
    @f,'x0',0.5,'lb',0,'ub',1,'options',options); %
gs = GlobalSearch;
disp('Solving started');
[xfinal, XfX] = run(gs,problem);

round(xfinal,3)
round(XfX,3)
%plot
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
xlabel('Weight of First Asset');
legend({'Portfo Value Function','Portfo ALPM','Portfo Return','Optimum Point'})
hold off
end
%$$$$$$$$$$$$$$$$$$$$$$$######
%#######################
function [y,A,rB]=f(w1)
if (w1>1) || (w1<0)
   y=inf; 
   return;
end
global r1 r2 a b c al bt
w2=1-w1;
n=size(r1,1);
% Calc
r=w1*r1+w2*r2;
rB=mean(r);
rBar=repmat(rB,n,1);
% A= a*max(0,rBar-r).^al+b*c*max(0,r-rBar).^bt;
%
LPM=max(rBar-r,0);
UMP=max(r-rBar,0);
pminus=sum(LPM>0)/n;
Pplus=1-pminus;
A= a*sum(LPM.^al)-b*c*sum(UMP.^bt);

y=A-rB;%lambda
end
