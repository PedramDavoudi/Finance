function F5th
% al & bt must be grater than one.. according to paper
global r1 a b c al bt Tau
% Inputs
a=0.1;
b=0.2;
c=0.1;
al=1.5;
bt=3;
Tau=nan; % the treshold valu, nan= mean of data
nn=10; % the number of samples to plot the graphs
JustSort=1;
% loading data
r1=xlsread('Input\data.xlsx');
[~,k] =size(r1);

% Create Efficiency Curve
 [Rt,ALPM]=findEC(k,nn,10);
% plot(ALPM,Rt);
% return

% find Minimum of objective function
[xfinal,XfX,x,y,A,r]=OpO(k,nn);
% plot Grapgh
if exist('Rt','var')
    Grph(JustSort,xfinal,XfX,x,y,A,r,Rt,ALPM);% plot Graph
    Expt(xfinal,XfX,x,y,A,r,Rt,ALPM);% Export Simulation data to excell file
else
    Grph(JustSort,xfinal,XfX,x,y,A,r);% plot Graph
    Expt(xfinal,XfX,x,y,A,r);% Export Simulation data to excell file
end

end
function  Expt(xfinal,XfX,x,y,A,r,Rt,ALPM)
global r1 a b c al bt Tau
if ~exist('out','dir')
    mkdir('out')
end
k=length(xfinal);
Capx=cell(1,k);
for i=1:k
    Capx{i}=['w' num2str(i)];
end

Fxl=[mat2dataset(x.','VarNames',Capx),dataset(y.',A.',r.','VarNames',{'Ob_function','ALPM','Rerutn'})];
export(Fxl,'xlsfile','out\Siml');
if exist('Rt','var')
    Fxl=dataset(Rt,ALPM,'VarNames',{'ERetrun','ALPM'});
    export(Fxl,'xlsfile','out\EC');
end

Fxl=cell(k+1+6,2);

Fxl(1:k,1)=Capx.';
Fxl(1:k,2)=num2cell(xfinal);

Fxl(k+1,1)={'ObjVal'};
Fxl(k+1,2)={XfX};

Fxl(k+2,1)={'a'};
Fxl(k+2,2)={a};
Fxl(k+3,1)={'b'};
Fxl(k+3,2)={b};
Fxl(k+4,1)={'c'};
Fxl(k+4,2)={c};
Fxl(k+5,1)={'al'};
Fxl(k+5,2)={al};
Fxl(k+6,1)={'bt'};
Fxl(k+6,2)={bt};
Fxl(k+7,1)={'Tau'};
Fxl(k+7,2)={Tau};

Fxl = cell2table(Fxl,'VariableNames',{'Name','Value'});
writetable(Fxl,'out\tabledata.txt')

end
function Grph(JustSort,xfinal,XfX,x,y,A,r,Rt,ALPM)
k=length(xfinal);
% plot Graph
for i=1:k
    figure();
    [x0r,r0] =Xfine(x(i,:),r,1); % Just Sorted and not refined
    [x0y,y0]=Xfine(x(i,:),y);
    [x0A,A0]=Xfine(x(i,:),A);
    %     [x0r,r0]=Xfine(x(i,:),r);
    hold on
    %     plot(sortedX,y(I),'b . ',sortedX,A(I),'g . ',sortedX,r(I),'c -  ');
    plot(x0y,y0,'b . ',x0A,A0,'g . ',x0r,r0,'c -  ');
    plot(xfinal(i), XfX,'r O')
    title('Portfo Analyze');
    xlabel(['Weight of Asset #' num2str(i)]);
    legend({'Portfo Value Function','Portfo ALPM','Portfo Return','Optimum Point'})
    hold off
end
% efficient frontier
[r,A]=Xfine(r,A,JustSort);
% [sortedr,I] = sort(r);
figure();
hold on
[~,A0,r0]=f(xfinal);
plot(A,r,'b . ',A0, r0,'r O');
if exist('Rt','var')
    plot(ALPM,Rt,'g');
end
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
rB=mean(r);
%ObjectiveFunction
y=A-2*rB;%lambda
end
%%% Curve refinry
function [x,y]=Xfine(x,y,JustSort)
%
if nargin<3
    JustSort=1;
end
if JustSort
    [x,I] =sort(x);
    y=y(I);
    return;
end
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
% Simulating Sample around xfinal
function [x,y,A,r,ASigma]=Simul(xfinal,nn)
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
end
% Eficiency Curve *****************************
%************************************************* Objective
function [A]=fEFO(w1)
% Get the ALPM from @f
[~,A]=f(w1);
end
%********************************************** Constraint
function [c,ceq]=fEFC(w1)
% c is inequlity less than zero
% ceq is equality with zero

global r1 Beq%a b c al bt Tau

% Extract number of Obsevation
[~,k] =size(r1);
c=[];
% Check length of w1
if length(w1)<k
    w1(end+1:k,1)=zeros(k-length(w1),1);
end
w1(k+1:end)=[];
% build final Weight
w1(k,1)=1-sum(w1(1:k-1));
% check
if any(w1>1) || any(w1<0)
    ceq=-1;
    return;
end

% Calc Portfo
r=r1*w1;
ceq=mean(r)-Beq;


end

function [Rt,ALPM]=findEC(k,nn,Resolution)
disp('Optimization of Efficiency Curve is Started');
if nargin<3
    Resolution=10;
end
x0=(1/k).*ones(k,1);
lb=[zeros(k-1,1);0];
ub=[ones(k-1,1);0];
[~,~,~,r,~]=Simul(x0,nn);
MinR=min(r(isfinite(r)));
MaxR=max(r(isfinite(r)));
Stp=(MaxR-MinR)/Resolution;
global Beq
Rt=nan(Resolution+2,1);
ALPM=Rt;
% Optimization
% we use global search instead of local ones
options = optimoptions(@fmincon,'Algorithm','sqp'); % this interior-point algorithm result was so good in the case of two asset model
% (Other available algorithms: 'active-set', 'sqp', 'trust-region-reflective', 'interior-point')
% First satring values
i=0;
XfX0=-inf;
for Beq=MinR:Stp:MaxR
    for j=1:3
        problem = createOptimProblem('fmincon','objective',...
            @fEFO,'x0',x0,'lb',lb,'ub',ub,'options',options,'nonlcon',@fEFC);%,'ConstraintTolerance',10^-4);
        % x0 is the stating point, lb the lower bound and up the upper bound in this case nothing has to be changed by user
        % the lastinput is the shadow price or lagrange multiplier
        gs = GlobalSearch('NumStageOnePoints',400,'NumTrialPoints',2000);
        disp(['Solving itration #' num2str(j)]);
        % xfinal is the optimum weght for asset one
        % XfX is the Valu of objective function
        [xfinal, XfX,exitflag] = run(gs,problem);
        xfinal(k)=1-sum(xfinal(1:k-1)); % Correct the Last Element
        if XfX<XfX0 || exitflag~=-2
            %             XfX=XfX0;
            %xfinal=xfinal0;
            %break;
            XfX0=XfX;
            xfinal0=xfinal;
        end
        % simulating for the next starting point
        [x,~,A,r,ASigma]=Simul(xfinal,nn);
        % Check whether the xfinal is better than the sampl
        XfX1=min(A(r>Beq-Stp & r<Beq+Stp & isfinite(A)));
        if XfX<=XfX1
            break;
        end
        
        [~,I]=find(A==XfX1);
        
        x0=x(:,I(1));
        lb=x0-ASigma;
        ub=x0+ASigma;
        lb(end)=0;ub(end)=0; % Correct the Last Element %% may be remove in the next version
    end
    
    i=i+1;
    [~,ALPM(i),Rt(i)]=f(xfinal0);
    home;
    disp([num2str(100*i/(Resolution+1)) ' Percent is completed.']);
end
home;
disp('efficieny Curve is completed.');
end
function [xfinal,XfX,x,y,A,r]=OpO(k,nn)
% Optimization
disp('Optimization of Objective Function is Started');
% we use global search instead of local ones
options = optimoptions(@fmincon,'Algorithm','sqp'); % this interior-point algorithm result was so good in the case of two asset model
% (Other available algorithms: 'active-set', 'sqp', 'trust-region-reflective', 'interior-point')
% First satring values

x0=(1/k).*ones(k,1);
lb=[zeros(k-1,1);0];
ub=[ones(k-1,1);0];

XfX0=-inf;

for j=1:5
    problem = createOptimProblem('fmincon','objective',...
        @f,'x0',x0,'lb',lb,'ub',ub,'options',options);
    % x0 is the stating point, lb the lower bound and up the upper bound in this case nothing has to be changed by user
    % the lastinput is the shadow price or lagrange multiplier
    gs = GlobalSearch('NumStageOnePoints',400,'NumTrialPoints',2000);
    disp(['Solving itration #' num2str(j)]);
    % xfinal is the optimum weght for asset one
    % XfX is the Valu of objective function
    [xfinal, XfX,exitflag] = run(gs,problem);
    if (XfX>=XfX0 || exitflag==-2 ) && j>1
        XfX=XfX0;
        xfinal=xfinal0;
        break;
    end
    
    xfinal(k)=1-sum(xfinal(1:k-1)); % Correct the Last Element
    % simulating for the next starting point
    [x,y,A,r,ASigma]=Simul(xfinal,nn);
    % Check whether the xfinal is better than the sampl
    XfX1=min(y(isfinite(y)));
    if XfX<=XfX1
        break;
    end
    XfX0=XfX;
    xfinal0=xfinal;
    [~,I]=find(y==XfX1);
    
    x0=x(:,I(1));
    lb=x0-ASigma;
    ub=x0+ASigma;
    lb(end)=0;ub(end)=0; % Correct the Last Element %% my be remove in the next version
end

% display the Results
disp('Optimization Result listed below:');
for i=1:k
    disp(['Optimum Asset #' num2str(i) ' Weight:' ,num2str(100*round(xfinal(i),3))]);
end
disp(['Optimum Value of Objective Function:' ,num2str(round(XfX,3))]);

end