function F6th
% al & bt must be grater than one.. according to paper
tic;
global r1 alpha b c a beta Tau
% Inputs
% alpha the coef of LPM
% a the power of LPM
% beta the coef of UMP
% b the coef of UMP
% c the power of UMP

%
% Resolution=1;
%JustSimulation=0;
JustSort=1;
Inp=dataset('xlsfile','Input\Senarios');
S=size(Inp,1);

% loading data
r1=xlsread('Input\data');
r1(any(isnan(r1),2),:)=[]; % Remove nan frome the data
[~,k] =size(r1);


for s=1:S
    SenarioName=Inp.SenarioName{s};
    a=Inp.a(s);
    b=Inp.b(s);
    c=Inp.c(s);
    alpha=Inp.alpha(s);
    beta=Inp.beta(s);
    Tau=Inp.Tau(s);
    JustSimulation=Inp.JustSimulation(s);
    CoverOut=Inp.CoverOut;
    
    nn=Inp.SampleSize(s);
    Resolution=Inp.Resolution(s);
    if ~(isfinite(a) && isfinite(b) &&isfinite(c) && isfinite(alpha) && isfinite(beta))
        warning('not ')
        continue;
    end
    
    
    if isempty(SenarioName)
        SenarioName=['Sen' num2str(s)];
    end
    if isnan(Resolution)
        Resolution=10;
    end
    if isnan(CoverOut)
        CoverOut=0;
    end
    % find a reliable Sample
    [x,A,r]=OpO(k,nn);
    
    
    % Create Efficiency Curve
    if JustSimulation==0
        [Rt,ALPM,xFinal]=findEC(x,A,r,k,nn,Resolution,CoverOut);
        PM=MPM(Rt);
    else
        PM=MPM([],Resolution);
    end
    
    
    % plot Grapgh
    if exist('Rt','var')
        Expt(x,A,r,SenarioName,PM,Rt,ALPM,xFinal);% Export Simulation data to excell file
        Grph(JustSort,x,A,r,SenarioName,PM,Rt,ALPM,xFinal);% plot Graph
    else
        Expt(x,A,r,SenarioName,PM);% Export Simulation data to excell file
        Grph(JustSort,x,A,r,SenarioName,PM);% plot Graph
    end
    disp(['************ Senrio: ' SenarioName ' Was Completed. *****************']);
    disp(['Elapsed Time is: ' datestr(toc/(24*3600), 'HH:MM:SS')]);
end


end
%****************************************** Data Exporting
function Expt(x,A,r,SenarioName,PM,Rt,ALPM,xFinal)
global alpha b beta a c Tau

if ~exist('SenarioName','var')
    SenarioName='Sen';
end
if ~exist('out','dir')
    mkdir('out')
end
if exist(['out\' SenarioName],'dir')
    rmdir(['out\' SenarioName],'s');
end
mkdir(['out\' SenarioName]);

k=size(x,1);
Capx=cell(1,k);

for i=1:k
    Capx{i}=['w' num2str(i)];
end
% Exporting Simulation Data
Fxl=[mat2dataset(x.','VarNames',Capx),dataset(A.',r.','VarNames',{'ALPM','Return'})];
export(Fxl,'xlsfile',['out\' SenarioName '\Siml.xlsx']);
clear Fxl
% Exporting Simulation Data Efficiency Curve

if exist('Rt','var')
    Fxl=[mat2dataset(xFinal,'VarNames',Capx),dataset(ALPM,Rt,'VarNames',{'ALPM','Return'})];
    %export(Fxl,'xlsfile',['out\' SenarioName '\EFA.xlsx'])
else
    Fxl=dataset();
end
% Fxl=cell(k+1+6,2);
if exist('PM','var')
    Fxl=[Fxl, dataset(PM.MV.A,PM.MV.R,PM.CV.A,PM.CV.R,PM.MAD.A,PM.MAD.R,'VarNames',{'Mean_Var_ALPM' 'Mean_Var_Return' 'CVaR_ALPM' 'CVaR_Return' 'Absolute_Deviation_ALPM' 'Absolute_Deviation_Return'})];
    %export(Fxl0,'xlsfile',['out\' SenarioName '\EFM.xlsx'])
end

export(Fxl,'XLSfile',['out\' SenarioName '\EF.xlsx'])


Fxl=table();
Fxl.ParameterName={'a';'alpha';'b';'beta';'c';'Tau'};
Fxl.ParameterValue=[a;alpha;b;beta;c;Tau];
writetable(Fxl,['out\' SenarioName '\table.txt'])

end
%****************************************** Graph Creater
function Grph(JustSort,x,A,r,SenarioName,PM,Rt,ALPM,xFinal)
global r1
if ~exist('SenarioName','var')
    SenarioName='Sen';
end
if exist('Rt','var')
    x=[x,xFinal.'];
    A=[A,ALPM.'];
    r=[r,Rt.'];
end
k=size(x,1);
% plot Graph
for i=1:k
    try %#ok<TRYNC>
        hist(r1(:,i));% ksdensity
        title(['Kernel of Asset #' num2str(i)]);
        saveas(gcf,['out\' SenarioName '\Kr' num2str(i) '.bmp'])
        close gcf
    end
end
for i=1:k
    try
        figure();
        hold on
        [x0r,r0] =Xfine(x(i,:),r,1); % Just Sorted and not refined in any case
        plot(x0r,r0,'b . ');
        legend({'Simulation'});
        if exist('Rt','var')
            [x0y,y0]=Xfine(xFinal(:,i),Rt,1);
            plot(x0y,y0,'r*');
            legend({'Simulation','Optimum Point in Return'});%,})
        end
        %plot(xfinal(i), XfX,'r O')
        title('Portfo Analyze');
        ylabel('Return');
        xlabel(['Weight of Asset #' num2str(i)]);
        
        hold off
        saveas(gcf,['out\' SenarioName '\wR' num2str(i) '.bmp'])
        close gcf
        %------------------------------------------------------
        figure();
        hold on
        [x0A,A0]=Xfine(x(i,:),A,JustSort);
        plot(x0A,A0,'b . ');
        legend({'Simulation'});
        if exist('ALPM','var')
            [x1y,y1]=Xfine(xFinal(:,i),ALPM,1);
            plot(x1y,y1,'r*');
            legend({'Simulation','Optimum Point in ALPM'});
        end
        %plot(xfinal(i), XfX,'r O')
        title('Portfo Analyze');
        ylabel('ALPM')
        xlabel(['Weight of Asset #' num2str(i)]);
        hold off
        saveas(gcf,['out\' SenarioName '\wA' num2str(i) '.bmp'])
        close gcf
    end
end
% efficient frontier
[r,A]=Xfine(r,A,JustSort);
% [sortedr,I] = sort(r);
figure();

%[~,A0,r0]=f(xfinal);

hold on
if exist('Rt','var')
    plot(ALPM,Rt,'g');
    legend([get(legend,'string'),{'ALPM'}]);
else
    plot(A,r,'y . ');%,A0, r0,'r O');Simul
    legend({'Simulation'});
end
if exist('PM','var')
    %     plot(PM.MV.A,PM.MV.R,'b')%,PM.CV.A,PM.CV.A,'r',PM.MAD.A,PM.MAD.R,'d');
    %     legend([get(legend,'string'),{'Mean-Variance'}]);%,'CVaR','MAD'})
    plot(PM.MV.A,PM.MV.R,'b.',PM.CV.A,PM.CV.R,'r-.',PM.MAD.A,PM.MAD.R,'c--');
    legend([get(legend,'string'),{'Mean-Variance','CVaR','MAD'}]);
end

title('Efficient Frontier');
ylabel('Portfo Expected Return');
xlabel('Portfo ALPM');

hold off
saveas(gcf,['out\' SenarioName '\EC.bmp'])
%close gcf
end
%$$$$$$$$$$$$$$$$$$$$$$$###### Post Modern portfo Managment
function [out]=MPM(ret,NumPoint)
% Define portfo

global r1
if ~isempty(ret)
    
    p0 = Portfolio('assetmean', mean(r1,1).', 'assetcovar', r1.'*r1);
    p0 = setDefaultConstraints(p0);
    w0 = estimateFrontierByReturn(p0,ret)*100;
    
    p1 = PortfolioCVaR;
    p1 = simulateNormalScenariosByData(p1, r1, 2000);
    p1 = setDefaultConstraints(p1);
    p1 = PortfolioCVaR(p1, 'ProbabilityLevel', 0.95); %'Scenarios', r1, 'Budget', 1,
    w1 = estimateFrontierByReturn(p1,ret)*100;
    
    p2 = PortfolioMAD;
    p2 = simulateNormalScenariosByData(p2, r1, 2000);
    p2 = setDefaultConstraints(p2);
    %p2 = PortfolioMAD(p2,'lb', 0.25,'ub', 0.55);% 'Scenarios', r1,
    w2 = estimateFrontierByReturn(p2,ret)*100;
    NumPoint=length(ret);
else
    if ~isempty('NumPoint','var')
        NumPoint=10;
    end
    p0 = Portfolio('assetmean', mean(r1,1).', 'assetcovar', r1.'*r1);
    p0 = setDefaultConstraints(p0);
    w0 = estimateFrontier(p0,NumPoint)*100;
    
    p1 = PortfolioCVaR;
    p1 = simulateNormalScenariosByData(p1, r1, 2000);
    p1 = setDefaultConstraints(p1);
    p1 = PortfolioCVaR(p1, 'ProbabilityLevel', 0.95); %'Scenarios', r1, 'Budget', 1,
    w1 = estimateFrontier(p1,NumPoint)*100;
    
    p2 = PortfolioMAD;
    p2 = simulateNormalScenariosByData(p2, r1, 2000);
    p2 = setDefaultConstraints(p2);
    %p2 = PortfolioMAD(p2,'lb', 0.25,'ub', 0.55);% 'Scenarios', r1,
    w2 = estimateFrontier(p2,NumPoint)*100;
    
end


out.MV.W=w0.';
out.MV.R=nan(NumPoint,1);
out.MV.A=nan(NumPoint,1);
for i=1:NumPoint
    [~,out.MV.A(i),out.MV.R(i)]=f(w0(:,i));
    if out.MV.R(i)<=-10^9
        out.MV.R(i)=nan;
        out.MV.A(i)=nan;
        out.MV.W(i,:)=nan;
    end
    
end

% [rsk1,ret1]=p1.plotFrontier(NumPoint);
% out.CV=[rsk1,ret1];

out.CV.W=w1.';
out.CV.R=nan(NumPoint,1);
out.CV.A=nan(NumPoint,1);
for i=1:NumPoint
    [~,out.CV.A(i),out.CV.R(i)]=f(w1(:,i));
    if out.CV.R(i)<=-10^9
        out.CV.R(i)=nan;
        out.CV.A(i)=nan;
        out.CV.W(i,:)=nan;
    end
end
%[out.CV.W,out.CV.A,out.CV.R]=refinery(out.CV.W,out.CV.A,out.CV.R);
%[out.MV.W,out.MV.A,out.MV.R]=refinery(out.MV.W,out.MV.A,out.MV.R);
% [rsk0,ret0]=p0.plotFrontier(NumPoint);
% out.MV=[rsk0,ret0];

out.MAD.W=w2.';
out.MAD.R=nan(NumPoint,1);
out.MAD.A=nan(NumPoint,1);
for i=1:NumPoint
    [~,out.MAD.A(i),out.MAD.R(i)]=f(w2(:,i));
    if out.MAD.R(i)<=-10^9
        out.MAD.R(i)=nan;
        out.MAD.A(i)=nan;
        out.MAD.W(i,:)=nan;
    end
end
%[out.MAD.W,out.MAD.A,out.MAD.R]=refinery(out.MAD.W,out.MAD.A,out.MAD.R);
% [rsk2,ret2]=plotFrontier(p2,NumPoint);
% out.MAD=[rsk2,ret2];
%close all
%}
end
%####################### Sampler Objective Function
function [y,A,rB]=f(w1)
% check the weights illegal usage
% the number of degree of freedome is k-1
global r1 alpha b beta a c Tau
w1=w1./100;
% Extract number of Obsevation
[n,k] =size(r1);

% Check length of w1
if length(w1)<k
    w1(end+1:k,1)=zeros(k-length(w1),1);
end
w1(k+1:end)=[];

% build final Weight
w1(k,1)=1-sum(w1(1:k-1));

% check
if any(w1>1) || any(w1<0)
    A=10^10;
    rB=-10^10;
    y=10^10;
    return;
end

% Calc Portfo
r=r1*w1;

% Expected return
if isfinite(Tau)
    rB=Tau;
else
    rB=mean(r);
end
rBar=repmat(rB,n,1);
%
LPM=max(rBar-r,0);
UMP=max(r-rBar,0);
pminus=sum(LPM>0)/n;
Pplus=sum(UMP>0)/n;
A= alpha*pminus*sum(LPM.^a)+b*Pplus*beta*sum(UMP.^c);

rB=mean(r);
%ObjectiveFunction
y=A-2*rB;%lambda
end
%%% Curve refinry
function [x,y]=Xfine(x,y,Resolution,JustSort)
%
if nargin<4
    JustSort=1;
end
if nargin<3
    Resolution=100;
end
if JustSort
    [x,I] =sort(x);
    y=y(I);
    return;
end
%}
lb=min(x(isfinite(x)));
ub=max(x(isfinite(x)));
Stp=(ub-lb)/Resolution;
x0=[];
y0=[];
for i=lb:Stp:ub
    m0=mean(x(x>=i & x<i+Stp));
    if ~isnan(m0)
        x0=[x0; m0]; %#ok<AGROW>
        [~,I]=find(x>=i & x<i+Stp);
        if isempty(I)
            y0=[y0,nan]; %#ok<AGROW>
        else
            y0=[y0,min(y(:,I))]; %#ok<AGROW>
        end
    end
end

x=x0;
y=y0;%(I);
end
% Simulating Sample around xfinal
function [x,y,A,r,ASigma]=Simul(xfinal,nn)
k=size(xfinal,1);
ASigma=xfinal./4;%repmat(0.02,k,1)
ASigma(xfinal>50)=(1-xfinal(xfinal>50))./4;
% Extreme Result ignorance
ASigma(ASigma<10^-4)=25;
%
Sigma=diag(ASigma.^2);
x=mvnrnd(xfinal,Sigma,nn).';%MU,SIGMA,n
x=abs(x);
x=x./repmat(sum(x,1),k,1);
y=nan(1,nn+1);
A=y;
r=y;
[y(1),A(1),r(1)]=f(xfinal);% Add the final point to Sapmle collection
x=[xfinal,x];
for i=2:nn+1
    [y(i),A(i),r(i)]=f(x(:,i));
end
x(:,r<-10^9)=[];
y(r<-10^9)=[];
A(r<-10^9)=[];
r(r<-10^9)=[];
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
w1=w1./100;
% w1.'
global r1 Beq%alpha b c a c Tau
%w1=w1/100;
% Extract number of Obsevation
[~,k] =size(r1);
c=-1;
ceq=[];
% Check length of w1
if length(w1)<k
    w1(end+1:k,1)=zeros(k-length(w1),1);
end
w1(k+1:end)=[];
% build final Weight
w1(k,1)=1-sum(w1(1:k-1));
% check
if any(w1>1) || any(w1<0)
    return;
end

% Calc Portfo
r=r1*w1;
c=Beq-mean(r);


end
%{
function [c,ceq]=fEFC0(w1)
% c is inequlity less than zero
% ceq is equality with zero

global r1 Beq%alpha b c a c Tau

% Extract number of Obsevation
[~,k] =size(r1);
c=[];
ceq=-1;
% Check length of w1
if length(w1)<k
    w1(end+1:k,1)=zeros(k-length(w1),1);
end
w1(k+1:end)=[];
% build final Weight
w1(k,1)=1-sum(w1(1:k-1));
% check
if any(w1>1) || any(w1<0)
    return;
end

% Calc Portfo
r=r1*w1;
ceq=mean(r)-Beq;


end
%}
% Revesrse get R and return w
function [w]=Reversef(rB)
% check the weights illegal usage
% the number of degree of freedome is k-1
global r1 
% Extract number of Obsevation
[~,k] =size(r1);
W= sym('W',[k,1]);
%assume(W>=0 & W<=1);
%assumeAlso(sum(W)==1);
% Check length of w1
r=mean(r1);
J=@(x)(r*x.'-rB)^2;

%Teta=symvar(W);

%
StartingPoint=repmat(1/k,1,k);
Aeq=ones(1,k);
A=[];
beq = 1;
b=[];
nonlcon=[];
%gradf = @(x)2*(r*x.'-rB)*r; % column gradf
%hessf =  @(x)(2*ones(k,k));

options = optimoptions('fmincon', ...
    'SpecifyObjectiveGradient', false, ...
    'Algorithm','interior-point', ...
    'Display','off','MaxFunctionEvaluations',10^20);% interior-point %trust-region-reflective
[x,fval,exitflag] = fmincon(J,StartingPoint,A,b,Aeq,beq,zeros(1,k),ones(1,k),nonlcon,options);
w=x.'*100;

end
%***********************************************  Sample Biulder
function [x,A,r]=OpO(k,nn)
%nn=k*nn;
% this function just replicate a good sample
x = randfixedsum(k,nn,100,0,100);
A=nan(1,nn);
%y=A;
r=A;
%[y(1),A(1),r(1)]=f(xfinal);% Add the final point to Sapmle collection
%x=[xfinal,x];
for i=1:nn
    [~,A(i),r(i)]=f(x(:,i));
end
x(:,r<-10^9)=[];
%y(r<-10^9)=[];
A(r<-10^9)=[];
r(r<-10^9)=[];
disp('************************ Sampling Done *************************');

%{
% Optimization
disp('Optimization of Objective Function To Build qualified Sample Started');
% we use global search instead of local ones
options = optimoptions(@fmincon, 'Algorithm','sqp'); % this interior-point algorithm result was so good in the case of two asset model
% (Other available algorithms: 'active-set', 'sqp', 'trust-region-reflective', 'interior-point')'UseParallel', true
% First satring values

x0=(1/k).*ones(k-1,1);
lb=zeros(k-1,1);
ub=ones(k-1,1);

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
    
    x0=x(1:k-1,I(1));
    lb=x0-ASigma(1:k-1);
    ub=x0+ASigma(1:k-1);
    
end
x(:,r<-10^9)=[];
y(r<-10^9)=[];
A(r<-10^9)=[];
r(r<-10^9)=[];
% display the Results
% disp('Optimization Result listed below:');
% for i=1:k
%     disp(['Optimum Asset #' num2str(i) ' Weight:' ,num2str(100*round(xfinal(i),3))]);
% end
% disp(['Optimum Value of Objective Function:' ,num2str(round(XfX,3))]);
%}

end
function [x,v] = randfixedsum(n,m,s,a,b)

% [x,v] = randfixedsum(n,m,s,a,b)
%
%   This generates an n by m array x, each of whose m columns
% contains n random values lying in the interval [a,b], but
% subject to the condition that their sum be equal to s.  The
% scalar value s must accordingly satisfy n*a <= s <= n*b.  The
% distribution of values is uniform in the sense that it has the
% conditional probability distribution of a uniform distribution
% over the whole n-cube, given that the sum of the x's is s.
%
%   The scalar v, if requested, returns with the total
% n-1 dimensional volume (content) of the subset satisfying
% this condition.  Consequently if v, considered as a function
% of s and divided by sqrt(n), is integrated with respect to s
% from s = a to s = b, the result would necessarily be the
% n-dimensional volume of the whole cube, namely (b-a)^n.
%
%   This algorithm does no "rejecting" on the sets of x's it
% obtains.  It is designed to generate only those that satisfy all
% the above conditions and to do so with a uniform distribution.
% It accomplishes this by decomposing the space of all possible x
% sets (columns) into n-1 dimensional simplexes.  (Line segments,
% triangles, and tetrahedra, are one-, two-, and three-dimensional
% examples of simplexes, respectively.)  It makes use of three
% different sets of 'rand' variables, one to locate values
% uniformly within each type of simplex, another to randomly
% select representatives of each different type of simplex in
% proportion to their volume, and a third to perform random
% permutations to provide an even distribution of simplex choices
% among like types.  For example, with n equal to 3 and s set at,
% say, 40% of the way from a towards b, there will be 2 different
% types of simplex, in this case triangles, each with its own
% area, and 6 different versions of each from permutations, for
% a total of 12 triangles, and these all fit together to form a
% particular planar non-regular hexagon in 3 dimensions, with v
% returned set equal to the hexagon's area.
%
% Roger Stafford - Jan. 19, 2006

% Check the arguments.
if (m~=round(m))||(n~=round(n))||(m<0)||(n<1)
    error('n must be a whole number and m a non-negative integer.')
elseif (s<n*a)||(s>n*b)||(a>=b)
    error('Inequalities n*a <= s <= n*b and a < b must hold.')
end

% Rescale to a unit cube: 0 <= x(i) <= 1
s = (s-n*a)/(b-a);

% Construct the transition probability table, t.
% t(i,j) will be utilized only in the region where j <= i + 1.
k = max(min(floor(s),n-1),0); % Must have 0 <= k <= n-1
s = max(min(s,k+1),k); % Must have k <= s <= k+1
s1 = s - [k:-1:k-n+1]; % s1 & s2 will never be negative
s2 = [k+n:-1:k+1] - s;
w = zeros(n,n+1); w(1,2) = realmax; % Scale for full 'double' range
t = zeros(n-1,n);
tiny = 2^(-1074); % The smallest positive matlab 'double' no.
for i = 2:n
    tmp1 = w(i-1,2:i+1).*s1(1:i)/i;
    tmp2 = w(i-1,1:i).*s2(n-i+1:n)/i;
    w(i,2:i+1) = tmp1 + tmp2;
    tmp3 = w(i,2:i+1) + tiny; % In case tmp1 & tmp2 are both 0,
    tmp4 = (s2(n-i+1:n) > s1(1:i)); % then t is 0 on left & 1 on right
    t(i-1,1:i) = (tmp2./tmp3).*tmp4 + (1-tmp1./tmp3).*(~tmp4);
end

% Derive the polytope volume v from the appropriate
% element in the bottom row of w.
v = n^(3/2)*(w(n,k+2)/realmax)*(b-a)^(n-1);

% Now compute the matrix x.
x = zeros(n,m);
if m == 0, return, end % If m is zero, quit with x = []
rt = rand(n-1,m); % For random selection of simplex type
rs = rand(n-1,m); % For random location within a simplex
s = repmat(s,1,m);
j = repmat(k+1,1,m); % For indexing in the t table
sm = zeros(1,m); pr = ones(1,m); % Start with sum zero & product 1
for i = n-1:-1:1  % Work backwards in the t table
    e = (rt(n-i,:)<=t(i,j)); % Use rt to choose a transition
    sx = rs(n-i,:).^(1/i); % Use rs to compute next simplex coord.
    sm = sm + (1-sx).*pr.*s/(i+1); % Update sum
    pr = sx.*pr; % Update product
    x(n-i,:) = sm + pr.*e; % Calculate x using simplex coords.
    s = s - e; j = j - e; % Transition adjustment
end
x(n,:) = sm + pr.*s; % Compute the last x

% Randomly permute the order in the columns of x and rescale.
rp = rand(n,m); % Use rp to carry out a matrix 'randperm'
[ig,p] = sort(rp); % The values placed in ig are ignored
x = (b-a)*x(p+repmat([0:n:n*(m-1)],n,1))+a; % Permute & rescale x

end
%***********************************************  Find outlier data
function [OutX,OutA,OutR]=Outer(Rt,ALPM,xSam,ASam,rSam)
%% Points Inside Convex Polygon
%%
% Define the x and y coordinates of polygon vertices to create a pentagon.
yv=Rt;
xv=ALPM;
% Define x and y coordinates of Simulation data
yq=rSam;
xq=ASam;

%Completed the curve
yv(end+1)=0;
xv(end+1)=0;


% Sort data
[yv,I]=unique(yv);
xv=xv(I);
yv(end+1)=yv(end);
xv(end+1)=2*max(xq);
yv(end+1)=0;
xv(end+1)=2*max(xq);
yv(end+1)=0;
xv(end+1)=0;



%%
% Determine whether each point lies inside or on the edge of the polygon
% area. Also determine whether any of the points lie on the edge of the
% polygon area.
[in,on] = inpolygon(xq,yq,xv,yv);
OutA=xq(~(in | on));
OutR=yq(~(in | on));
OutX=xSam(~(in | on),:);
% refine data
[OutX,OutA,OutR]=refinery(OutX,OutA,OutR);
% OutA=OutA(isfinite(OutR));
% OutX=OutX(isfinite(OutR),:);
% OutR=OutR(isfinite(OutR));
% [OutR,ia]=unique(OutR);
% OutA=OutA(ia);
% OutX=OutX(ia,:);
%%
end
% ************************************** Asure Convexity
function [x,A,R]=refinery(x,A,R)
%Notice: the unconvex point must be Removed
%each row shows a perid of time A , R is vectoral column
A(A>10^8)=nan; R(R<-10^8)=nan; 
A=A(isfinite(R));
x=x(isfinite(R),:);
R=R(isfinite(R));


x=x(isfinite(A),:);
R=R(isfinite(A));
A=A(isfinite(A));

n=length(R);
for i=1:n
    if ~isempty(find(R([1:i-1,i+1:end])>=R(i) & A([1:i-1,i+1:end])<=A(i),1))
        R(i)=nan;
    end
end

[R,ia]=unique(R);
A=A(ia);
x=x(ia,:);

A=A(isfinite(R));
x=x(isfinite(R),:);
R=R(isfinite(R));


x=x(isfinite(A),:);
R=R(isfinite(A));
A=A(isfinite(A));

A=A(any(isfinite(x),2));
R=R(any(isfinite(x),2));
x=x(any(isfinite(x),2),:);
end
% Optimizer
function [Wx,A,R]=Optimiz(rSam0,xSam,ASam,rSam,x00,lb,ub,k,nn)
% rSam0 is the point of Constraint r>rSam0(l)
% ?Sam is a basic simulation to test the answer quality
% x00 is the starting points
% lb and ub are lower and upper bound
% k and nn are the number of asset and simulation size
global Beq
% rSam0=r;
L=length(rSam0);
% k=size(x,1);

R=nan(L,1);
A=R;
Wx=nan(L,k);

% Optimization
% we use global search instead of local ones
options = optimoptions(@fmincon,'Algorithm','sqp','ConstraintTolerance',10^-4,'display','off'); % this interior-point algorithm result was so good in the case of two asset model
% (Other available algorithms: 'active-set', 'sqp', 'trust-region-reflective', 'interior-point')
% First satring values

XfX0=-inf;
for l=1:L
    Beq= rSam0(L-l+1);
    x0=x00(:,L-l+1);
    xfinal0=x0;%xfinal00(:,i);
    for j=1:3
        problem = createOptimProblem('fmincon','objective',...
            @fEFO,'x0',x0,'lb',lb,'ub',ub,'options',options,'nonlcon',@fEFC);%,'ConstraintTolerance',10^-4);
        % x0 is the stating point, lb the lower bound and up the upper bound in this case nothing has to be changed by user
        % the lastinput is the shadow price or lagrange multiplier
        gs = GlobalSearch('NumStageOnePoints',1000,'NumTrialPoints',4000);
        disp(['Solving itration #' num2str(j)]);
        % xfinal is the optimum weght for asset one
        % XfX is the Valu of objective function
        [xfinal, XfX,exitflag] = run(gs,problem);
        xfinal(k)=100-sum(xfinal(1:k-1)); % Correct the Last Element
        if XfX<XfX0 || exitflag~=-2 || exitflag~=-8
            %             XfX=XfX0;
            %xfinal=xfinal0;
            %break;
            XfX0=XfX;
            xfinal0=xfinal;
        end
        % simulating for the next starting point
        [x,~,A0,r]=Simul(xfinal,nn);
        % Augment initail Sample
        x=[x,xSam,Wx(1:l-1,:).']; %#ok<AGROW>
        A0=[A0,ASam,A(1:l-1).']; %#ok<AGROW>
        r=[r,rSam,R(1:l-1).']; %#ok<AGROW>
        % Check whether the xfinal is better than the sampl
        XfX1=min(A0(r>=Beq & isfinite(A0)));%% Notice: it must change to r>Beq ---- r>Beq-Stp & r<Beq+Stp
        if isempty(XfX1)
            break;
        elseif XfX<=XfX1
            break;
        end
        
        [~,I]=find(A0==XfX1);
        
        x0=x(:,I(1));
    end
    [~,A(l),R(l)]=f(xfinal0);
    Wx(l,:)=xfinal0.';
    %home;
    disp('------------------------------------------------');
    disp([num2str(round(100*l/(L+1),1)) ' Percent is completed.']);
    disp('------------------------------------------------');
end
Wx(R<-10^9,:)=[];
A(R<-10^9)=[];
R(R<-10^9)=[];
end
%***********************************************  Efficiency Curve Biulder
function [Rt,ALPM,WeI]=findEC(xSam,ASam,rSam,k,nn,Resolution,CoverOlp)
disp('Optimization of Efficiency Curve is Started');
global r1
if nargin<7
    CoverOlp=0;
end
if nargin<6
    Resolution=nan;
end
[xSam,ASam,rSam]=refinery(xSam.',ASam.',rSam.');
xSam=xSam.'; ASam=ASam.'; rSam=rSam.';

rSam0=rSam;
L=length(rSam0);
if L<1
    warning('Bad Parametrization. No Efficeincy frontier Created.');
    return
end
rr=mean(r1);
rmin=min(rr);
rmax=max(rr);
%% assymetric point distribution
%r =rmin+random('beta',3,4,1,Resolution)*(rmax-rmin);%unique([fix(random('beta',1.5,8,1,3*Resolution)*L),1,L]);%betarnd(1.4,10,1,Resolution)*L));
%r =random('uni',rmin,rmax,1,Resolution);%unique([fix(random('beta',1.5,8,1,3*Resolution)*L),1,L]);%betarnd(1.4,10,1,Resolution)*L));
r=rmin:(rmax-rmin)/Resolution:rmax;
r(r<rmin | r>rmax)=[];

r=[rmin,r,rmax];
rSam0=unique(r);
% if length(r)>Resolution
%   nk=length(r)-Resolution;
%   r = r(1:end-nk);
% end

%rSam0=sort(r);

% rSam0=[rSam0(1:Stp:L-1),rSam0(L)];
% ia=[ia(1:Stp:L-1).',ia(L)];
%rSam0=rSam0(r);
L=length(rSam0);
x00=ones(k,L).*(100/k);
for l=1:L
    x00(:,l)=Reversef(rSam0(l));
end
% Rt=nan(L,1);
% ALPM=Rt;
% WeI=nan(L,k);

% for l=1:L
%     A=min(ASam(r(l)));
%     [~,I]=find(ASam==A);
%     x00(1:k-1,l)=xSam(1:k-1,I(1));
% end
lb=zeros(k,1);
ub=[100*ones(k-1,1);0];

[WeI,ALPM,Rt]=Optimiz(rSam0,xSam,ASam,rSam,x00,lb,ub,k,nn);

%%
%------------------ Find Oulier point
if CoverOlp==1
    beep
    home;
    disp('/\/\/\/\/\/\/\/\/\/\/\/\/\ Covering Oulier Point /\/\/\/\/\/\/\/\/\/\/\');
    [Outw,~,OutR]=Outer(Rt,ALPM,xSam.',ASam.',rSam.');
   % [~,~,OutR]=refinery(Outw,OutA,OutR);
    L=length(OutR);
    disp(['/*\ /*\ /*\ /*\ /*\ /*\ /*\ /*\ (' num2str(L) ') outlaw points found from ' num2str(length(rSam)) ' Sample . /*\ /*\ /*\ /*\ /*\ /*\ /*\ /*\']);
    if L>0
        x00=Outw.';%repmat(100/k,k,L);
        lb=zeros(k,1);
        ub=[100.*ones(k-1,1);0];
        
        [WeIA,ALPMA,RtA]=Optimiz(OutR.',xSam,ASam,rSam,x00,lb,ub,k,nn);
        
        WeI=[WeI;WeIA];
        ALPM=[ALPM;ALPMA];
        Rt=[Rt;RtA];
    end
end


%%
% refine data
[WeI,ALPM,Rt]=refinery(WeI,ALPM,Rt);
%home;
disp('**************%*********%********* efficieny Curve is completed. ****%********%**********');
end
