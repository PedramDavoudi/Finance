function F5th
% al & bt must be grater than one.. according to paper
tic;
global r1 a b c al bt Tau
% Inputs
% a=0.1;
% b=0.2;
% c=0.1;
% al=1.5;
% bt=3;
% Tau=nan; % the treshold valu, nan= mean of data
% nn=20; % the number of samples to plot the graphs
%
% Resolution=1;
JustSimulation=0;
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
    al=Inp.alpha(s);
    bt=Inp.beta(s);
    Tau=Inp.Tau(s);
    nn=Inp.SampleSize(s);
    Resolution=Inp.Resolution(s);
    if ~(isfinite(a) && isfinite(b) &&isfinite(c) && isfinite(al) && isfinite(bt))
        warning('not ')
        continue;
    end
    if isempty(SenarioName)
        SenarioName=['Sen' num2str(s)];
    end
    if isnan(Resolution)
        Resolution=nn;
    end
    % find a reliable Sample
    [~,~,x,~,A,r]=OpO(k,nn);
    
    
    % Create Efficiency Curve
    if JustSimulation==0
        [Rt,ALPM,xFinal]=findEC(x,A,r,k,nn,Resolution);
    end
    
    
    
    % plot Grapgh
    if exist('Rt','var')
        Expt(x,A,r,SenarioName,Rt,ALPM,xFinal);% Export Simulation data to excell file
        Grph(JustSort,x,A,r,SenarioName,Rt,ALPM,xFinal);% plot Graph
    else
        Expt(x,A,r,SenarioName);% Export Simulation data to excell file
        Grph(JustSort,x,A,r,SenarioName);% plot Graph
    end
    disp(['************ Senrio: ' SenarioName 'Was Completed. *****************']);
end
toc
end
% Data Exporting
function  Expt(x,A,r,SenarioName,Rt,ALPM,xFinal)
global a b c al bt Tau

if ~exist('SenarioName','var')
    SenarioName='Sen';
end
if ~exist('out','dir')
    mkdir('out')
end
if ~exist(['out\' SenarioName],'dir')
    mkdir(['out\' SenarioName])
end

k=size(x,1);
Capx=cell(1,k);

for i=1:k
    Capx{i}=['w' num2str(i)];
end
% Exporting Simulation Data
Fxl=[mat2dataset(x.','VarNames',Capx),dataset(A.',r.','VarNames',{'ALPM','Rerutn'})];
export(Fxl,'xlsfile',['out\' SenarioName '\Siml.xlsx']);
% Exporting Simulation Data Efficiency Curve
if exist('Rt','var')
    Fxl=[mat2dataset(xFinal,'VarNames',Capx),dataset(Rt,ALPM,'VarNames',{'ERetrun','ALPM'})];
    export(Fxl,'xlsfile',['out\' SenarioName '\EC.xlsx']);
end
clear Fxl
% Fxl=cell(k+1+6,2);

%Fxl(1:k,1)=Capx.';
%Fxl(1:k,2)=num2cell(xfinal);

%Fxl(k+1,1)={'ObjVal'};
%Fxl(k+1,2)={XfX};
Fxl=table();
Fxl.ParameterName={'a';'b';'c';'al';'bt';'Tau'};
Fxl.ParameterValue=[a;b;c;al;bt;Tau];
writetable(Fxl,['out\' SenarioName '\tabledata.txt'])

end
% Graph Creater
function Grph(JustSort,x,A,r,SenarioName,Rt,ALPM,xFinal)
if ~exist('SenarioName','var')
    SenarioName='Sen';
end
k=size(x,1);
% plot Graph
for i=1:k
    figure();
    hold on
    [x0r,r0] =Xfine(x(i,:),r,1); % Just Sorted and not refined in any case
    plot(x0r,r0,'c . ');
    if exist('Rt','var')
        [x0y,y0]=Xfine(xFinal(:,i),Rt,1);
        plot(x0y,y0,'g');
    end
    %plot(xfinal(i), XfX,'r O')
    title('Portfo Analyze');
    xlabel(['Weight of Asset #' num2str(i)]);
    legend({'Return','Optimum Point in Return'})%,})
    hold off
    saveas(gcf,['out\' SenarioName '\wR' num2str(i) '.bmp'])
    close gcf
    %------------------------------------------------------
        figure();
    hold on
   [x0A,A0]=Xfine(x(i,:),A,JustSort);
    plot(x0A,A0,'b . ');
    if exist('Rt','var')
        [x1y,y1]=Xfine(xFinal(:,i),ALPM,1);
        plot(x1y,y1,'g');
    end
    %plot(xfinal(i), XfX,'r O')
    title('Portfo Analyze');
    xlabel(['Weight of Asset #' num2str(i)]);
    legend({'ALPM','Optimum Point in ALPM'})%,})
    hold off
    saveas(gcf,['out\' SenarioName '\wA' num2str(i) '.bmp'])
    close gcf
end
% efficient frontier
[r,A]=Xfine(r,A,JustSort);
% [sortedr,I] = sort(r);
figure();
hold on
%[~,A0,r0]=f(xfinal);
plot(A,r,'b . ');%,A0, r0,'r O');
if exist('Rt','var')
    plot(ALPM,Rt,'g');
end
title('Efficient Frontier');
ylabel('Portfo Expected Return');
xlabel('Portfo ALPM');
%legend({'R/R','Optimum Point'})
hold off
saveas(gcf,['out\' SenarioName '\EC.bmp'])
%close gcf
end
%$$$$$$$$$$$$$$$$$$$$$$$######
%####################### Sampler Objective Function
function [y,A,rB]=f(w1)
% check the weights illegal usage
% the number of degree of freedome is k-1
global r1 a b c al bt Tau

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
if ~isfinite(Tau)
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
ASigma=xfinal./4;%repmat(0.02,k,1)
ASigma(xfinal>0.5)=(1-xfinal(xfinal>0.5))./4;
Sigma=diag(ASigma.^2);
x=mvnrnd(xfinal,Sigma,nn).';%MU,SIGMA,n
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

global r1 Beq%a b c al bt Tau

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
%***********************************************  Sample Biulder
function [xfinal,XfX,x,y,A,r]=OpO(k,nn)
% this function just replicate a good sample
% Optimization
disp('Optimization of Objective Function To Build qualified Sample Started');
% we use global search instead of local ones
options = optimoptions(@fmincon,'Algorithm','sqp'); % this interior-point algorithm result was so good in the case of two asset model
% (Other available algorithms: 'active-set', 'sqp', 'trust-region-reflective', 'interior-point')
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
disp('************************ Sampling Done *************************');
end
%{
function [Rt,ALPM]=findEC0(xSam,ASam,rSam,k,nn,Resolution)
disp('Optimization of Efficiency Curve is Started');
if nargin<3
    Resolution=10;
end
[x,y]=Xfine(xSam,rSam,0);
lb=[zeros(k-1,1);0];
ub=[ones(k-1,1);0];
%[~,~,~,r,~]=Simul(x0,nn);
MinR=min(rSam(isfinite(rSam)))*(1-0.2);
MaxR=max(rSam(isfinite(rSam)))*(1+0.2);
Stp=(MaxR-MinR)/Resolution;
fI=min(ASam(rSam>MinR-Stp & rSam<MinR+Stp & isfinite(ASam)));
if isempty(fI)
    x0=(1/k).*ones(k,1);
else
    x0=xSam(:,ASam==fI);
    x0=x0(:,1);
end

global Beq
Rt=nan(Resolution+2,1);
ALPM=Rt;
% Optimization
% we use global search instead of local ones
options = optimoptions(@fmincon,'Algorithm','sqp','ConstraintTolerance',10^-4); % this interior-point algorithm result was so good in the case of two asset model
% (Other available algorithms: 'active-set', 'sqp', 'trust-region-reflective', 'interior-point')
% First satring values
i=0;
XfX0=-inf;
for Beq=MinR:Stp:MaxR
    xfinal0=ones(k,1);
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
        % Augment initail Sample
        x=[x,xSam];
        A=[A,ASam];
        r=[r,rSam];
        % Check whether the xfinal is better than the sampl
        XfX1=min(A(r>Beq-Stp & r<Beq+Stp & isfinite(A)));
        if isempty(XfX1)
            break;
        elseif XfX<=XfX1
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
%}
%***********************************************  Efficiency Curve Biulder
function [Rt,ALPM,WeI]=findEC(xSam,ASam,rSam,k,nn,Resolution)
disp('Optimization of Efficiency Curve is Started');
global Beq
if nargin<3
    Resolution=nan;
end
xSam(:,~isfinite(rSam))=[];
ASam(:,~isfinite(rSam))=[];
rSam(:,~isfinite(rSam))=[];
[rSam0,ia]=unique(rSam);
L=length(rSam0);
if L<1
    warning('Bad Parametrization. No Efficeincy frontier Created.');
    return
end
if isnan(Resolution) || Resolution>L-1
    Resolution=L;
end
if Resolution>1
    Stp=fix(L/(Resolution-1));
else
    Stp= 1;
end
rSam0=[rSam0(1:Stp:L-1),rSam0(L)];
ia=[ia(1:Stp:L-1).',ia(L)];
L=length(rSam0);
x00=zeros(k,L);
Rt=nan(L,1);
ALPM=Rt;
WeI=nan(L,k);

for l=1:L
    A=min(ASam(ia(l)));
    [~,I]=find(ASam==A);
    x00(1:k-1,l)=xSam(1:k-1,I(1));
end
lb=[zeros(k-1,1);0];
ub=[ones(k-1,1);0];

% Optimization
% we use global search instead of local ones
options = optimoptions(@fmincon,'Algorithm','sqp','ConstraintTolerance',10^-4); % this interior-point algorithm result was so good in the case of two asset model
% (Other available algorithms: 'active-set', 'sqp', 'trust-region-reflective', 'interior-point')
% First satring values

XfX0=-inf;
Stp=10^-4;
for l=1:L
    Beq= rSam0(l);
    x0=x00(:,l);
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
        xfinal(k)=1-sum(xfinal(1:k-1)); % Correct the Last Element
        if XfX<XfX0 || exitflag~=-2 || exitflag~=-8
            %             XfX=XfX0;
            %xfinal=xfinal0;
            %break;
            XfX0=XfX;
            xfinal0=xfinal;
        end
        % simulating for the next starting point
        [x,~,A,r]=Simul(xfinal,nn);
        % Augment initail Sample
        x=[x,xSam]; %#ok<AGROW>
        A=[A,ASam]; %#ok<AGROW>
        r=[r,rSam]; %#ok<AGROW>
        % Check whether the xfinal is better than the sampl
        XfX1=min(A(r>Beq-Stp & r<Beq+Stp & isfinite(A)));
        if isempty(XfX1)
            break;
        elseif XfX<=XfX1
            break;
        end
        
        [~,I]=find(A==XfX1);
        
        x0=x(:,I(1));
    end
    [~,ALPM(l),Rt(l)]=f(xfinal0);
    WeI(l,:)=xfinal0.';
    home;
    disp([num2str(100*l/(L+1)) ' Percent is completed.']);
end
WeI(Rt<-10^9,:)=[];
ALPM(Rt<-10^9)=[];
Rt(Rt<-10^9)=[];
home;
disp('**************%*********%********* efficieny Curve is completed. ****%********%**********');
end