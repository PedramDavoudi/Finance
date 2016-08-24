function F5th
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
    alpha=Inp.alpha(s);
    beta=Inp.beta(s);
    Tau=Inp.Tau(s);
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
        Resolution=nn;
    end
     if isnan(CoverOut)
        CoverOut=0;
    end
    % find a reliable Sample
    [~,~,x,~,A,r]=OpO(k,nn);
    
    
    % Create Efficiency Curve
    if JustSimulation==0
        [Rt,ALPM,xFinal]=findEC(x,A,r,k,nn,Resolution,CoverOut);
    end
    
    
    PM=MPM();
   % plot Grapgh
    if exist('Rt','var')
        Expt(x,A,r,SenarioName,PM,Rt,ALPM,xFinal);% Export Simulation data to excell file
        Grph(JustSort,x,A,r,SenarioName,PM,Rt,ALPM,xFinal);% plot Graph
    else
        Expt(x,A,r,SenarioName);% Export Simulation data to excell file
        Grph(JustSort,x,A,r,SenarioName);% plot Graph
    end
    disp(['************ Senrio: ' SenarioName 'Was Completed. *****************']);
end
toc
end
% Data Exporting
function  Expt(x,A,r,SenarioName,PM,Rt,ALPM,xFinal)
global alpha b beta a c Tau

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
Fxl=[mat2dataset(x.','VarNames',Capx),dataset(A.',r.','VarNames',{'ALPM','Return'})];
export(Fxl,'xlsfile',['out\' SenarioName '\Siml.xlsx']);
% Exporting Simulation Data Efficiency Curve
if exist('Rt','var')
    Fxl=[mat2dataset(xFinal,'VarNames',Capx),dataset(Rt,ALPM,'VarNames',{'Return','ALPM'})];
    export(Fxl,'xlsfile',['out\' SenarioName '\EC.xlsx']);
end
clear Fxl
% Fxl=cell(k+1+6,2);

%Fxl(1:k,1)=Capx.';
%Fxl(1:k,2)=num2cell(xfinal);

%Fxl(k+1,1)={'ObjVal'};
%Fxl(k+1,2)={XfX};
Fxl=table();
Fxl.ParameterName={'a';'alpha';'b';'beta';'c';'Tau'};
Fxl.ParameterValue=[a;alpha;b;beta;c;Tau];
writetable(Fxl,['out\' SenarioName '\table.txt'])

end
% Graph Creater
function Grph(JustSort,x,A,r,SenarioName,PM,Rt,ALPM,xFinal)
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
    plot(ALPM,Rt,'g',PM.MV(:,1),PM.MV(:,2),'b',PM.CV(:,1),PM.CV(:,2),'r',PM.MAD(:,1),PM.MAD(:,2),'d');
end


title('Efficient Frontier');
ylabel('Portfo Expected Return');
xlabel('Portfo ALPM');
legend({'Simul','ALPM','Mean-Variance','CVaR','MAD'})
hold off
saveas(gcf,['out\' SenarioName '\EC.bmp'])
%close gcf
end
%$$$$$$$$$$$$$$$$$$$$$$$###### Post Modern portfo Managment
function [out]=MPM
% Define portfo 
global r1
p0 = Portfolio('assetmean', mean(r1,1).', 'assetcovar', r1.'*r1, 'lowerbound', 0, 'lowerbudget', 1, 'upperbudget', 1);
[rsk0,ret0]=p0.plotFrontier;
out.MV=[rsk0,ret0];
p1 = PortfolioCVaR('Scenarios', r1, 'LowerBound', 0, 'Budget', 1, 'ProbabilityLevel', 0.95);
[rsk1,ret1]=p1.plotFrontier;
out.CV=[rsk1,ret1];
p2 = PortfolioMAD('Scenarios', r1,'LowerBound', 0, 'LowerBudget', 1, 'UpperBudget', 1);
[rsk2,ret2]=plotFrontier(p2);
out.MAD=[rsk2,ret2];
close all
end
%####################### Sampler Objective Function
function [y,A,rB]=f(w1)
% check the weights illegal usage
% the number of degree of freedome is k-1
global r1 alpha b beta a c Tau

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
Pplus=1-pminus;
A= alpha*pminus*sum(LPM.^a)-b*Pplus*beta*sum(UMP.^c);

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
% Extreme Result ignorance
ASigma(ASigma<10^-4)=0.25;
%
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
function [c,ceq]=fEFC(w1)
% c is inequlity less than zero
% ceq is equality with zero

global r1 Beq%alpha b c a c Tau

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
%***********************************************  Sample Biulder
function [xfinal,XfX,x,y,A,r]=OpO(k,nn)
% this function just replicate a good sample
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
disp('************************ Sampling Done *************************');
end
%***********************************************  Find outlier data
function [OutX,OutA,OutR]=Outer(Rt,ALPM,xSam,ASam,rSam)
%% Points Inside Convex Polygon
%%
% Define the x and y coordinates of polygon vertices to create a pentagon.
yv=Rt;
xv=ALPM;
%Completed the curve
yv(end+1)=-inf;
xv(end+1)=inf;
yv(end+1)=inf;
xv(end+1)=inf;
% Sort data
[yv,I]=sort(yv);
xv=xv(I);

% Define x and y coordinates of Simulation data
yq=rSam;
xq=ASam;

%%
% Determine whether each point lies inside or on the edge of the polygon
% area. Also determine whether any of the points lie on the edge of the
% polygon area.
[in,on] = inpolygon(xq,yq,xv,yv);
OutA=xq(~(in | on));
OutR=yq(~(in | on));
OutX=xSam(~(in | on),:);
% refine data
OutA=OutA(isfinite(OutR));
OutX=OutX(isfinite(OutR),:);
OutR=OutR(isfinite(OutR));
[OutR,ia]=unique(OutR);
OutA=OutA(ia);
OutX=OutX(ia,:);
%%
end

%***********************************************  Efficiency Curve Biulder
function [Rt,ALPM,WeI]=findEC(xSam,ASam,rSam,k,nn,Resolution,CoverOlp)
disp('Optimization of Efficiency Curve is Started');
global Beq
if nargin<7
    CoverOlp=0;
end
if nargin<6
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
%%
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
% (Other available algorithms: 'active-set', 'sqp', 'trust-region-reflective', 'interior-point')'UseParallel', true, 
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
%%
%------------------ Find Oulier point
if CoverOlp==1
    beep
    home;
    disp('/\/\/\/\/\/\/\/\/\/\/\/\/\ Covering Oulier Point /\/\/\/\/\/\/\/\/\/\/\');
    [OutX,~,OutR]=Outer(Rt,ALPM,xSam.',ASam.',rSam.');
    rSam0=OutR.';
    rSam0=[0,rSam0];
    L=length(rSam0);
    disp(['/*\ /*\ /*\ /*\ /*\ /*\ /*\ /*\ (' num2str(L) ') outlaw points found. /*\ /*\ /*\ /*\ /*\ /*\ /*\ /*\']);
    %A=OutA.';
    x00=[repmat(0.5,k,1),OutX.'];
    
    RtA=nan(L,1);
    ALPMA=RtA;
    WeIA=nan(L,k);
    
    lb=[zeros(k-1,1);0];
    ub=[ones(k-1,1);0];
    
    % Optimization
    % we use global search instead of local ones
   % options = optimoptions(@fmincon,'UseParallel', true, 'UseCompletePoll', true, 'UseVectorized', false,'Algorithm','sqp','ConstraintTolerance',10^-4); % this interior-point algorithm result was so good in the case of two asset model
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
        [~,ALPMA(l),RtA(l)]=f(xfinal0);
        WeIA(l,:)=xfinal0.';
        home;
        disp([num2str(100*l/(L+1)) ' Percent is completed.']);
    end
    WeIA(Rt<-10^9,:)=[];
    ALPMA(Rt<-10^9)=[];
    RtA(Rt<-10^9)=[];
    
    WeI=[WeI;WeIA];
    ALPM=[ALPM;ALPMA];
    Rt=[Rt;RtA];
end
% refine data
ALPM=ALPM(isfinite(Rt));
WeI=WeI(isfinite(Rt),:);
Rt=Rt(isfinite(Rt));
[Rt,ia]=unique(Rt);
ALPM=ALPM(ia);
WeI=WeI(ia,:);
%%
home;
disp('**************%*********%********* efficieny Curve is completed. ****%********%**********');
end
