function TestALPM
% al & bt must be grater than one.. according to paper
% this code drop simulating around optimum point
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
Inp=dataset('xlsfile','Input\SenariosTest');
S=size(Inp,1);

% loading data
r0=dataset('xlsfile','Input\dataTe');
Asset_Names=r0.Properties.VarNames(2:end);
r0 = sortrows(r0,'date','ascend');
r0=double(r0);
r0(any(isnan(r0),2),:)=[]; % Remove nan frome the data
Dates=r0(:,1);
r0(:,1)=[];
%rB=1.35;
n =size(r0,1);

for s=1:S
    SenarioName=Inp.SenarioName{s};
    a=Inp.a(s);
    b=Inp.b(s);
    c=Inp.c(s);
    alpha=Inp.alpha(s);
    beta=Inp.beta(s);
    Tau=Inp.Tau(s);
    WindowsSize=Inp.WindowsSize(s);
    DropDominated=Inp.DropDominated(s);
    JustSimulation=Inp.JustSimulation(s);
    CoverOut=Inp.CoverOut;
    
    nn=Inp.SampleSize(s);
    Resolution=Inp.Resolution(s);
    if isnan(WindowsSize) || WindowsSize<1
        WindowsSize=n;
    end
    if isnan(DropDominated)
        DropDominated=1;
    end
    
    
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
    for d=min(Dates):WindowsSize:max(Dates)
        r1=r0(Dates>=d & Dates<d+WindowsSize,:);
        
        % Asset_NamesDroped=DomAsset(Asset_Names,DropDominated,[SenarioName '_Dates' num2str(d) 'To' num2str(d+WindowsSize)]);
        
        k =size(r1,2);
        %find a reliable Sample
        x0=0:0.01:1;
        x00=[x0',1-x0'];
        
        for i=1:size(x00,1)
           [ S00(i),r00(i)]= ALPMCal(r1*x00(i,:).');
        
          % Tau(i)=mean( r1*x00(i,:).');
                   
        end
%         plot3(r00,x0',1-x0','r',S00,x0',1-x0','g')
    
figure;
        %plot(x0,r00,'r',x0,S00,'g')
         plot(S00,r00,'b')
         title(SenarioName);
        disp(['Elapsed Time is: ' datestr(toc/(24*3600), 'HH:MM:SS')]);
        
    end
end


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
    if isempty(NumPoint)
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

%####################### ALPM Calculator
function [A,rB]=ALPMCal(r)
% check the weights illegal usage
% the number of degree of freedome is k-1
global Tau alpha a b c  beta
[n,~] =size(r);

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
LPM(LPM==0)=[];
UMP(UMP==0)=[];
if isempty(LPM)
    LPM=0;
end
if isempty(UMP)
    UMP=0;
end
A= alpha*pminus*mean(LPM.^a)-b*Pplus*beta*mean(UMP.^c);

rB=mean(r);
%ObjectiveFunction
end
