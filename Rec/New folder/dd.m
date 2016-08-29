function [xfinal, XfX]=dd()

global r1 r2 ;
r1=1;
r2=2;
options = optimoptions('fminunc','Display','final','Algorithm','quasi-newton', 'OptimalityTolerance',10^-20,'MaxFunctionEvaluations',1000);
%fh2 = matlabFunction(f,'vars',{x});




StartingPoint=100;
%fh2 = objective with no gradient or Hessian

%[xfinal,fval,exitflag,output2] = fminunc(@f,StartingPoint,options);

%
options = optimoptions(@fmincon,'Algorithm','interior-point');

problem = createOptimProblem('fmincon','objective',...
    @f,'x0',StartingPoint,'lb',-1000,'ub',+1000,'options',options); %
gs = GlobalSearch;
disp('Solving started');
[xfinal, XfX] = run(gs,problem)
%}
end
function y=f(x)
global r1 r2 ;
if r1>r2
    y=r1*x^2+x;
else
    y=r2*x^2+x;
end
% y=max(x,0);
end

function y=f1(x)
if x<=1
    y=2;
elseif x>1 && x<10
    y=1;
elseif x>10 && x<10.1
    y=-1;
else
    y=3;
end
% y=max(x,0);
end

function y=f0(x)
if x<1
    y=3*x^2-3*x+6;
elseif x>1 || x<10
    y=3*x^2-6*x+6;
else
    y=3;
end
end