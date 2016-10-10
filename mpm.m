clc
clear
S=[0,0;0,10000];
r=[21;15];
rB=20;

% 
% x0=0:0.01:1;
% x00=[x0',1-x0'];
% plot3(x00*r>=rB,x0',1-x0')
% hold on
% for i=1:size(x00,1)
%    S00(i)= x00(i,:)*S*x00(i,:)';
% end
% plot3(S00,x0',1-x0')
% hold off
% plot(x0,x00*r>=rB,'r',x0,S00)

n=length(r);
I=ones(n,1);
%%
%{
W=sym('W',[1,n]);
assume(W>=0);
%assume(sum(W)==1);
%assume(W*r>=rB);
lnd=sym('lnd',[1,n]);
assume(lnd>=0);

 Prb=[(2*S*W.'-lnd(1)*r+lnd(2)*I);lnd(1)*(W*r-rB);(W*I-1)];
%Prb=[(2*S.'*W.'-lnd(1)*r+lnd(2)*I);lnd(1)*(W*r-rB);(W*I-1)];
%Prb=[W.'.*(2*S*W.'-lnd(1)*r);(W*r-rB);(W*I-1)];
%Prb=[W.'.*(2*S*W.'+lnd(1)*r-lnd(2)*I);(W*r-rB);(W*I-1)];
%Prb=[(2*S*W.'-lnd(1)*r-lnd(2)*I)];
pretty(Prb);
aa=solve(Prb);

W0=[double(aa.W1),double(aa.W2)]
% R=W0*r;
% lmdd=[double(aa.lnd1),double(aa.lnd2)]
% W0=W0(1,:);
% R=W0*r;%-rB

%SS=W0*S*W0.';
%}
%%
%fun = @(w)w.'*S*w;
% fmincon(fun,'x0',I/n,'A',r.','b',rB,'Aeq',I','beq',1,'lb',zeros(n,1))
fmincon(@(w) w.'*S*w,I/n,-r.',-rB,I',1,zeros(n,1),[])%'display','off'