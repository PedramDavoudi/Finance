%% Points Inside Convex Polygon
% Define a pentagon and a set of points. Then, determine which points lie
% inside (or on the edge) of the pentagon.

% Copyright 2015 The MathWorks, Inc.


%%
% Define the x and y coordinates of polygon vertices to create a pentagon.
L = linspace(0,2.*pi,12000);
%xv = cos(L)';
%yv = sin(L)';
% xv=2*rand(5,1);
% yv=2*rand(5,1);

A=dataset('xlsfile','Ex2\EC.xlsx');
yv=A.ERetrun;
xv=A.ALPM;
%a0=yv(xv==min(xv));
yv(end+1)=-inf;
xv(end+1)=inf;
yv(end+1)=inf;
xv(end+1)=inf;
[yv,I]=sort(yv);
xv=xv(I);

B=dataset('xlsfile','Ex2\Siml.xlsx');
yq=B.Rerutn;
xq=B.ALPM;
%%x
% Define x and y coordinates of 250 random query points. Initialize the
% random-number generator to make the output of |randn| repeatable.
% rng default
% xq = randn(250,1);
% yq = randn(250,1);


%%
% Determine whether each point lies inside or on the edge of the polygon
% area. Also determine whether any of the points lie on the edge of the
% polygon area.
[in,on] = inpolygon(xq,yq,xv,yv);
Out=[xq(~(in | on)),yq((~in | on))];
%%
% Determine the number of points lying inside or on the edge of the polygon area.
numel(xq(in))

%%
% Determine the number of points lying on the edge of the polygon area.
numel(xq(on))

%%
% Since there are no points lying on the edge of the polygon area, all 80
% points identified by |xq(in)|, |yq(in)| are strictly inside the polygon area.

%%
% Determine the number of points lying outside the polygon area (not inside or on the edge).
numel(xq(~in))

%%
% Plot the polygon and the query points. Display the points inside the
% polygon with a red plus. Display the points outside the polygon with a
% blue circle.
figure

plot(xv,yv) % polygon
%axis equal

hold on
plot(xq(in),yq(in),'r+') % points inside
plot(xq(~in),yq(~in),'bo') % points outside
hold off


