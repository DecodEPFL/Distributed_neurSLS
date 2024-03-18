clear all
close all
clc


load("dataTrainedSystem_REN.mat")

animation = 1;

p1x = x_log(:,1:4:end);
p1y = x_log(:,2:4:end);
p2x = x_log(:,5:4:end);
p2y = x_log(:,6:4:end);
p3x = x_log(:,9:4:end);
p3y = x_log(:,10:4:end);
p4x = x_log(:,13:4:end);
p4y = x_log(:,14:4:end);

obs = [-2.25 0;
        -3.5 0;
        3.5 0;
       2.25 0];
robs = 0.5;

obse = [-4 0;
        4 0];

simTime = size(x_log,1);
Ts = 0.01;
tspan = 0:Ts:simTime*Ts;

n_agents = 4;
n = 4;

c1 = [0 0.4470 0.7410];
c2 = [0.8500 0.3250 0.0980];
c3 = [0.4660 0.6740 0.1880];
c4 = [0.6350 0.0780 0.1840];
grey = [0.749 0.749 0.749];
alpha = 0.3;

ax = [-5.6 5.6 -5.6 5.6];


for i = 1:3
    figure(i)
    for nobs = 1:size(obse,1)
        hold on;
        [xe, ye, ze] = ellipsoid(obse(nobs,1),obse(nobs,2),0,2.25,1,70);
        surf(xe, ye, ze,'FaceAlpha',0.2,'EdgeAlpha',0,'EdgeColor','none');
        %circle2(obs(nobs,1),obs(nobs,2),robsi,'k'); 
        colormap winter;
    end

end

tend = size(x_log,1);

%%%% FIGURE 1
figure(1);
plot(p1x,p1y,'Color',grey); hold on;
plot(p2x,p2y,'Color',grey); 
plot(p3x,p3y,'Color',grey); 
plot(p4x,p4y,'Color',grey); 

time = 5;
plot(p1x(1:time,:),p1y(1:time,:),'Color',c1,'LineWidth',1);
plot(p2x(1:time,:),p2y(1:time,:),'Color',c2,'LineWidth',1); 
plot(p3x(1:time,:),p3y(1:time,:),'Color',c3,'LineWidth',1); 
plot(p4x(1:time,:),p4y(1:time,:),'Color',c4,'LineWidth',1); 
scatter(xbar(1),xbar(2),70,"pentagram","filled",'MarkerFaceColor',c1)
scatter(xbar(5),xbar(6),70,"pentagram","filled",'MarkerFaceColor',c2)
scatter(xbar(9),xbar(10),70,"pentagram","filled",'MarkerFaceColor',c3)
scatter(xbar(13),xbar(14),70,"pentagram","filled",'MarkerFaceColor',c4)
scatter(p1x(1,1),p1y(1,1),70,"o",'MarkerEdgeColor',c1)
scatter(p2x(1,1),p2y(1,1),70,"o",'MarkerEdgeColor',c2)
scatter(p3x(1,1),p3y(1,1),70,"o",'MarkerEdgeColor',c3)
scatter(p4x(1,1),p4y(1,1),70,"o",'MarkerEdgeColor',c4)

drawcirclef(0.5,[p1x(time) p1y(time)],c1,alpha)
drawcirclef(0.5,[p2x(time) p2y(time)],c2,alpha)
drawcirclef(0.5,[p3x(time) p3y(time)],c3,alpha)
drawcirclef(0.5,[p4x(time) p4y(time)],c4,alpha)
plot([p1x(time);p2x(time)],[p1y(time);p2y(time)],'k')
plot([p2x(time);p3x(time)],[p2y(time);p3y(time)],'k')
plot([p3x(time);p4x(time)],[p3y(time);p4y(time)],'k')
plot([p4x(time);p1x(time)],[p4y(time);p1y(time)],'k')

axis(ax)
view(0,90)
%set(gca,'YTickLabel',[],'XTickLabel',[]);

%%%% FIGURE 2

figure(2);
plot(p1x,p1y,'Color',grey); hold on;
plot(p2x,p2y,'Color',grey); 
plot(p3x,p3y,'Color',grey); 
plot(p4x,p4y,'Color',grey); 

time = tend/4;
plot(p1x(1:time,:),p1y(1:time,:),'Color',c1,'LineWidth',1);
plot(p2x(1:time,:),p2y(1:time,:),'Color',c2,'LineWidth',1); 
plot(p3x(1:time,:),p3y(1:time,:),'Color',c3,'LineWidth',1); 
plot(p4x(1:time,:),p4y(1:time,:),'Color',c4,'LineWidth',1); 
scatter(xbar(1),xbar(2),70,"pentagram","filled",'MarkerFaceColor',c1)
scatter(xbar(5),xbar(6),70,"pentagram","filled",'MarkerFaceColor',c2)
scatter(xbar(9),xbar(10),70,"pentagram","filled",'MarkerFaceColor',c3)
scatter(xbar(13),xbar(14),70,"pentagram","filled",'MarkerFaceColor',c4)
scatter(p1x(1,1),p1y(1,1),70,"o",'MarkerEdgeColor',c1)
scatter(p2x(1,1),p2y(1,1),70,"o",'MarkerEdgeColor',c2)
scatter(p3x(1,1),p3y(1,1),70,"o",'MarkerEdgeColor',c3)
scatter(p4x(1,1),p4y(1,1),70,"o",'MarkerEdgeColor',c4)

drawcirclef(0.5,[p1x(time) p1y(time)],c1,alpha)
drawcirclef(0.5,[p2x(time) p2y(time)],c2,alpha)
drawcirclef(0.5,[p3x(time) p3y(time)],c3,alpha)
drawcirclef(0.5,[p4x(time) p4y(time)],c4,alpha)
plot([p1x(time);p2x(time)],[p1y(time);p2y(time)],'k')
plot([p2x(time);p3x(time)],[p2y(time);p3y(time)],'k')
plot([p3x(time);p4x(time)],[p3y(time);p4y(time)],'k')
plot([p4x(time);p1x(time)],[p4y(time);p1y(time)],'k')

axis(ax)
view(0,90)
%set(gca,'YTickLabel',[],'XTickLabel',[]);

% FIGURE 3
figure(3);

plot(p1x,p1y,'Color',grey); hold on;
plot(p2x,p2y,'Color',grey); 
plot(p3x,p3y,'Color',grey); 
plot(p4x,p4y,'Color',grey); 

time = tend;
plot(p1x(1:time,:),p1y(1:time,:),'Color',c1,'LineWidth',1);
plot(p2x(1:time,:),p2y(1:time,:),'Color',c2,'LineWidth',1); 
plot(p3x(1:time,:),p3y(1:time,:),'Color',c3,'LineWidth',1); 
plot(p4x(1:time,:),p4y(1:time,:),'Color',c4,'LineWidth',1); 
scatter(xbar(1),xbar(2),70,"pentagram","filled",'MarkerFaceColor',c1)
scatter(xbar(5),xbar(6),70,"pentagram","filled",'MarkerFaceColor',c2)
scatter(xbar(9),xbar(10),70,"pentagram","filled",'MarkerFaceColor',c3)
scatter(xbar(13),xbar(14),70,"pentagram","filled",'MarkerFaceColor',c4)
scatter(p1x(1,1),p1y(1,1),70,"o",'MarkerEdgeColor',c1)
scatter(p2x(1,1),p2y(1,1),70,"o",'MarkerEdgeColor',c2)
scatter(p3x(1,1),p3y(1,1),70,"o",'MarkerEdgeColor',c3)
scatter(p4x(1,1),p4y(1,1),70,"o",'MarkerEdgeColor',c4)


drawcirclef(0.5,[p1x(time) p1y(time)],c1,alpha)
drawcirclef(0.5,[p2x(time) p2y(time)],c2,alpha)
drawcirclef(0.5,[p3x(time) p3y(time)],c3,alpha)
drawcirclef(0.5,[p4x(time) p4y(time)],c4,alpha)
plot([p1x(time);p2x(time)],[p1y(time);p2y(time)],'k')
plot([p2x(time);p3x(time)],[p2y(time);p3y(time)],'k')
plot([p3x(time);p4x(time)],[p3y(time);p4y(time)],'k')
plot([p4x(time);p1x(time)],[p4y(time);p1y(time)],'k')


axis(ax)
view(0,90)
%set(gca,'YTickLabel',[],'XTickLabel',[]);








%%
if animation 
    figure;
    for nobs = 1:size(obse,1)
        hold on;
        [xe, ye, ze] = ellipsoid(obse(nobs,1),obse(nobs,2),0,2.25,1,70);
        surf(xe, ye, ze,'FaceAlpha',0.2,'EdgeAlpha',0,'EdgeColor','none');
        %circle2(obs(nobs,1),obs(nobs,2),robsi,'k'); 
        colormap winter;
    end
    view(0,90)
    
    hold on;
    XX = x_log';
    y = 1;
    axis([-6 6 -8 8])
    p1 = scatter(XX(1,1),XX(2,1));p2 =scatter(XX(5,1),XX(6,1));
    p3 =scatter(XX(9,1),XX(10,1));p4 =scatter(XX(13,1),XX(14,1));
    p7 =scatter(XX(9,1),XX(10,1));p8 =scatter(XX(13,1),XX(14,1));
    p9 =scatter(XX(9,1),XX(10,1));p10 =scatter(XX(13,1),XX(14,1));
    traj1 = plot(XX(1,1),XX(2,1));traj2 =plot(XX(5,1),XX(6,1));
    traj3 =plot(XX(9,1),XX(10,1));traj4 =plot(XX(13,1),XX(14,1));
    for t = 1:20:length(tspan)-1
        delete(p1);delete(p2);delete(p3);delete(p4);
        delete(p7);delete(p8);delete(p9);delete(p10);
        delete(traj1);delete(traj2);delete(traj3);delete(traj4);
        p1 = circle2(XX(1,t),XX(2,t),0.5,'r');
        p2=circle2(XX(5,t),XX(6,t),.500,'b');
        p3=circle2(XX(9,t),XX(10,t),.500,'g');
        p4=circle2(XX(13,t),XX(14,t),.500,[0.4660 0.6740 0.1880]);
        p7 = plot([XX(1,t);XX(5,t)],[XX(2,t);XX(6,t)],'k');
        p8 = plot([XX(5,t);XX(9,t)],[XX(6,t);XX(10,t)],'k');
        p9 = plot([XX(9,t);XX(13,t)],[XX(10,t);XX(14,t)],'k');
        p10 = plot([XX(13,t);XX(1,t)],[XX(14,t);XX(2,t)],'k');
    
        traj1 = plot(XX(1,1:t),XX(2,1:t),'r-');
        traj2 = plot(XX(5,1:t),XX(6,1:t),'b--');
        traj3 = plot(XX(9,1:t),XX(10,1:t),'g-.');
        traj4 = plot(XX(13,1:t),XX(14,1:t),'--','Color',[0.4660 0.6740 0.1880]);
        %title (sprintf('time %d: ',t*param.Ts))
        title (sprintf('time %d: ',t))
        pause(0.5);
    end


end




%circle(0.5,[p1x(1) p1y(1)],[0 0.4470 0.7410])
function drawcirclef(r,c,Color,alpha)
%// number of points
n = 1000;
%// running variable
t = linspace(0,2*pi,n);

x = c(1) + r*sin(t);
y = c(2) + r*cos(t);

%// draw filled polygon
fill(x,y,[1,1,1],'FaceColor',Color,'EdgeColor',Color,'facealpha',alpha)
axis equal

end

function h = circle(r,xy,col)
x = xy(1); y = xy(2);
th = 0:pi/50:2*pi;
xunit = r * cos(th) + x;
yunit = r * sin(th) + y;
h = plot(xunit, yunit,"--",'Color',col);
end


function h = circle2(x,y,r,colr)
hold on
th = 0:pi/50:2*pi;
xunit = r * cos(th) + x;
yunit = r * sin(th) + y;
h = plot(xunit, yunit,'Color',colr);
end