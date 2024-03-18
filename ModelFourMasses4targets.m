
clear all
close all
clc

animatedplot = 1;

[param.m1, param.m2, param.m3, param.m4] = deal(1);
[param.k1, param.k2, param.k3, param.k4] = deal(1);
[param.k5, param.k6, param.k7, param.k8, param.k9, param.k10] = deal(1);
[param.c1, param.c2, param.c3, param.c4] = deal(1);
[param.c5, param.c6, param.c7, param.c8, param.c9, param.c10] = deal(1);
% TO BE MODIFIED
xbar = [-2; 5; 0; 0;
         2; 5; 0;0;
         2; 3.5;0;0;
         -2; 3.5;0;0];

initial_state = [-2; -3.5; 0; 0;
                 2; -3.4; 0;0;
                 2;-4.9;0;0;
                 -2;-5;0;0];

initial_state = [-1.5; -3.5; 0.01; 0.02;
                 2.5; -3.4; 0.02;0.01;
                 3;-4.9;0.01;0.02;
                 -4;-5;0.01;0.02];


%xbar = initial_state;

param.xbar = xbar;

obs = [-1.5 0 
       -2.5 0
       1.5 0
       2.5 0];
robs = 0.5;


param.Ts = 0.01;
param.nx = 16;
param.nu = 8;

inputmax = 20;

num_exp = 1;

simTime = 1500

for exp = 1:num_exp+1

    % Define simulation parameters
    tspan = 0:param.Ts:simTime*param.Ts; % Time span from 0 to 5 with a time step of 0.01
    input_signal = zeros(param.nu,length(tspan));
    input_signal = zeros*randn(param.nu,length(tspan)); % Exogenous input signal (example: sinusoidal input) 
    %  input_signal(2,30:33) = -8;
    % input_signal(4,30:33) = -8;
    %  input_signal(6,30:33) = 8;
    %  input_signal(8,30:33) = 8;
    %input_signal(1,1:55) = 20;
    %input_signal(5,1:55) = -20;
    %input_signal(4,1:5) = -30;
    %input_signal(8,1:5) = 30;
    %input_signal(2,1:5) = 30;
    %input_signal(6,1:5) = -30;

    %input_signal(1,1:50) = 100;
    %input_signal(3,1:50) = -100;


    %input_signal(1,:) = 100;
    %input_signal(5,:) = -100;
    %input_signal(4,1:70) = -10;
    %input_signal(6,1:70) = 10;
    %input_signal(8,1:70) = 10;

    %input_signal = zeros(param.nu,length(tspan));
    % Initialize arrays to store states and time
    XX = zeros(param.nx,length(tspan));
    XX(:,1) = initial_state;
    
    % Simulate the system using Euler's method
    for i = 2:length(tspan)
        
        xdot = dynamics(XX(:,i-1), input_signal(:,i-1),param);
        %xdot = dynamics(XX(:,i-1), input_signal(:,i-1),param);
        %XX(:,i) = XX(:,i-1)-xbar + xdot * param.Ts+xbar;
        XX(:,i) = XX(:,i-1) + xdot * param.Ts;
    end
    
    if max(max(isnan(XX)))
        error('There is a NaN')
    end
    
    %figure;plot(XX(1,:),XX(5,:)); hold on;plot(XX(2,:),XX(6,:));
    
    if exp <= num_exp
        dExp{1,exp}= [zeros(2,length(tspan));input_signal(1,:);zeros(4,length(tspan));input_signal(2,:);zeros(2,length(tspan));input_signal(3,:)];
        yExp{1,exp} = XX;
    else
        dExp_val{1,1}= [zeros(2,length(tspan));input_signal(1,:);zeros(4,length(tspan));input_signal(2,:);zeros(2,length(tspan));input_signal(3,:)];
        yExp_val{1,1} = XX;
    end

end

if animatedplot
    figure;
    for nobs = 1:size(obs,1)
        for robsi = robs:-0.1:0.1
        circle(obs(nobs,1),obs(nobs,2),robsi,'k');
        end
    end

    hold on;
    y = 1;
    axis([-6 6 -12 12])
    p1 = scatter(XX(1,1),XX(2,1));p2 =scatter(XX(5,1),XX(6,1));
    p3 =scatter(XX(9,1),XX(10,1));p4 =scatter(XX(13,1),XX(14,1));
    p7 =scatter(XX(9,1),XX(10,1));p8 =scatter(XX(13,1),XX(14,1));
    p9 =scatter(XX(9,1),XX(10,1));p10 =scatter(XX(13,1),XX(14,1));
    traj1 = plot(XX(1,1),XX(2,1));traj2 =plot(XX(5,1),XX(6,1));
    traj3 =plot(XX(9,1),XX(10,1));traj4 =plot(XX(13,1),XX(14,1));
    for t = 1:10:length(tspan)
        delete(p1);delete(p2);delete(p3);delete(p4);
        delete(p7);delete(p8);delete(p9);delete(p10);
        delete(traj1);delete(traj2);delete(traj3);delete(traj4);
        p1 = circle(XX(1,t),XX(2,t),0.5,'r');
        p2=circle(XX(5,t),XX(6,t),.500,'b');
        p3=circle(XX(9,t),XX(10,t),.500,'g');
        p4=circle(XX(13,t),XX(14,t),.500,[0.4660 0.6740 0.1880]);
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
        pause(0.1);
    end

    figure;
    subplot(1,3,1);title('pos')
    hold on;
    for ind = 1:4:size(XX,1)
        plot(1:t,XX(ind,1:t));
        plot(1:t,XX(ind+1,1:t));
    end
    subplot(1,3,2);title('vel')
    hold on;
    for ind = 3:4:size(XX,1)
        plot(1:t,XX(ind,1:t));
        plot(1:t,XX(ind+1,1:t));
    end
    subplot(1,3,3);
    hold on;
    title('u')
    for ind = 1:size(input_signal,1)
        plot(1:t,input_signal(ind,1:t));
    end
end


%%%%%%%%%
%%%%%%%%%
%%%%%%%%%

Ts = param.Ts;

save('datasetFisso_sysID_3carrelli_20exp_ts005_l2gainbasso','Ts','dExp','dExp_val','yExp','yExp_val')



% Define the system dynamics function f(x,u)
function xdot = dynamics(x, u,param)
    k1 = param.k1;k2=param.k2;k3=param.k3;k4=param.k4;k5=param.k5;k6=param.k6;k7=param.k7;k8=param.k8;
    m1 = param.m1;m2=param.m2;m3=param.m3;m4=param.m4;
    c1=param.c1;c2=param.c2;c3=param.c3;c4=param.c4;c5 = param.c5;c6=param.c6;c7=param.c7;c8=param.c8;
    k9 = param.k9; k10 = param.k10;
    c9 = param.c9; c10 = param.c10;
    xdot = zeros(param.nx,1);
    xt1x = param.xbar(1); xt1y = param.xbar(2); 
    xt2x = param.xbar(5); xt2y = param.xbar(6); 
    xt3x = param.xbar(9); xt3y = param.xbar(10); 
    xt4x = param.xbar(13); xt4y = param.xbar(14);

    maxlength5 = 4;
    maxlength6 =1.5;
    maxlength7 =4;
    maxlength8 =1.5;
    maxlength9 =sqrt(maxlength5^2+maxlength8^2);
    maxlength10 =sqrt(maxlength7^2+maxlength6^2);

    maxlength = [0 maxlength5 maxlength9 maxlength8;
                maxlength5 0 maxlength6 maxlength10;
                maxlength9 maxlength6 0 maxlength7;
                maxlength8 maxlength10 maxlength7 0];


    p1x = x(1);p1y = x(2);v1x = x(3);v1y = x(4);
    p2x = x(5);p2y = x(6);v2x = x(7);v2y = x(8);
    p3x = x(9);p3y = x(10);v3x = x(11);v3y = x(12);
    p4x = x(13);p4y = x(14);v4x = x(15);v4y = x(16);

    px = [p1x;p2x;p3x;p4x];
    py = [p1y;p2y;p3y;p4y];
    vx = [v1x;v2x;v3x;v4x];
    vy = [v1y;v2y;v3y;v4y];


    %12
    deltax12 = p1x-p2x;deltax21 = -deltax12;
    deltavx12 = v1x-v2x;deltavx21 = -deltavx12;
    deltay12 = p1y-p2y;deltay21 = -deltay12;
    deltavy12 = v1y-v2y;deltavy21 = -deltavy12;

    %13
    deltax13 = p1x-p3x;deltax31 = -deltax13;
    deltavx13 = v1x-v3x;deltavx31 = -deltavx13;
    deltay13 = p1y-p3y;deltay31 = -deltay13;
    deltavy13 = v1y-v3y;deltavy31 = -deltavy13;

    %14
    deltax14 = p1x-p4x;deltax41 = -deltax14;
    deltavx14 = v1x-v4x;deltavx41 = -deltavx14;
    deltay14 = p1y-p4y;deltay41 = -deltay14;
    deltavy14 = v1y-v4y;deltavy41 = -deltavy14;

    %23
    deltax23 = p2x-p3x;deltax32 = -deltax23;
    deltavx23 = v2x-v3x;deltavx32 = -deltavx23;
    deltay23 = p2y-p3y;deltay32 = -deltay23;
    deltavy23 = v2y-v3y;deltavy32 = -deltavy23;

    %24
    deltax24 = p2x-p4x;deltax42 = -deltax24;
    deltavx24 = v2x-v4x;deltavx42 = -deltavx24;
    deltay24 = p2y-p4y;deltay42 = -deltay24;
    deltavy24 = v2y-v4y;deltavy42 = -deltavy24;

    %34
    deltax34 = p3x-p4x;deltax43 = -deltax34;
    deltavx34 = v3x-v4x;deltavx43 = -deltavx34;
    deltay34 = p3y-p4y;deltay43 = -deltay34;
    deltavy34 = v3y-v4y;deltavy43 = -deltavy34;

    %xt
    deltaxt1 = p1x-xt1x; deltayt1 = p1y-xt1y;
    deltaxt2 = p2x-xt2x; deltayt2 = p2y-xt2y;
    deltaxt3 = p3x-xt3x; deltayt3 = p3y-xt3y;
    deltaxt4 = p4x-xt4x; deltayt4 = p4y-xt4y;

    deltax = [0 deltax12 deltax13 deltax14;
              deltax21 0 deltax23 deltax24;
              deltax31 deltax32 0 deltax34;
              deltax41 deltax42 deltax43 0];

    deltay = [0 deltay12 deltay13 deltay14;
              deltay21 0 deltay23 deltay24;
              deltay31 deltay32 0 deltay34;
              deltay41 deltay42 deltay43 0];

    deltavx = [0 deltavx12 deltavx13 deltavx14;
              deltavx21 0 deltavx23 deltavx24;
              deltavx31 deltavx32 0 deltavx34;
              deltavx41 deltavx42 deltavx43 0];

    deltavy = [0 deltavy12 deltavy13 deltavy14;
              deltavy21 0 deltavy23 deltavy24;
              deltavy31 deltavy32 0 deltavy34;
              deltavy41 deltavy42 deltavy43 0];

    deltaxt = [deltaxt1;deltaxt2;deltaxt3;deltaxt4];
    deltayt = [deltayt1;deltayt2;deltayt3;deltayt4];



    

    projx12 = cos(atan2(deltay12,deltax12));
    projy12 = sin(atan2(deltay12,deltax12));
    projx13 = cos(atan2(deltay13,deltax13));
    projy13 = sin(atan2(deltay13,deltax13));
    projx21 = cos(atan2(deltay21,deltax21));
    projy21 = sin(atan2(deltay21,deltax21));
    projx14 = cos(atan2(deltay14,deltax14));
    projy14 = sin(atan2(deltay14,deltax14));
    projx24 = cos(atan2(deltay24,deltax24));
    projy24 = sin(atan2(deltay24,deltax24));
    projx41 = cos(atan2(deltay41,deltax41));
    projy41 = sin(atan2(deltay41,deltax41));
    projx23 = cos(atan2(deltay23,deltax23));
    projy23 = sin(atan2(deltay23,deltax23));
    projx31 = cos(atan2(deltay31,deltax31));
    projy31 = sin(atan2(deltay31,deltax31));
    projx32 = cos(atan2(deltay32,deltax32));
    projy32 = sin(atan2(deltay32,deltax32));
    projx34 = cos(atan2(deltay34,deltax34));
    projy34 = sin(atan2(deltay34,deltax34));
    projx43 = cos(atan2(deltay43,deltax43));
    projy43 = sin(atan2(deltay43,deltax43));
    projx42 = cos(atan2(deltay42,deltax42));
    projy42 = sin(atan2(deltay42,deltax42));

    


    projvx12 = cos(atan2(deltavy12,deltavx12));
    projvy12 = sin(atan2(deltavy12,deltavx12));
    projvx13 = cos(atan2(deltavy13,deltavx13));
    projvy13 = sin(atan2(deltavy13,deltavx13));
    projvx21 = cos(atan2(deltavy21,deltavx21));
    projvy21 = sin(atan2(deltavy21,deltavx21));
    projvx24 = cos(atan2(deltavy24,deltavx24));
    projvy24 = sin(atan2(deltavy24,deltavx24));
    projvx14 = cos(atan2(deltavy14,deltavx14));
    projvy14 = sin(atan2(deltavy14,deltavx14));
    projvx41 = cos(atan2(deltavy41,deltavx41));
    projvy41 = sin(atan2(deltavy41,deltavx41));
    projvx23 = cos(atan2(deltavy23,deltavx23));
    projvy23 = sin(atan2(deltavy23,deltavx23));
    projvx31 = cos(atan2(deltavy31,deltavx31));
    projvy31 = sin(atan2(deltavy31,deltavx31));
    projvx32 = cos(atan2(deltavy32,deltavx32));
    projvy32 = sin(atan2(deltavy32,deltavx32));
    projvx34 = cos(atan2(deltavy34,deltavx34));
    projvy34 = sin(atan2(deltavy34,deltavx34));
    projvx43 = cos(atan2(deltavy43,deltavx43));
    projvy43 = sin(atan2(deltavy43,deltavx43));
    projvx42 = cos(atan2(deltavy42,deltavx42));
    projvy42 = sin(atan2(deltavy42,deltavx42));

    projxt1 = cos(atan2(deltayt1,deltaxt1));
    projxt2 = cos(atan2(deltayt2,deltaxt2));
    projxt3 = cos(atan2(deltayt3,deltaxt3));
    projxt4 = cos(atan2(deltayt4,deltaxt4));
    projyt1 = sin(atan2(deltayt1,deltaxt1));
    projyt2 = sin(atan2(deltayt2,deltaxt2));
    projyt3 = sin(atan2(deltayt3,deltaxt3));
    projyt4 = sin(atan2(deltayt4,deltaxt4));

    projvxt1 = cos(atan2(v1y,v1x));
    projvxt2 = cos(atan2(v2y,v2x));
    projvxt3 = cos(atan2(v3y,v3x));
    projvxt4 = cos(atan2(v4y,v4x));

    projvyt1 = sin(atan2(v1y,v1x));
    projvyt2 = sin(atan2(v2y,v2x));
    projvyt3 = sin(atan2(v3y,v3x));
    projvyt4 = sin(atan2(v4y,v4x));

    projx = [0 projx12 projx13 projx14;
            projx21 0 projx23 projx24;
            projx31 projx32 0 projx34;
            projx41 projx42 projx43 0];
    projy = [0 projy12 projy13 projy14;
            projy21 0 projy23 projy24;
            projy31 projy32 0 projy34;
            projy41 projy42 projy43 0];

    projvx = [0 projvx12 projvx13 projvx14;
            projvx21 0 projvx23 projvx24;
            projvx31 projvx32 0 projvx34;
            projvx41 projvx42 projvx43 0];
    projvy = [0 projvy12 projvy13 projvy14;
            projvy21 0 projvy23 projvy24;
            projvy31 projvy32 0 projvy34;
            projvy41 projvy42 projvy43 0];

    projxt = [projxt1; projxt2; projxt3;projxt4];
    projyt = [projyt1; projyt2; projyt3;projyt4];
    projvxt = [projvxt1; projvxt2; projvxt3;projvxt4];
    projvyt = [projvyt1; projvyt2; projvyt3;projvyt4];

    if maxlength5 == 0
        overlength5 = 1; overlength6 = 1; overlength7 = 1; overlength8 = 1; overlength9 = 1; overlength10 = 1;
    else
        if sqrt(deltax12^2+deltay12^2)<maxlength5
           overlength5 = -1;
        else
           overlength5 = 1;
        end
    
        if sqrt(deltax23^2+deltay23^2)<maxlength6
           overlength6 = -1;
        else
           overlength6 = 1;
        end
    
        if sqrt(deltax34^2+deltay34^2)<maxlength7
           overlength7 = -1;
        else
           overlength7 = 1;
        end
    

        if sqrt(deltax14^2+deltay14^2)<maxlength8
           overlength8 = -1;
        else
           overlength8 = 1;
        end
    
        if sqrt(deltax31^2+deltay31^2)<maxlength9
               overlength9 = -1;
            else
               overlength9 = 1;
        end
        
        if sqrt(deltax24^2+deltay24^2)<maxlength10
           overlength10 = -1;
        else
           overlength10 = 1;
        end
    end
    overlenthtot = [overlength5;overlength6;overlength7;overlength8;overlength9;overlength10];

    

    Fk01 = k1*sqrt(deltaxt1^2+deltayt1^2);
    Fk02 = k2*sqrt(deltaxt2^2+deltayt2^2);
    Fk03 = k3*sqrt(deltaxt3^2+deltayt3^2);
    Fk04 = k4*sqrt(deltaxt4^2+deltayt4^2);
    Fk5 = k5*abs(sqrt(deltax12^2+deltay12^2)-maxlength5);
    Fk6 = k6*abs(sqrt(deltax23^2+deltay23^2)-maxlength6);
    Fk7 = k7*abs(sqrt(deltax34^2+deltay34^2)-maxlength7);
    Fk8 = k8*abs(sqrt(deltax41^2+deltay41^2)-maxlength8);

    Fk9 = k9*abs(sqrt(deltax31^2+deltay31^2)-maxlength9);
    Fk10 = k10*abs(sqrt(deltax42^2+deltay42^2)-maxlength10);

    Fc01 = c1*sqrt(v1x^2+v1y^2);
    Fc02 = c2*sqrt(v2x^2+v2y^2);
    Fc03 = c3*sqrt(v3x^2+v3y^2);
    Fc04 = c4*sqrt(v4x^2+v4y^2);
    Fc5 =  + c5*sqrt(deltavx12^2+deltavy12^2);
    Fc6 =  + c6*sqrt(deltavx23^2+deltavy23^2);
    Fc7 =  + c7*sqrt(deltavx34^2+deltavy34^2);
    Fc8 =  + c8*sqrt(deltavx41^2+deltavy41^2);

    Fc9 =  + c9*sqrt(deltavx31^2+deltavy31^2);
    Fc10 =  + c10*sqrt(deltavx42^2+deltavy42^2);

    
    Fc = [Fc01;Fc02;Fc03;Fc04;Fc5;Fc6;Fc7;Fc8;Fc9;Fc10];
    Fk = [Fk01;Fk02;Fk03;Fk04;Fk5;Fk6;Fk7;Fk8;Fk9;Fk10];
    
    
    xdot(1) = x(3);
    xdot(2) = x(4);
    xdot(3) = -(overlength5*Fk5*projx12+overlength8*Fk8*projx14+overlength9*Fk9*projx13)/m1-(Fc5*projvx12+Fc8*projvx14+Fc9*projvx13)/m1 -Fk01*projxt1/m1 -Fc01*projvxt1/m1 +u(1)/m1;
    xdot(4) = -(overlength5*Fk5*projy12+overlength8*Fk8*projy14+overlength9*Fk9*projy13)/m1-(Fc5*projvy12+Fc8*projvy14+Fc9*projvy13)/m1 -Fk01*projyt1/m1 -Fc01*projvyt1/m1 +u(2)/m1;
    xdot(5) = x(7);
    xdot(6) = x(8);
    xdot(7) = -(overlength5*Fk5*projx21+overlength6*Fk6*projx23+overlength10*Fk10*projx24)/m2 -(Fc5*projvx21+Fc6*projvx23+Fc10*projvx24)/m2 -Fk02*projxt2/m2 -Fc02*projvxt2/m2 +u(3)/m2;
    xdot(8) = -(overlength5*Fk5*projy21+overlength6*Fk6*projy23+overlength10*Fk10*projy24)/m2 -(Fc5*projvy21+Fc6*projvy23+Fc10*projvy24)/m2 -Fk02*projyt2/m2 -Fc02*projvyt2/m2 +u(4)/m2;
    xdot(9) = x(11);
    xdot(10) = x(12);
    xdot(11) = -(overlength6*Fk6*projx32+overlength7*Fk7*projx34+overlength9*Fk9*projx31)/m3 -(Fc6*projvx32+Fc7*projvx34+Fc9*projvx31)/m3 -Fk03*projxt3/m3 -Fc03*projvxt3/m3 +u(5)/m3;
    xdot(12) = -(overlength6*Fk6*projy32+overlength7*Fk7*projy34+overlength9*Fk9*projy31)/m3 -(Fc6*projvy32+Fc7*projvy34+Fc9*projvy31)/m3 -Fk03*projyt3/m3 -Fc03*projvyt3/m3 +u(6)/m3;
    xdot(13) = x(15);
    xdot(14) = x(16);
    xdot(15) = -(overlength7*Fk7*projx43+overlength8*Fk8*projx41+overlength10*Fk10*projx42)/m4-(Fc7*projvx43 + Fc8*projvx41+Fc10*projvx42)/m4 -Fk04*projxt4/m4 -Fc04*projvxt4/m4 +u(7)/m4;
    xdot(16) = -(overlength7*Fk7*projy43+overlength8*Fk8*projy41+overlength10*Fk10*projy42)/m4-(Fc7*projvy43 + Fc8*projvy41+Fc10*projvy42)/m4 -Fk04*projyt4/m4 -Fc04*projvyt4/m4 +u(8)/m4;


end

% Nonlinear spring dynamics
function kx = nonspring(k,x)
    kx = k*x^3;
end

function h = circle(x,y,r,colr)
hold on
th = 0:pi/50:2*pi;
xunit = r * cos(th) + x;
yunit = r * sin(th) + y;
h = plot(xunit, yunit,'Color',colr);
end