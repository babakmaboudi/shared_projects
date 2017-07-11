function [ vargout ] = damped_spring(vargin )
%damped_spring simulates a damped spring with unit mass and damping
%coefficient 0.01. Solving the system in enlarged phase space (spring coupled with infinitely long strings), one expects
%the Hamiltonian H of the enlarged system to be constant over time.

[t,y] = ode45(@ode,[0 100],[0.1; 0.1; 0.1]);

subplot(1,2,1), plot(t,y(:,1),'-o',t,y(:,2),'-o',t,y(:,3),'-o')
title('Solution of a damped spring with ODE45');
xlabel('Time t');
ylabel('Solution y');
legend('y_1','y_2','y_3')

s = [flip(t);t(2:end)];

phi(:,1) = y(:,1);
for j = 2:length(t)
    phi(:,j) = circshift(phi(:,j-1),[1,0]); phi(1:j-1,j) = NaN;
end

H = 1/2*(y(:,2) + sqrt(2*0.01)*phi(:,1)).^2 + 1/2*y(:,1).^2 + trapz(t,(gradient(phi,[diff(t);t(end)-t(end-1)],[diff(t);t(end)-t(end-1)])).^2,2);

subplot(1,2,2), plot(t,H)
end

function dydt = ode(t,y)
%VDP1  Evaluate the van der Pol ODEs for mu = 1
%
%   See also ODE113, ODE23, ODE45.

%   Jacek Kierzenka and Lawrence F. Shampine
%   Copyright 1984-2014 The MathWorks, Inc.

dydt = [y(3); -y(1); -y(1)-0.01*y(3)];
end