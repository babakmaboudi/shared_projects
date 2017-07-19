function [ vargout ] = damped_spring(vargin )
%damped_spring simulates a damped spring with unit mass and damping
%coefficient 0.01. Solving the system in enlarged phase space (spring coupled with infinitely long strings), one expects
%the Hamiltonian H of the enlarged system to be constant over time.

% [t,y] = ode45(@ode,[0 1000],[0.1; 0.1; 0.1]);

[t,y] = implicit_midpoint(@ode,[0 100],0.01,[0.1; 0.1; 0.1]);

% y(:,1) is position, y(:,2) is momentum, and y(:,3) is the force function
% f(t)
subplot(1,2,1), plot(t,y(:,1),'-o',t,y(:,2),'-o',t,y(:,3),'-o')
title('Solution of a damped spring with ODE45');
xlabel('Time t');
ylabel('Solution y');
legend('y_1','y_2','y_3')

% define string displacements based on the spring displacements,
% displacements are defined in one-quarter of the plane because of their
% symmetry about the s-axis
phi(:,1) = y(:,1);
for j = 2:length(t)
    phi(:,j) = circshift(phi(:,j-1),[1,0]); phi(1:j-1,j) = 0;
end
phi = sqrt(0.01/2)*phi;

% define energy of the enlarged system
[phi_s, phi_t] = gradient(phi,t,t);

H_spring = 1/2*(y(:,2) + sqrt(2*0.01)*phi(:,1)).^2 + 1/2*(y(:,1)).^2;
H_string = trapz(t,phi_s.^2 + phi_t.^2,2);
H = H_spring + H_string;

subplot(1,2,2), plot(t,H,t,H_spring,t,H_string)
end

function dydt = ode(t,y)
%VDP1  Evaluate the van der Pol ODEs for mu = 1
%
%   See also ODE113, ODE23, ODE45.

%   Jacek Kierzenka and Lawrence F. Shampine
%   Copyright 1984-2014 The MathWorks, Inc.

dydt = [y(3); -y(1); -y(1)-0.01*y(3)];
end