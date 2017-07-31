function [ vargout ] = damped_spring(vargin )
%damped_spring simulates a damped spring with unit mass and damping
%coefficient 0.01. Solving the system in enlarged phase space (spring coupled with infinitely long strings), one expects
%the Hamiltonian H of the enlarged system to be constant over time.

% A. Bhatt July 2017

% [t,y] = ode45(@ode,[0 100],[0.1; 0.1; 0.1]);

[t,y] = numint(@ode,[0 100],0.01,[0; 0.1; 0.1],'implicit_midpoint');

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

H_spring = 1/2*(y(:,2) - sqrt(2*0.01)*phi(:,1)).^2 + 1/2*(y(:,1)).^2;
H_string = trapz(t,phi_s.^2 + phi_t.^2,2);
% H_string = sum(phi_s.^2 + phi_t.^2,2)*0.01;
H = H_spring + H_string;

subplot(1,2,2), plot(t,H,t,H_spring,t,H_string)
end

function dydt = ode(t,y)
dydt = [y(3); -y(1); 0];
end

function [t,y] = numint(ode, tspan, dt, y0, int_flag)
% function [t,y] = numint(ode, tspan, y0, jac)
% solves first order ode y' = f(t,y) in the time interval tspan with initial condition
% y0 with the (symplectic) implicit midpoint rule
%   Fully implicit: if jacobian is provided
%   Ecplicit, iterative sheme: otherwise
%
% A. Bhatt, April 2017
% Patrick Buchfink, June 2017

t = tspan(1):dt:tspan(2);
N = length(t);
At = @(x,y) (x+y)/2;
tol = 10^-15;
n_dof = size(y0,1);
y = zeros(n_dof, N);
y(:,1) = y0;

% right hand side for implicit midpoint
rhs = @(t,y,n) y(:,n) +[0;0;(y(2,n+1)-0.01*dt*sum(y(3,1:n)))/(1+0.01*dt)-y(3,n)] + dt*ode(At(t(n),t(n+1)),At(y(:,n),y(:,n+1)));
% rhs = @(t,y,n) y(:,n) +[0;0;(y(2,n+1) -0.01*trapz(t(1:n),y(3,1:n)))/(1+0.01*dt)-y(3,n)] + dt*ode(At(t(n),t(n+1)),At(y(:,n),y(:,n+1)));

% fixed point iteration
switch int_flag
    case 'implicit_midpoint'
        for n = 1:N-1
            y(:,n+1) = y(:,n);
            yk = rhs(t,y,n);
            
            while 1
                y(:,n+1) = yk;
                y(:,n+1) = rhs(t,y,n);
                
                if norm(yk-y(:,n+1),inf) < tol
                    break
                end
                yk = y(:,n+1);
            end
            
            % Print progress in multiples of 10%
            if floor(((n-1)/N)*10) < floor((n/N)*10)
                fprintf('implicit midpoint: %3d%%\n', floor((n/N)*10)*10)
            end
        end
        % Print progress of 100%
        fprintf('implicit midpoint: %3d%%\n', 100)
        
        % Fully implicit (for linear systems)
    case 'explicit_euler'
        for n = 1:N-1
           y(2,n+1) = y(2,n) -dt*y(1,n);
           y(3,n+1) = (y(2,n+1)-0.01*dt*sum(y(3,1:n)))/(1+0.01*dt);
           y(1,n+1) = y(1,n) + dt*y(3,n+1);
        end
    otherwise
        error('Unkown numerical integrator')
end

y = y';
end