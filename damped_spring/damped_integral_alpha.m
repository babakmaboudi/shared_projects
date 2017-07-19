function damped_integral()
	close all;
	q0 = 0;
	p0 = 1;
	alpha = 0.1;
	dt = 0.01;
	max_iter = 100;

	[Q,P,f] = time_stepping(q0,p0,alpha,dt,max_iter);

%	phi = find_phi(f,dt);
%	theta = find_theta(f);
%	dphidx = find_dphidx(f);
%	figure;
%	plot(phi)
%	hold on
%	plot(theta,'r')
%	plot(dphidx,'g')
%	hold off

	E1 = [];
	E2 = [];
	for i=2:length(Q)
		phi = find_phi(f(1:i),alpha,dt);
		theta = find_theta(f(1:i),alpha);
		dphidx = find_dphidx(f(1:i),alpha);
		Tphi = sqrt(2)*sqrt(alpha)*phi(1);
		
		e1 = find_energy1(Q(i),P(i),Tphi,theta,dphidx,dt);
		e2 = find_energy2(Q(i),P(i),Tphi,theta,dphidx,alpha,dt);
		E1 = [E1,e1];
		E2 = [E2,e2];
	end
	plot(E1)
	hold on
	plot(E2,'r');
	plot(E1 + E2,'g');
	hold off
	ylim([0,1])
	grid on

	figure
	for i=2:length(Q)
		phi = find_phi(f(1:i),alpha,dt);
		plot(phi);
		hold on
		plot(1,Q(i)/sqrt(2),'r*')
		hold off
		ylim([-1,1])
		xlim([0,2000])
		drawnow
	end

function [Q,P,f] = time_stepping(q0,p0,alpha,dt,max_iter)
	Q = q0;
	P = p0;
	q = q0;
	p = p0;

	f = p0;

	for i=1:max_iter
		[q,p,f] = deriv(q,p,f,alpha,dt);
		Q = [Q ; q];
		P = [P ; p];
	end

%	phi = find_phi(f,alpha,dt);
%	theta = find_theta(f,alpha);
	
%	figure;
%	plot(phi)
%	hold on
%	plot(theta,'r')
%	hold off

function [q,p,f] = deriv(q,p,f,alpha,dt)
	p = p - dt*q;

	v = (p - dt*alpha*sum(f))/(1+alpha*dt);
	f = [f ; v];

	q = q + dt*f(end);

%	phi = find_phi(f,dt);
%	theta = find_theta(f);
%	plot(theta)
%	drawnow;

function phi = find_phi(f,alpha,dt)
	phi = zeros(length(f),1);
	for i=1:length(phi)
		phi(i) = sqrt(2)/2*sqrt(alpha)*sum(f(1:end-i))*dt;
	end

function theta = find_theta(f,alpha)
	theta = zeros(length(f),1);
	for i=1:length(theta)
		theta(i) = sqrt(2)/2*sqrt(alpha)*f(end-i+1);
	end

function dphidx = find_dphidx(f,alpha)
	dphidx = zeros(length(f),1);
	for i=1:length(dphidx)
		dphidx(i) = -sqrt(2)/2*sqrt(alpha)*f(end-i+1);
	end


function e = find_energy1(q,p,Tphi,theta,dphidx,dt)
	e = 1/2*((p- Tphi)^2 + q^2);

function e = find_energy2(q,p,Tphi,theta,dphidx,alpha,dt)
	e = (dot(theta,theta) + dot(dphidx,dphidx))*dt;

