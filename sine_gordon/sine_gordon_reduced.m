function wave()
	load('reduced_basis.mat')

	L = 50;
	N = 250;
	dx = L/N;
	x0 = 10;

	T = 500;
	dt = 0.1;
	MAX_ITER = T/dt;
	v = 0.2;

	diag1 = -2 * ones(1,N);
	diag2 = ones(1,N-1);


	Dxx = diag(diag1,0) + diag(diag2,1) + diag(diag2,-1);
	Dxx = 1/(dx^2)*Dxx;

	bd = zeros(2*N,1);
	K = [ zeros(N,N) , eye(N) ; Dxx , zeros(N,N) ];

	K_energy = [-Dxx , zeros(N,N) ; zeros(N,N) , eye(N) ];

	bd(end) = 2*pi/(dx^2);

	[q,p] = initial_condition(N,L,x0,v);
	z0 = [q;p];

	N_num = size(Phi,1)/2;
	K_num = size(Phi,2)/2;
	Jk = construct_J(K_num);
	Jn = construct_J(N_num);
	Jt = X*Jn*X;
	Phi_cross = Jk'*Phi'*X*X*Jn;
	PhiX = Phi_cross*X*X;
	Kr = Phi_cross*X*X*K*Phi;
	z0r = Phi_cross*X*X*z0;
	bdr = PhiX*bd;

%	x = newton(z0r,z0r,dt,Kr,bdr,Phi,PhiX);
%	f = func(x,z0r,dt,Kr,bdr,Phi,PhiX)

	Z = time_stepping(z0r,Kr,bdr,dt,MAX_ITER,K_energy,N,dx,Phi,PhiX)

function Z = time_stepping(z0,K,bd,dt,MAX_ITER,K_energy,N,dx,Phi,PhiX)
	z = z0;
	Z = z;

	for i=1:MAX_ITER
%		z = newton(z,z,dt,K,bd);
		z = newton(z,z,dt,K,bd,Phi,PhiX);

		e = hamil(Phi*z,K_energy,N,dx)

		plot(Phi*z)
		grid on
%		ylim([0,2*pi])
		drawnow()

%		if(mod(i,10)==1)
			Z = [Z,z];
%		end
	end

function f = func(x,x0,dt,L,bd,Phi,PhiX)
	x_full = Phi*x;
	x0_full = Phi*x0;
	N = size(x_full,1)/2;
	q = x_full(1:N);
	q0 = x0_full(1:N);
	nonlin = sin( (q+q0)/2 );
	nonlin = PhiX*[ zeros(N,1) ; nonlin ];
	f = (x-x0)/dt - L*(x+x0)/2 + nonlin - bd;

function Jf = jacob(x,x0,dt,L,Phi,PhiX)
	x_full = Phi*x;
	x0_full = Phi*x0;
	N = size(x_full,1)/2;
	q = x_full(1:N);
	q0 = x0_full(1:N);
	nonlin = cos( (q+q0)/2 )/2;
	nonlin = PhiX*[ zeros(N,2*N) ; diag(nonlin) , zeros(N,N) ]*Phi;

	K = size(x,1)/2;
	Jf = eye(2*K)/dt - L/2 + nonlin;

function x = newton(x0,z,dt,L,bd,Phi,PhiX)
	x = x0;
	for i=1:5
		f = func(x,z,dt,L,bd,Phi,PhiX);
		Jf = jacob(x,x0,dt,L,Phi,PhiX);
		x = x - inv(Jf)*f;
	end

function [q,p] = initial_condition(N,L,x0,v)
	X = linspace(0,L,N);
	q = 4*atan(exp((X-x0)/sqrt(1-v^2)));
	q = q';
	p = - 4 * v * exp((X-x0)/sqrt(1-v^2)) ./ ( sqrt(1-v^2) * (1 + exp(2*(X-x0)/sqrt(1-v^2)) ) );
	p = p';

function J = construct_J(K)
	J = [ zeros(K,K) , eye(K) ; -eye(K) , zeros(K,K) ];

function e = hamil(z,K_energy,N,dx)
	q = z(1:N);
	e = (z'*K_energy*z + sum(cos(q)) )/dx;
