function wave()
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

%	x = newton(zeros(2*N,1),z0,dt,K,bd);
%	max(abs(func(x,z0,dt,K,bd)))
%	plot(x)

	[Y,NL] = time_stepping(z0,K,bd,dt,MAX_ITER,K_energy,N,dx);
	save('snaps.mat','Y','NL')

function [Z,NL] = time_stepping(z0,K,bd,dt,MAX_ITER,K_energy,N,dx)
	z = z0;
	Z = z;
	NL = [];

	for i=1:MAX_ITER
		z = newton(z,z,dt,K,bd);

		e = hamil(z,K_energy,N,dx)

		plot(z(1:N))
		grid on
%		ylim([0,2*pi])
		drawnow()

%		if(mod(i,10)==1)
			Z = [Z,z];
			q = z(1:N);
			nl = [zeros(N,1); sin(q) ];
			NL = [NL,nl];
%		end
	end

function [q,p] = initial_condition(N,L,x0,v)
	X = linspace(0,L,N);
	q = 4*atan(exp((X-x0)/sqrt(1-v^2)));
	q = q';
	p = - 4 * v * exp((X-x0)/sqrt(1-v^2)) ./ ( sqrt(1-v^2) * (1 + exp(2*(X-x0)/sqrt(1-v^2)) ) );
	p = p';

%function f = Func(x,z,dt,L,bd)
%	N = size(x,1);
%	q = x(1:N/2);
%	q0 = z(1:N/2);
%	nonlin = [ zeros(N/2,1); -sin( (q + q0)/2 )];
%	f = (x-z) - dt*L*( x+z )/2 - dt*nonlin ;
%
%function Jf = JFunc(x,z,dt,L)
%	N = size(x,1);
%	q = x(1:N/2);
%	q0 = z(1:N/2);
%	nonlin = [ zeros(N/2,N); diag( -cos( (q + q0)/2 ))/2 , zeros(N/2,N/2)  ];
%	one_vec = ones(N,1);
%	Jf = eye(N) - dt*L/2 - dt*nonlin;

function f = func(x,x0,dt,L,bd)
	N = size(x,1)/2;
	q = x(1:N);
	q0 = x0(1:N);
	nonlin = sin( (q+q0)/2 );
	nonlin = [ zeros(N,1) ; nonlin ];
	f = (x-x0)/dt - L*(x+x0)/2 + nonlin - bd;

function Jf = jacob(x,x0,dt,L,bd)
	N = size(x,1)/2;
	q = x(1:N);
	q0 = x0(1:N);
	nonlin = cos( (q+q0)/2 )/2;
	nonlin = [ zeros(N,2*N) ; diag(nonlin) , zeros(N,N) ];
	Jf = eye(2*N)/dt - L/2 + nonlin;

function x = newton(x0,z,dt,L,bd)
	x = x0;
	for i=1:5
		f = func(x,z,dt,L,bd);
		Jf = jacob(x,z,dt,L);
		x = x - inv(Jf)*f;
	end

function e = hamil(z,K_energy,N,dx)
	q = z(1:N);
	e = (z'*K_energy*z + sum(cos(q)) )/dx;
