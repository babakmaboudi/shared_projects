function wave()
	L = 1;
	N = 500;
	dx = L/N;
	c = 0.1;

	T = 50;
	dt = 0.01;
	MAX_ITER = T/dt;

	diag1 = -2 * ones(1,N);
	diag2 = ones(1,N-1);

	
	Dxx = diag(diag1,0) + diag(diag2,1) + diag(diag2,-1);
	Dxx(1,end) = 1;
	Dxx(end,1) = 1;

	Dxx = c^2/(dx^2)*Dxx;

	L = [zeros(N,N) , eye(N) ; Dxx , zeros(N,N)];

	[q0,p0] = initial_condition(N);
	y0 = [q0;p0];

	step = inv(eye(2*N) - dt/2*L)*(eye(2*N) + dt/2*L);

	Y = time_stepping(y0,step,MAX_ITER);

%	save('snaps.mat','Y');
	Q = Y(1:N,:);
	P = Y(N+1:end,:);

	for i=1:size(Q,2)
		energy(Q(:,i),P(:,i),Dxx,dx)
	end

function Y = time_stepping(y0,step,MAX_ITER)
	y = y0;
	Y = y0;

	for i=1:MAX_ITER
		y = step*y;
		if(mod(i,10) == 1)
		plot(y);
		drawnow
		end

		Y = [Y , y];
	end

function [q,p] = initial_condition(N)
	X = linspace(0,1,N);
	q = zeros( length(X) , 1 );
	p = zeros( length(X) , 1 );
	for i = 1 : length(X)
		s = 10*abs( X(i) - 1.0/2.0 );
		if( s <= 1 )
			q(i) = 1.0 - 3.0/2.0*s^2 + 3.0/4.0*s^3;
		elseif( s <=2 )
			q(i) = 1.0/4.0*(2.0-s)^3;
		else
			q(i) = 0;
		end
	end

function e = energy(q,p,Dxx,dx)
	e = sum(-q.*(Dxx*q) + p.*p)*dx;
