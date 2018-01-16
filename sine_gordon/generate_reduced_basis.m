function generate_reduced_basis()
	load('snaps.mat')

	X = generate_X_mat();

	fprintf('generating greedy basis...\n')
	Phi = greedy_energy(WY,X,100);
	fprintf('generating basis for the nonlinear term...\n')	
	[Phi,P] = greedy_nonlinear(Phi,NL,X,100);
	fprintf('saving reduced basis...\n')
	save('reduced_basis.mat','Phi','X','P')
function POD(Y)
	[U,S,V] = svd(Y);

	Phi = U(:,1:100);

	save('reduced_basis.mat','Phi')

function POD_energy(Y,X)
	snaps = X*Y;

	[U,S,V] = svd(snaps);

	Phi = U(:,1:38);
	Phi = inv(X)*Phi;
	save('reduced_basis.mat','Phi','X')

function PSD(Y)
	Q = Y(1:500,:);
	P = Y(501:end,:);

	snaps = [Q,P];
	[U,S,V] = svd(snaps);
%	semilogy(diag(S))

	N = size(Q,1);
	K = 100;
	Phi = [ U(:,1:K) , zeros(N,K) ; zeros(N,K) , U(:,1:K) ];

	save('reduced_basis.mat','Phi')

function greedy(Y,MAX_ITER)

%	idx = randperm(size(Y,2));
%	snaps = Y(:,idx(1:200));
%	num_snaps = 200;
	snaps = Y;
	num_snaps = size(Y,2);

	N = size(Y,1)/2;
	J2n = [ zeros(N,N) , eye(N) ; -eye(N) , zeros(N,N) ];

%	E = snaps(:,fix( rand * 100 ));
	E = snaps(:,1);
	E = E/norm(E);
	F = J2n'*E;
	Phi = [E,F];

	K=1;

	for iter=1:MAX_ITER
		projmat = Phi*Phi';
		er = zeros(num_snaps,1);
		for i=1:num_snaps
			samp = snaps(:,i);
			e = norm( samp - projmat*samp );
			er(i) = e;
		end
		[dummy,argmax] = max(er);
		[iter , argmax]

		e = snaps(:,argmax);

		e = sqr(e,E,F,J2n);
		e = sqr(e,E,F,J2n);

		E = [E,e];
		F = [F,J2n'*e];
		Phi = [E,F];
	end
	
	save('reduced_basis.mat','Phi')

function Phi = greedy_energy(Y,X,MAX_ITER)
	X_inv = inv(X);

%	idx = randperm(size(Y,2));
%	snaps = Y(:,idx(1:200));
%	num_snaps = 200;
	snaps = Y;
	num_snaps = size(Y,2);
	snaps = X*Y;

	N = size(Y,1)/2;
	J2n = construct_J(N);

	Jtn = X*J2n*X;
	Jtn_inv = -X_inv*J2n*X;

%	E = snaps(:,fix( rand * 100 ));
	E = snaps(:,1);
	E = E/sqrt( E'*X*X*E );
	F = Jtn_inv*E;
	Phi = [E,F];

	K=1;

	for iter=1:MAX_ITER
		J2k = construct_J(K);
		Phi_plus = J2k'*Phi'*Jtn;
		projmat = Phi*Phi_plus;
		for i=1:num_snaps
			samp = snaps(:,i);
			e = norm( samp - projmat*samp );
			er(i) = e;
		end
		[dummy,argmax] = max(er);
		[iter , argmax , dummy]

		e = snaps(:,argmax);

		e = sqrt_Jt(e,E,F,Jtn,X);
		e = sqrt_Jt(e,E,F,Jtn,X);

		E = [E,e];
		F = [F,Jtn_inv*e];
		Phi = [E,F];

		K = K+1;
	end

	Phi = inv(X)*Phi;

%	save('temp_data.mat','Phi');

%	save('reduced_basis.mat','Phi','X')

function [Phi,P] = greedy_nonlinear(Phi,NL,X,MAX_ITER)
	Phit = X*Phi;

	X_inv = inv(X);
	K = size(Phi,2)/2;

	snaps = NL;
	num_snaps = size(NL,2);
	snaps = X_inv*snaps;

	N = size(Phi,1)/2;
	J2n = construct_J(N);

	Jbn = X_inv*J2n*X_inv;
	Jbn_inv = -X*J2n*X_inv;
	Jtn = X*J2n*X;

	J2k = construct_J(K);
	Psit = (J2k'*Phit'*Jtn)';
	
	E = Psit(:,1:K);
	F = Psit(:,K+1:end);

	for iter=1:MAX_ITER
		J2k = construct_J(K);
		Psit_plus = J2k'*Psit'*Jbn;
		projmat = Psit*Psit_plus;
		for i=1:num_snaps
			samp = snaps(:,i);
			e = norm( samp - projmat*samp );
			er(i) = e;
		end
		[dummy,argmax] = max(er);
		[iter , argmax , dummy]

		e = snaps(:,argmax);

		e = sqrt_Jt(e,E,F,Jbn,X_inv);
		e = sqrt_Jt(e,E,F,Jbn,X_inv);

		E = [E,e];
		F = [F,Jbn_inv*e];
		Psit = [E,F];

		K = K+1;
	end

	Psi = X*Psit;

	[ P , P_list ] = deim(Psi);

	J2k = construct_J(K);
	Phit = (J2k'*Psit'*Jbn)';

	Phi = X_inv*Phit;

function J = construct_J(K)
	J = [ zeros(K,K) , eye(K) ; -eye(K) , zeros(K,K) ];

function e = sqrt_Jt(v,E,F,Jt,X)
	e = v;
	for i=1:size(E,2)
		e = Jt_orthogonalize( e , E(:,i) , F(:,i) , Jt );	
	end
	e = e / sqrt( e'*X*X*e );

function res = Jt_orthogonalize(vec,e,f,Jt)
	alpha = -vec'*Jt*f;
	beta = vec'*Jt*e;
	res = vec + alpha*e + beta*f;

function e = sqr(v,E,F,J)
	e = v;
	for i = 1:size(E,2)
		e = symplectify( e , E(:,i) , F(:,i) , J );
	end
	e = e / norm(e);

function res = symplectify(vec,e,f,J)
	alpha = -bilin(vec,f,J);
	betta = bilin(vec,e,J);
	res = vec + alpha*e + betta*f;

function res = bilin(v,w,J)
	res = v'*J*w;

function X_mat = generate_X_mat()
	L = 50;
	N = 250;
	dx = L/N;
	x0 = 10;

	T = 50;
	dt = 0.5;
	MAX_ITER = T/dt;
	v = 0.2;

	diag1 = -2 * ones(1,N);
	diag2 = ones(1,N-1);


	Dxx = diag(diag1,0) + diag(diag2,1) + diag(diag2,-1);
	Dxx = 1/(dx^2)*Dxx;

	K = [-Dxx , zeros(N,N) ; zeros(N,N) , eye(N) ];

	X_mat = real(sqrtm(K));

function [ P , P_list ] = deim(U_psi)
	[ a , b ] = max(U_psi(:,1));
	P_list = [ b ];
	P = zeros( size(U_psi,1) , 1 );
	P( b , 1 ) = 1;
	U = U_psi(:,1);
	for i = 2 : size(U_psi,2)
		u_l = U_psi(:,i);
		c = (P'*U)\(P'*u_l);
		r = u_l - U*c;
		[a,b] = max(abs(r));
		U = [ U , U_psi(:,i) ];
		P = [ P , zeros( size(U_psi,1) , 1 ) ];
		P(b,i) = 1;
		P_list = [ P_list , b ];

	end
