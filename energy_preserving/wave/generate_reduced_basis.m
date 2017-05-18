function generate_reduced_basis()
	load('snaps.mat')

	[U,S,V] = svd(Y);
	semilogy(diag(S))

	A = U(:,1:200);

	save('reduced_basis.mat','A')
