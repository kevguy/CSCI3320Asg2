x1 = normrnd (0.0, 1.0, [1000,1]);		% x1 is N(0,1)
x2 = -1.0*4.0*x1;						% x2 = -4*x1
x3 = 10.0*x1 + 10.0;					% x3 = 10*x1 + 10
x4 = exprnd(10.0, 1000, 1);				% x4 is exp with mean=10.0
x5 =-1.0* 0.5*x4;						% x5 = -x4/2;
x6 = x4.^2.0;							% x6 = (x4)^2;
x7 = unifrnd (-100, 100, [1000, 1]);	% x7 is uniform(-100,100)
x8 = -1.0*x7/10; 						% x8 = -x7/10;
x9 = 2.0*x7 + 2.0; % x9 = 2*x7 + 2.0
X = [x1,x2,x3,x4,x5,x6,x7,x8,x9];		% construct input X
clear x1 x2 x3 x4 x5 x6 x7 x8 x9		% de-allocate xi vectors 

mean_vector = transpose(sum(X)/1000.0);	% vector m
X1= X - ones(1000,1)*(transpose(mean_vector))tra;		%normalized X
S = cov(X1);							% S = covariance matrix (9x9)
[V,D] = eig(S);							% V=eigenvectors matrix;
										% D=diag(eigenvalues)
										% in increasing order
E = D*ones(9,1); 						% E=column vector of eigenvalues
E 										% display E

[coeff,score,latent] = pca(X1);
% coeff is the eigenvector matrix
% latent is the eigenvalues vector (descending order)
latent % display eigenvalues 