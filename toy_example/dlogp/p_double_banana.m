function p = p_double_banana(X,obs,prior)
	[N,d] = size(X);
	a = 1;
	b = 100;
	model.F = @(u) log( (a - u(1))^2 + b*(u(2) - u(1)^2)^2 ); 
	model.m = 1;
	model.n = 2;

	if nargin<2
		rng(1);
		% Number of independent observations
		obs.nobs    = 1;
		% Noise standard deviation
		obs.std     = 0.3;
		% Nose variance
		obs.std2    = obs.std^2;
		% Noise
		obs.noise   = obs.std*randn(obs.nobs,1); 
		% Real parameter vector
		obs.u_true  = rand(1,d);
		% Observations
		u1 = obs.u_true(1,1); u2 = obs.u_true(1,2);
		obs.y       = log( (a - u1)^2 + b*(u2 - u1^2)^2 ) + obs.noise;
	end

	if nargin<3
		% Prior mean
		prior.m0      = zeros(d,1);
		% Prior covariance matrix
		prior.C0      = eye(d);
		% Prior precision matrix
		prior.C0i     = prior.C0^(-1);
		% Square root of prior covariance matrix
		prior.C0sqrt  = real(sqrtm(prior.C0));
		% Square root of prior precision matrix
		prior.C0isqrt = real(sqrtm(prior.C0i));
	end

	p = zeros([N,1]);
	for i = 1:N
		x = X(i,:)';
		Fu = feval(model.F, x);
		mlprr = 0.5*(x - prior.m0)' * prior.C0i * (x - prior.m0);
		misfit = (obs.y - Fu) / obs.std;
        mllkd  = 0.5*sum(misfit(:).^2);
        p(i) = exp( -(mllkd + mlprr) );
	end

end