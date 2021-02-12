function [Xs,out] = WGF_m(X_init, dlog_p, opts)
%%%---------------------------------------------%%%
% This implements the particle version of discrete-time
% Wasserstein Gradient flow (WGF).
% 
% Input:
% 		X_init --- initial particle positions, N*d matrix
%		dlog_p --- function handle to compute the derivative of log p(x)
%		opts   --- options structure with fields
%				   tau --- step size
%		      iter_num --- number of iterations
%				 ktype --- the type of the kernel
%        		   ibw --- inverse of the bandwidth of the kernel
%                trace --- whether to trace the information during the process
%						   0: no || 1: yes
%				record --- whether to print the recorded inforamtion during the process
%						   0: no || 1: yes
%			   itPrint --- the interval of printing
%			 	 ptype --- the type of the problem
%						   0: Gaussian
%							  opts.A shall NOT be empty
%						   1: Bayesian logistic regression
%							  opts.X_test, opts.y_test shall NOT be empty
%						   2: Other
%				 rtype --- option for the kernel bandwidth selection subroutine (HE method)
%				 h_pow --- parameter for the kernel bandwidth selection subroutine (HE method)
%				 ttype --- option whether to decay step size
%			   tau_dec --- decay rate of the step size
%			   tau_itv --- decay interval of the step size
%
%
% Author: Yifei Wang, 2020
% 
%%%---------------------------------------------%%%
	tic;
	[N,d] = size(X_init);
	if nargin < 3; opts = []; end
	if ~isfield(opts, 'tau'); opts.tau = 0.1;  end
	if ~isfield(opts, 'iter_num'); opts.iter_num = 1e3; end
	if ~isfield(opts, 'ktype');  opts.ktype = 1; end
	if ~isfield(opts, 'ibw');  opts.ibw = -1; end

	if ~isfield(opts, 'trace'); opts.trace = 0; end
	if ~isfield(opts, 'record'); opts.record = 0; end
	if ~isfield(opts, 'itPrint'); opts.itPrint = 1; end
	if ~isfield(opts, 'ptype'); opts.ptype = 1; end 
	if ~isfield(opts, 'rtype'); opts.rtype = 1; end 

	if ~isfield(opts, 'h_pow'); opts.h_pow = d/2+2; end

	if ~isfield(opts, 'ttype'); opts.ttype = 1; end
	if ~isfield(opts, 'tau_dec'); opts.tau_dec = 1; end
	if ~isfield(opts, 'tau_itv'); opts.tau_itv = 100; end

	if ~isfield(opts, 'fun'); opts.fun = []; end 

	tau = opts.tau;
	iter_num = opts.iter_num;
	ktype = opts.ktype;
	ibw = opts.ibw;


	record = opts.record;
	itPrint = opts.itPrint;
	ptype = opts.ptype;

	X = X_init;

	if itPrint>1
		idx = 1;
	else
		idx = 0;
	end

	trace_ = [];
	if opts.trace
		switch ptype
			case 0
				trace_.H = zeros(floor(iter_num/itPrint)+idx,1);
				trace_.iter = zeros(floor(iter_num/itPrint)+idx,1);
			case 1
				trace_.test_acc = zeros(floor(iter_num/itPrint)+idx,1);
				trace_.test_llh = zeros(floor(iter_num/itPrint)+idx,1);
				trace_.iter = zeros(floor(iter_num/itPrint)+idx,1);
				trace_.time = zeros(floor(iter_num/itPrint)+idx,1);
			case 2
				trace_.ibw = zeros(floor(iter_num/itPrint)+idx,1);
				trace_.fmean = zeros(floor(iter_num/itPrint)+idx,1);
				trace_.fvar = zeros(floor(iter_num/itPrint)+idx,1);
				trace_.iter = zeros(floor(iter_num/itPrint)+idx,1);
		end
	end

	if opts.ptype == 0
		A = opts.A;
		logdetA = log(det(A));
	end

	if record == 1
	    % set up print format
	    if ispc; str1 = '  %10s'; str2 = '  %6s';
	    else     str1 = '  %10s'; str2 = '  %6s'; end
	    stra = ['%5s',str2,str2,str2,str1, str2,'\n'];
	    str_head = sprintf(stra, ...
	        'iter', 'TBA','TBA','TBA', 'TBA', 'TBA');
	    str_num = ['%4d | %+2.4e %+2.4e %+2.4e %2.1e %2.1e \n'];
	end
	
	eva_time = 0;
	for iter = 1:iter_num
		if opts.trace && (iter == 1 || mod(iter,itPrint)==0)
			switch ptype
				case 0
					tmp1 = toc;
					Sigma = cov(X);
					if cond(Sigma)<1e-12;
						break
					end
					tmp = A*Sigma;
					H = (-d-log(det(Sigma))-logdetA+trace(tmp))/2;
					eva_time = eva_time+toc-tmp1;
					trace_.H(floor(iter/itPrint)+idx) = H;
					trace_.iter(floor(iter/itPrint)+idx) = iter;
					trace_.time(floor(iter/itPrint)+idx) = toc-eva_time;
				case 1
					tmp1 = toc;
					[test_acc, test_llh] = bayeslr_evaluation(X, opts.X_test, opts.y_test);
					eva_time = eva_time+toc-tmp1;
					trace_.test_acc(floor(iter/itPrint)+idx) = test_acc;
					trace_.test_llh(floor(iter/itPrint)+idx) = test_llh;
					trace_.iter(floor(iter/itPrint)+idx) = iter;
					trace_.time(floor(iter/itPrint)+idx) = toc-eva_time;
				case 2
					trace_.ibw(floor(iter/itPrint)+idx) = ibw;
					trace_.iter(floor(iter/itPrint)+idx) = iter;
				case 3
					fX = feval(opts.fun,X);
					mean_fun = mean(fX);
					var_fun = var(fX);
					trace_.ibw(floor(iter/itPrint)+idx) = ibw;
					trace_.fmean(floor(iter/itPrint)+idx) = mean_fun;
					trace_.fvar(floor(iter/itPrint)+idx) = var_fun;
					trace_.iter(floor(iter/itPrint)+idx) = iter;
			end
			if record
		        if iter == 1 || mod(iter,20*itPrint) == 0 
		            fprintf('\n%s', str_head);
		        end
		        if iter == 1 || mod(iter,itPrint) == 0
		        	switch ptype
						case 0
							fprintf(str_num,iter,H, min(eig(Sigma)), max(eig(Sigma)),ibw,0);
						case 1
							fprintf(str_num,iter,test_acc, test_llh, 0,0,0);
						case 2
							fprintf(str_num,iter,ibw, 0, 0,0,0);
						case 3
							fprintf(str_num,iter,mean_fun, var_fun, 0,0,0);
					end
		        end
			end
		end
		
		switch opts.ttype
			case 1
				tau = opts.tau*opts.tau_dec^(floor(iter/opts.tau_itv));
			case 2
				tau = opts.tau/iter^opts.tau_dec;
		end
		grad = dlog_p(X);
		kopts = struct('iter',iter,'rtype',opts.rtype,'ibw',ibw,'h_pow',opts.h_pow,'tau',tau);
		if ktype == 0
			X = X+tau*grad-sqrt(2*tau)*randn([N,d]);
		else
			[xi, ibw_out] = dlog_p_prac(X,ktype,kopts);
			ibw = ibw_out;
			X = X+tau*grad-tau*xi;
		end
		
	end

	Xs = X;
	out = [];
	if opts.trace
		out.trace = trace_;
	end
end