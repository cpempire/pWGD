function [Xs,out] = SVGD_m(X_init, dlog_p, opts)
%%%---------------------------------------------%%%
% This implements Stein Variational Gradient Descent(SVGD).
% Stein Variational Gradient Descent (SVGD): A General Purpose Bayesian Inference Algorithm. NIPS, 2016.
% Qiang Liu and Dilin Wang. 
%
% The main part of this code is based on
% https://github.com/DartML/Stein-Variational-Gradient-Descent
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
%
%
% 
%%%---------------------------------------------%%%
	tic;
	if nargin<3; opts = []; end
	if ~isfield(opts, 'tau'); opts.tau = 0.1;  end
	if ~isfield(opts, 'iter_num'); opts.iter_num = 1e3; end
	if ~isfield(opts, 'ktype');  opts.ktype = 1; end
	if ~isfield(opts, 'ibw');  opts.ibw = -1; end

	if ~isfield(opts, 'trace'); opts.trace = 0; end
	if ~isfield(opts, 'record'); opts.record = 0; end
	if ~isfield(opts, 'itPrint'); opts.itPrint = 1; end
	if ~isfield(opts, 'adagrad'); opts.adagrad = 0; end
	if ~isfield(opts, 'ptype'); opts.ptype = 1; end 

	tau = opts.tau;
	iter_num = opts.iter_num;
	ktype = opts.ktype;

	record = opts.record;
	itPrint = opts.itPrint;

	ptype = opts.ptype;

	[N,d] = size(X_init);
	X = X_init;

	trace_ = [];
	if opts.trace
		switch ptype
			case 0
				trace_ = zeros(floor(iter_num/itPrint)+1,1);
			case 1
				trace_.test_acc = zeros(floor(iter_num/itPrint)+1,1);
				trace_.test_llh = zeros(floor(iter_num/itPrint)+1,1);
				trace_.iter = zeros(floor(iter_num/itPrint)+1,1);
				trace_.time = zeros(floor(iter_num/itPrint)+1,1);
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
	

	xi_history = 0;
	coef = 0.9;
	xi = 0;

	eva_time = 0;
	for iter = 1:iter_num
		if opts.trace && (iter == 1 || mod(iter,itPrint)==0)
			switch ptype
				case 0
					Sigma = cov(X);
					if cond(Sigma)<1e-12;
						break
					end
					tmp = A*Sigma;
					H = (-d-log(det(Sigma))-logdetA+trace(tmp))/2;
					trace_(floor(iter/itPrint)+1) = H;
				case 1
					tmp1 = toc;
					[test_acc, test_llh] = bayeslr_evaluation(X, opts.X_test, opts.y_test);
					eva_time = eva_time+toc-tmp1;
					trace_.test_acc(floor(iter/itPrint)+1) = test_acc;
					trace_.test_llh(floor(iter/itPrint)+1) = test_llh;
					trace_.iter(floor(iter/itPrint)+1) = iter;
					trace_.time(floor(iter/itPrint)+1) = toc-eva_time;
			end
			if record
		        if iter == 1 || mod(iter,20*itPrint) == 0 
		            fprintf('\n%s', str_head);
		        end
		        if iter == 1 || mod(iter,itPrint) == 0
		        	switch ptype
						case 0
							fprintf(str_num,iter,H, min(eig(Sigma)), max(eig(Sigma)),0,0);
						case 1
							fprintf(str_num,iter,test_acc, test_llh, 0,0,0);
					end
		        end
			end
		end
		grad = dlog_p(X);
		XY = X*X';
		X2 = sum(X.^2,2);
		X2e = repmat(X2,1,N);
		H = X2e+X2e'-2*XY;
		switch ktype
			case 1
				ibw = 1/(0.5*median(H(:))/log(N+1));
			case 2
				if iter == 1
					ibw = 1/(0.5*median(H(:))/log(N+1));
				end
				HEopts.rtype = 1;
				ibw = HE_bandwidth(X,ibw,HEopts);
			case 3
				ibw = opts.ibw;
		
		end
		Kxy = exp(-H*0.5*ibw);
		dxKxy = -Kxy*X;
		sumKxy = sum(Kxy,2);
		for i=1:d
			dxKxy(:,i)=dxKxy(:,i) + X(:,i).*sumKxy;
		end
		dxKxy = dxKxy*ibw;
		xi = (Kxy*grad + dxKxy)/N;
		if opts.adagrad
			if xi_history ==0
				xi_history =  xi_history+xi.^2;
			else
				xi_history =  coef*xi_history + (1-coef)*xi.^2;
				%xi_history =  xi_history + xi.^2;
			end
			X = X+tau*xi./(sqrt(xi_history)+1e-6);
		else
			X = X+tau*xi;
		end
		
	end

	Xs = X;
	out = [];
	if opts.trace
		out.trace = trace_;
	end
end