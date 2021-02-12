function [ibw_out] = BM_bandwidth(X,ibw_in, BMopts)
%%%---------------------------------------------%%%
% This solves the subproblem in the BM method
%%%---------------------------------------------%%%
	[N,d] = size(X);
	if nargin<3; BMopts = []; end
	if ~isfield(BMopts, 'tau'); BMopts.tau = 1; end

	tau = BMopts.tau;
	
	Y = X+sqrt(2*tau)*randn([N,d]);

	XY = X*X';
	X2 = sum(X.^2,2);
	X2e = repmat(X2,1,N);
	H = X2e+X2e'-2*XY;

	
	function [res] = get_obj_ibandw(ibw)
		Kxy = exp(-H*0.5*ibw);
		dxKxy = Kxy*X;
		sumKxy = sum(Kxy,2);
		xi = ibw*(dxKxy./sumKxy-X);
		X_new = X-tau*xi;
		res = MMD(X_new,Y,1);
	end

	explore_ratio = 1.1;
	obj_ibw_in = get_obj_ibandw(ibw_in);
	epsi = 1e-6;
	grad_ibw_in = (get_obj_ibandw(ibw_in+epsi)-obj_ibw_in)/epsi;
	if grad_ibw_in<0
		ibw_1 = ibw_in*explore_ratio;
	else
		ibw_1 = ibw_in/explore_ratio;
	end
	obj_ibw_1 = get_obj_ibandw(ibw_1);
	slope_ibw = (obj_ibw_1-obj_ibw_in)/(ibw_1-ibw_in);
	ibw_2 = (ibw_in * slope_ibw - 0.5 * grad_ibw_in * (ibw_1 + ibw_in)) / (slope_ibw - grad_ibw_in);
	obj_ibw_2 = get_obj_ibandw(ibw_2);
	if ~isnan(ibw_2)&&ibw_2>0
		if obj_ibw_1<obj_ibw_in
			if obj_ibw_2<obj_ibw_1
				ibw_out = ibw_2;
			else
				ibw_out = ibw_1;
			end
		else
			if obj_ibw_2<obj_ibw_in
				ibw_out = ibw_2;
			else
				ibw_out = ibw_in;
			end
		end
	else
		if obj_ibw_1<obj_ibw_in
			ibw_out = ibw_1;
		else
			ibw_out = ibw_in;
		end
	end
	

end