%%%---------------------------------------------%%%
% This implements the double banana example in the numerical experiment.
%%%---------------------------------------------%%%

clear all;
addpath('solvers');
addpath('utils');
addpath(genpath('dlogp'));

rng(1);
N = 100;
a = 1;
b = 100;
d = 2;

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


X_init = randn([N,2]);
dlog_p = @(X) dlog_p_double_banana(X,obs,prior);
dhess_log_p = @(X) dhess_log_p_double_banana(X,obs,prior);

p = @(X) p_double_banana(X,obs,prior);

xnum = 100; ynum = 100;
xmin = -2; ymin = -2;
xmax = 2; ymax = 2;
x = ([1:xnum]/xnum-0.5)*(xmax-xmin)+(xmax+xmin)/2;
y = ([1:ynum]/ynum-0.5)*(ymax-ymin)+(ymax+ymin)/2;
[X,Y] = meshgrid(x,y);
Z = zeros(xnum,ynum,2);
Z(:,:,1)=X;
Z(:,:,2)=Y;
Z_aux = reshape(Z,[xnum*ynum,2]);
Z_ans = p(Z_aux);
Z_plot = reshape(Z_ans,[xnum,ynum]);

iters = [2,5,10,20];


save_root = '../Particle_result/result/WN_double_banana/';
if ~exist(save_root,'dir')
    mkdir(save_root)
end

algorithms = {};
WGF.name = 'WGF';
WGF.Sname = 'WGF-MED';
WGF.opts = struct('tau',3e-3,'iter_num',1,'ktype',1,'ibw',-1,'ptype',2);

WGF_BM.name = 'WGF';
WGF_BM.Sname = 'WGF-BM';
WGF_BM.opts = struct('tau',3e-3,'iter_num',1,'ktype',6,'ibw',-1,'ptype',2);

SVGD.name = 'SVGD';
SVGD.Sname = 'SVGD';
SVGD.opts = struct('tau',0.1,'iter_num',1,'ktype',1,'ptype',2,'adagrad',1);

algorithms = {WGF,WGF_BM,SVGD};
%algorithms = {WNa};

clf
figure(1);

l_alg = length(algorithms);
l_iter = length(iters);

for j =1:length(algorithms)
	for i = 1:length(iters)
		iter = iters(i);

		opts = algorithms{j}.opts;
		opts.iter_num = iter;

		switch algorithms{j}.name
			case 'WGF'
				[Xout,out] = WGF_m(X_init, dlog_p, opts);
			case 'SVGD'
				[Xout,out] = SVGD_m(X_init, dlog_p, opts);
		end

		subplot('Position',[(i-0.85)/(l_iter+0.1),(l_alg-j+0.2)/(l_alg+0.2),0.8/(l_iter+0.1),0.8/(l_alg+0.2)])
		contourf(X,Y,Z_plot,7);
		colormap('white')
		hold on;
		Xp = Xout(:,1);
		Yp = Xout(:,2);
		hp = scatter(Xp,Yp,10,'filled');
		hold off;
		% set(gcf,'position',[0,0,480,400]);
		alpha(hp,0.9);
		hold off;
		title(strcat(algorithms{j}.Sname,' Iter: ',mat2str(iter)),'FontSize',12);
		% save_path1 = strcat(save_root,algorithms{j}.Sname,'_iter',mat2str(iter),'_db.png');
		% saveas(gcf,save_path1);

	end
end

set(gcf,'position',[0,0,240*length(iters),200*length(algorithms)]);
save_path = strcat(save_root,'toy_db.png');
saveas(gcf,save_path);



