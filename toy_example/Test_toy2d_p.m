%%%---------------------------------------------%%%
% This implements the 2D toy example in the numerical experiment.
%%%---------------------------------------------%%%

clear all;

addpath('solvers');
addpath('utils');
addpath(genpath('dlogp'));

p = @(X) p_toy2d_p(X);

xnum = 100; ynum = 100;
xmin = -4; ymin = -4;
xmax = 4; ymax = 4;
x = ([1:xnum]/xnum-0.5)*(xmax-xmin)+(xmax+xmin)/2;
y = ([1:ynum]/ynum-0.5)*(ymax-ymin)+(ymax+ymin)/2;
[X,Y] = meshgrid(x,y);
Z = zeros(xnum,ynum,2);
Z(:,:,1)=X;
Z(:,:,2)=Y;
Z_aux = reshape(Z,[xnum*ynum,2]);
Z_ans = p(Z_aux);
Z_plot = reshape(Z_ans,[xnum,ynum]);

rng(2);
N = 100;
X_init0 = randn([N,2])+[0,10];
dlog_p = @dlog_p_toy2d_p;

iters = [20,40,80,160];

save_root = '../Particle_result/result/toy2d_p/';
if ~exist(save_root,'dir')
    mkdir(save_root)
end

algorithms = {};
WGF.name = 'W-GF';
WGF.Sname = 'WGD-MED';
WGF.opts = struct('tau',0.1,'iter_num',1,'ktype',1,'ibw',-1,'ptype',2);

WGF_BM.name = 'W-GF';
WGF_BM.Sname = 'WGD-BM';
WGF_BM.opts = struct('tau',0.1,'iter_num',1,'ktype',6,'ibw',-1,'ptype',2);


SVGD.name = 'SVGD';
SVGD.Sname = 'SVGD';
SVGD.opts = struct('tau',0.1,'iter_num',1,'ktype',3,'ibw',1,'ptype',2,'adagrad',1);


algorithms = {WGF,WGF_BM,SVGD};

clf
figure(1);

l_alg = length(algorithms);
l_iter = length(iters);

iters_diff = diff(iters);

for j =1:length(algorithms)
	X_init = X_init0;
	for i = 1:length(iters)
		iter = iters(i);
		opts = algorithms{j}.opts;
		if i == 1
			opts.iter_num = iter;
		else
			X_init = Xout;
			opts.iter_num = iters_diff(i-1);
		end

		switch algorithms{j}.name
			case 'W-GF'
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
		title(strcat(algorithms{j}.Sname,' Iter: ',mat2str(iter)),'FontSize',12);
		% save_path1 = strcat(save_root,algorithms{j}.Sname,'_iter',mat2str(iter),'_2d.png');
		% saveas(gcf,save_path1);

	end
end

set(gcf,'position',[0,0,200*length(iters),200*length(algorithms)]);
save_path = strcat(save_root,'toy_2d.png');
saveas(gcf,save_path);



