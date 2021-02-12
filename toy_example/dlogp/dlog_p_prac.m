function [X_out,ibw_out] = dlog_p_prac(X, ktype, kopts)
	if nargin<2; ktype = 1; end
	if nargin<3; kopts = []; end
	if ~isfield(kopts, 'ibw'); kopts.ibw = -1; end
	if ~isfield(kopts, 'iter'); kopts.iter = 1; end
	if ~isfield(kopts, 'rtype'); kopts.rtype = 1; end

	[N,d] = size(X);

	switch ktype
		case 1
			ibw_out = kopts.ibw;
			XY = X*X';
			X2 = sum(X.^2,2);
			X2e = repmat(X2,1,N);
			H = X2e+X2e'-2*XY;
			if kopts.ibw == -1
				ibw = 1/(0.5*median(H(:))/log(N+1));
			else
				ibw = kopts.ibw;
			end
			Kxy = exp(-H*0.5*ibw);
			dxKxy = -Kxy*X;
			sumKxy = sum(Kxy,2);
			for i=1:d
				dxKxy(:,i)=dxKxy(:,i) + X(:,i).*sumKxy;
			end
			dxKxy = dxKxy*ibw;
			X_out = -dxKxy./sumKxy;
		case 2
			ibw_out = kopts.ibw;
			Xtmp = reshape(X,[N,1,d])-reshape(X,[1,N,d]);
			H = sum(Xtmp.^2,3);
			if kopts.ibw == -1
				ibw = 1/(0.5*median(H(:))/log(N+1));
			end
			Gxy = exp(-H*0.5*ibw);
			Kxy = Gxy./sqrt(sum(Gxy,1));
			dxKxy = -Xtmp.*(Kxy.*ibw);
			X_out = dxKxy./sumKxy;
		case 3
			ibw_out = kopts.ibw;
			Sigma = cov(X);
			mX = mean(X,1);
			X_out = -(X-mX)*inv(Sigma);
		case 4
			ibw_out = kopts.ibw;
			Xtmp = reshape(X,[N,1,d])-reshape(X,[1,N,d]);
			H = sum(Xtmp.^2,3);
			if kopts.ibw == -1
				ibw = 1/(0.5*median(H(:))/log(N+1));
			end
			Kxy = exp(-H*0.5*ibw);
			sumxKxy = sum(Kxy,1);
			sumyKxy = sum(Kxy,2);
			dxKxy = -Xtmp.*(Kxy.*ibw);
			xi = sum(dxKxy./sumxKxy,2)+sum(dxKxy,2)./sumyKxy;
			X_out = reshape(xi, [N,d]);
		case 5
			XY = X*X';
			X2 = sum(X.^2,2);
			X2e = repmat(X2,1,N);
			H = X2e+X2e'-2*XY;
			ibw = kopts.ibw;
			if kopts.iter == 1
				ibw = 1/(0.5*median(H(:))/log(N+1));
			end
			ibw_out = HE_bandwidth(X,ibw,kopts);
			Kxy = exp(-H*0.5*ibw_out);
			dxKxy = -Kxy*X;
			sumKxy = sum(Kxy,2);
			for i=1:d
				dxKxy(:,i)=dxKxy(:,i) + X(:,i).*sumKxy;
			end
			dxKxy = dxKxy*ibw_out;
			X_out = -dxKxy./sumKxy;
		case 6
			XY = X*X';
			X2 = sum(X.^2,2);
			X2e = repmat(X2,1,N);
			H = X2e+X2e'-2*XY;
			ibw = kopts.ibw;
			if kopts.iter == 1
				ibw = 1/(0.5*median(H(:))/log(N+1));
			end
			ibw_out = BM_bandwidth(X,ibw,kopts);
			Kxy = exp(-H*0.5*ibw_out);
			dxKxy = -Kxy*X;
			sumKxy = sum(Kxy,2);
			for i=1:d
				dxKxy(:,i)=dxKxy(:,i) + X(:,i).*sumKxy;
			end
			dxKxy = dxKxy*ibw_out;
			X_out = -dxKxy./sumKxy;
	end