function [dlog_p] = dlog_p_dw2(X_in)
	X = X_in;
	[N,d] = size(X);
	norm_X = sqrt(sum(X.^2,2));
	X1 = X(:,1);
	Z2 = zeros(N,1);
	expX11 = exp(-2*(X1-3).^2);
	expX12 = exp(-2*(X1+3).^2);
	dexpX1 = 4*((X1-3).*expX11+(X1+3).*expX12)./(expX11+expX12);
	dlog_p = -4*X./norm_X.*(norm_X-3)-[dexpX1,Z2]-0.0001*X;

end