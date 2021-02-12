function p = p_toy2d(X)
	
	p = exp(-2*(sqrt(sum(X.^2,2))-3).^2)...
     	.*(exp(-2*(X(:,1)-3).^2)+exp(-2*(X(:,1)+3).^2))...
        .*exp(-0.0001/2*sum(X.^2,2));

end