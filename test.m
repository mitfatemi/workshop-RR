clear all
close all
clc

path(path, 'toolbox/');
path(path, 'E-Spline/');
path(path, 'results/');
path(path, 'Data/');
path(path, genpath('../../Splines (Loic Baboulaz)/'));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate the algebraic curve and its indicator function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load bounded_curve.mat  %contains some stored curve parameters

ind     = 3;
c       = C(ind,:)';
param.lim   = Lim(ind,:);

x_lim   = param.lim;
y_lim   = param.lim;



n       = 4;
Res     = 1e-2; %discretization resolution
Scale   = 1/Res;

param.c = c;
param.grid  = param.lim(1) : Res : param.lim(2)-Res;


[HR , c]    = algebraic_curve(n , param);  %the HR shape

figure(1)
imshow(HR)
title(sprintf('Original shape with an algebraic boundary of order %d', n))
xlabel('x'), ylabel('y')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
SplineOrder = 2*n;
alpha   = 0;
Quant = 0;
PlotOrNot = 0;
snr_lr = 0;
Support = [param.lim(1)-SplineOrder , param.lim(2)+SplineOrder];


extended_grid = Support(1):Res:Support(2)-Res;
ind1 = find(abs(extended_grid-param.lim(1))<1e-4);
ind2 = find(abs(extended_grid-param.lim(2))<1e-4);


L = (Support(2)-Support(1)-SplineOrder);  % number of involved sampling kernels




kernel = Espline(SplineOrder,0,Scale,0);



KernelMatrix = zeros(L , numel(extended_grid));
for i=1:L
    KernelMatrix(i,(i-1)*Scale+1:(i+SplineOrder)*Scale) = kernel;
end

ind = find(abs(extended_grid)<1e-4);
KernelMatrix_integral = (cumsum(KernelMatrix,2) - repmat(sum(KernelMatrix(:,1:ind),2) , size(extended_grid)))*Res;


% SF_coef = StrangFix_coefficients(SplineOrder,0,Scale,[param.lim(1),param.lim(2)],0);
if PlotOrNot
    figure
    for i = 1:2*n+1
        plot(extended_grid(ind1:ind2),SF_coef(i,:)*KernelMatrix(:,ind1:ind2),'b'),
        hold on,
        plot(extended_grid(ind1:ind2),extended_grid(ind1:ind2).^(i-1),'r')
        hold off
        shg,
        pause
    end
end

% b = diag([1 : 2*n+1]) * SF_coef;
% b = b';
% b = b(:)';

if PlotOrNot
    figure
    for i = 0:2*n
        plot(extended_grid(ind1:ind2) , b(L*i+1:L*(i+1))*KernelMatrix_integral(:,ind1:ind2),'b')
        hold on,
        plot(extended_grid(ind1:ind2),extended_grid(ind1:ind2).^(i+1),'r')
        hold off
        shg
        pause
    end
end
if PlotOrNot
    figure
    for i = 1:2*n-1
        f1 = ((extended_grid(ind1:ind2).^(2*n-i)) .* (b(L*i+1:L*(i+1))*KernelMatrix_integral(:,ind1:ind2)));
        f2 = (b(end-L+1:end)*KernelMatrix_integral(:,ind1:ind2));
        %     plot(grid(:,ind1:ind2),f1,'b')
        %     hold on
        %     plot(grid(:,ind1:ind2),f2,'r')
        %     hold off
        plot(extended_grid(ind1:ind2),f1-f2)
        shg
        pause
    end
end

num_var = L*(2*n);

const = @(b) const_func(b,L,KernelMatrix_integral);
fun = @(b) min_func(b,n,L,extended_grid,KernelMatrix_integral,Res);

options = optimoptions('fmincon','Display','iter','TolX' , 1e-8, 'MaxFunEvals', 2000000,'MaxIter',1000);

b0 = rand(1,num_var);
[b,fval] = fmincon(fun , b0 , [] , [] , [] ,[] , [] , [], const , options);

SF_coef = reshape(b,[L,2*n])';

if PlotOrNot
    figure,
    for i = 1:2*n-1
        plot(extended_grid, extended_grid .* (SF_coef(i,:)*KernelMatrix_integral),'b')
        hold on
        plot(extended_grid, SF_coef(i+1,:)*KernelMatrix_integral,'r')
        hold off
        shg
        pause
    end
        
end


SF_coef  = SF_coef *15;
LR = KernelMatrix(:,ind1:ind2-1)*HR(end:-1:1,:)*KernelMatrix(:,ind1:ind2-1)';

CM = zeros(SplineOrder+1,SplineOrder+1);

for i = 1:SplineOrder+1
    for j=1:SplineOrder+1
        CM(i,j) = SF_coef(i,:) * LR * SF_coef(j,:)';    
    end
end

[c_reg , c_sys] = polynomial_from_moments(CM , CM , n , alpha);

param.c     = c_reg; %the solution of minimization problem with a regularizer
[HR_reg, ~] = algebraic_curve(n , param);
figure(3),imshow(HR_reg),shg
title('Estimation with the regularizer')
xlabel('x'), ylabel('y')

% HR_error = norm(double(HR)-double(HR_reg),'fro')/norm(double(HR),'fro')

param.c     = c_sys;  %the solution of the minimization problem without the regularizer
[HR_sys, ~] = algebraic_curve(n , param);
figure(4),imshow(HR_sys),shg
title('Estimation without the regularizer')
xlabel('x'), ylabel('y')

figure(5),imshow(abs(HR-HR_reg)),shg
title('Error of the estimation with the regularizer')
xlabel('x'), ylabel('y')

figure(6),imshow(abs(HR-HR_sys)),shg
title('Error of the estimation without the regularizer')
xlabel('x'), ylabel('y')


