#-09.03.2025-#
#-Sample Julia code for Algorithm 2: Newton's algorithm for variable refractive index, using a spectral method (ApproxFun) to solve the BVPs with appropriate scaling, starting from asymptotics-#
#-Paper: Bouchra Bensiali and Stefan Sauter, Computation of whispering gallery modes for spherical symmetric heterogeneous Helmholtz problems with piecewise smooth refractive index-#
#-Authors: BB&SS-#

using LinearAlgebra, ApproxFun, SpecialFunctions, Plots;

function Newton_var(xi,n1,n2,m,eps,lmax,k0)
    l=0;
    k=k0;

    if iseven(m)
        operator=lneumann()
    else
        operator=ldirichlet()
    end

    d1=Domain(0..xi);
    x1=Fun(d1);
    D1=Derivative(d1);

    d2=Domain(xi..1);
    x2=Fun(d2)
    D2=Derivative(d2);

    #loop initialization l=0
    L1=x1^2*D1^2+x1*D1+((k*n1(x1))^2*x1^2-m^2)*I;
    L2=x2^2*D2^2+x2*D2+((k*n2(x2))^2*x2^2-m^2)*I;

    #solve f11k, f22k
    f11k = [operator;rneumann()-(im*k*n1(xi))*rdirichlet();L1] \ [0,1,0]; 
    f22k = [-lneumann()-(im*k*n2(xi))*ldirichlet();rneumann()-(k*n2(1))*1/2*((hankelh1(m-1,k*n2(1))-hankelh1(m+1,k*n2(1)))/hankelh1(m,k*n2(1)))*rdirichlet();L2] \ [1,0,0];
    
    #solve dkf11k, dkf22k
    dkf11k = [operator;rneumann()-(im*k*n1(xi))*rdirichlet();L1] \ [0,im*n1(xi)*f11k(xi),-x1^2*2*k*(n1(x1))^2*f11k(x1)];
    bc1 = (1/(4*hankelh1(m,n2(1)*k)^2))*n2(1)*(-n2(1)*k*hankelh1(-1 + m, n2(1)*k)^2 +n2(1)*k*hankelh1(-2 + m, n2(1)*k)*hankelh1(m, n2(1)*k) - 2*n2(1)* k *hankelh1(m, n2(1)* k)^2 -2* hankelh1(m, n2(1)* k) *hankelh1(1 + m, n2(1)* k) -n2(1)* k* hankelh1(1 + m, n2(1)* k)^2 +2 *hankelh1(-1 + m, n2(1)* k)* (hankelh1(m, n2(1)* k) + n2(1)* k *hankelh1(1 + m, n2(1)* k)) + n2(1)* k *hankelh1(m, n2(1)* k)* hankelh1(2 + m, n2(1)* k)) * f22k(1);
    dkf22k = [-lneumann()-(im*k*n2(xi))*ldirichlet();rneumann()-(k*n2(1))*1/2*((hankelh1(m-1,k*n2(1))-hankelh1(m+1,k*n2(1)))/hankelh1(m,k*n2(1)))*rdirichlet();L2] \ [im*n2(xi)*f22k(xi),bc1,-x2^2*2*k*(n2(x2))^2*f22k(x2)];

    #determinant and derdeterminant
    Determinant = im*k*(n2(xi)+n1(xi))*f11k(xi)*f22k(xi)+f11k(xi)+f22k(xi);
    DerDeterminant = im*(n2(xi)+n1(xi))*(f11k(xi)*f22k(xi)+k*dkf11k(xi)*f22k(xi)+k*f11k(xi)*dkf22k(xi))+dkf11k(xi)+dkf22k(xi);
    #scaling by 1/(k*c1k*c2k)
    c1k = 2/(k*n1(xi)*(besselj(m-1,k*n1(xi)*xi)-2*im*besselj(m,k*n1(xi)*xi)-besselj(m+1,k*n1(xi)*xi)));
    c2k = -2/(k*n2(xi)*(hankelh1(m-1,k*n2(xi)*xi)+2*im*hankelh1(m,k*n2(xi)*xi)-hankelh1(m+1,k*n2(xi)*xi)));
    dkCk = 1/8 *k *n1(xi)* n2(xi)* (-4 *(besselj(-1 + m, k* n1(xi)* xi) - 2* im* besselj(m, k* n1(xi)* xi) -besselj(1 + m, k *n1(xi)* xi)) *(hankelh1(-1 + m, k *n2(xi) *xi) +2* im* hankelh1(m, k *n2(xi)* xi) - hankelh1(1 + m, k *n2(xi)* xi)) -k *n1(xi)* xi* (besselj(-2 + m, k *n1(xi)* xi) - 2* im* besselj(-1 + m, k* n1(xi)* xi) -2 *besselj(m, k *n1(xi)* xi) + 2 *im* besselj(1 + m, k *n1(xi)* xi) +besselj(2 + m, k *n1(xi)* xi)) *(hankelh1(-1 + m, k *n2(xi)* xi) +2 *im* hankelh1(m, k *n2(xi)* xi) - hankelh1(1 + m, k *n2(xi)* xi)) -k *n2(xi)* xi* (besselj(-1 + m, k *n1(xi)* xi) - 2 *im* besselj(m, k *n1(xi)* xi) -besselj(1 + m, k *n1(xi) *xi)) *(hankelh1(-2 + m, k *n2(xi)* xi) +2 *im* hankelh1(-1 + m, k *n2(xi)* xi) - 2 *hankelh1(m, k *n2(xi)* xi) -2 *im* hankelh1(1 + m, k *n2(xi) *xi) + hankelh1(2 + m, k *n2(xi)* xi)));
    Determinant_scal = 1/(k*c1k*c2k)*Determinant;
    DerDeterminant_scal = 1/(k*c1k*c2k)*DerDeterminant - (1/k^2)/(c1k*c2k)*Determinant + 1/k*dkCk*Determinant;

    while (abs(Determinant_scal) > eps) && (l < lmax)
        
        k=k-Determinant_scal/DerDeterminant_scal;

        l=l+1;

        L1=x1^2*D1^2+x1*D1+((k*n1(x1))^2*x1^2-m^2)*I;
        L2=x2^2*D2^2+x2*D2+((k*n2(x2))^2*x2^2-m^2)*I;

        #solve f11k, f22k
        f11k = [operator;rneumann()-(im*k*n1(xi))*rdirichlet();L1] \ [0,1,0]; 
        f22k = [-lneumann()-(im*k*n2(xi))*ldirichlet();rneumann()-(k*n2(1))*1/2*((hankelh1(m-1,k*n2(1))-hankelh1(m+1,k*n2(1)))/hankelh1(m,k*n2(1)))*rdirichlet();L2] \ [1,0,0];
        
        #solve dkf11k, dkf22k
        dkf11k = [operator;rneumann()-(im*k*n1(xi))*rdirichlet();L1] \ [0,im*n1(xi)*f11k(xi),-x1^2*2*k*(n1(x1))^2*f11k(x1)];
        bc1 = (1/(4*hankelh1(m,n2(1)*k)^2))*n2(1)*(-n2(1)*k*hankelh1(-1 + m, n2(1)*k)^2 +n2(1)*k*hankelh1(-2 + m, n2(1)*k)*hankelh1(m, n2(1)*k) - 2*n2(1)* k *hankelh1(m, n2(1)* k)^2 -2* hankelh1(m, n2(1)* k) *hankelh1(1 + m, n2(1)* k) -n2(1)* k* hankelh1(1 + m, n2(1)* k)^2 +2 *hankelh1(-1 + m, n2(1)* k)* (hankelh1(m, n2(1)* k) + n2(1)* k *hankelh1(1 + m, n2(1)* k)) + n2(1)* k *hankelh1(m, n2(1)* k)* hankelh1(2 + m, n2(1)* k)) * f22k(1);
        dkf22k = [-lneumann()-(im*k*n2(xi))*ldirichlet();rneumann()-(k*n2(1))*1/2*((hankelh1(m-1,k*n2(1))-hankelh1(m+1,k*n2(1)))/hankelh1(m,k*n2(1)))*rdirichlet();L2] \ [im*n2(xi)*f22k(xi),bc1,-x2^2*2*k*(n2(x2))^2*f22k(x2)];

        #determinant and derdeterminant
        Determinant = im*k*(n2(xi)+n1(xi))*f11k(xi)*f22k(xi)+f11k(xi)+f22k(xi);
        DerDeterminant = im*(n2(xi)+n1(xi))*(f11k(xi)*f22k(xi)+k*dkf11k(xi)*f22k(xi)+k*f11k(xi)*dkf22k(xi))+dkf11k(xi)+dkf22k(xi);
        #scaling by 1/(k*c1k*c2k)
        c1k = 2/(k*n1(xi)*(besselj(m-1,k*n1(xi)*xi)-2*im*besselj(m,k*n1(xi)*xi)-besselj(m+1,k*n1(xi)*xi)));
        c2k = -2/(k*n2(xi)*(hankelh1(m-1,k*n2(xi)*xi)+2*im*hankelh1(m,k*n2(xi)*xi)-hankelh1(m+1,k*n2(xi)*xi)));
        dkCk = 1/8 *k *n1(xi)* n2(xi)* (-4 *(besselj(-1 + m, k* n1(xi)* xi) - 2* im* besselj(m, k* n1(xi)* xi) -besselj(1 + m, k *n1(xi)* xi)) *(hankelh1(-1 + m, k *n2(xi) *xi) +2* im* hankelh1(m, k *n2(xi)* xi) - hankelh1(1 + m, k *n2(xi)* xi)) -k *n1(xi)* xi* (besselj(-2 + m, k *n1(xi)* xi) - 2* im* besselj(-1 + m, k* n1(xi)* xi) -2 *besselj(m, k *n1(xi)* xi) + 2 *im* besselj(1 + m, k *n1(xi)* xi) +besselj(2 + m, k *n1(xi)* xi)) *(hankelh1(-1 + m, k *n2(xi)* xi) +2 *im* hankelh1(m, k *n2(xi)* xi) - hankelh1(1 + m, k *n2(xi)* xi)) -k *n2(xi)* xi* (besselj(-1 + m, k *n1(xi)* xi) - 2 *im* besselj(m, k *n1(xi)* xi) -besselj(1 + m, k *n1(xi) *xi)) *(hankelh1(-2 + m, k *n2(xi)* xi) +2 *im* hankelh1(-1 + m, k *n2(xi)* xi) - 2 *hankelh1(m, k *n2(xi)* xi) -2 *im* hankelh1(1 + m, k *n2(xi) *xi) + hankelh1(2 + m, k *n2(xi)* xi)));
        Determinant_scal = 1/(k*c1k*c2k)*Determinant;
        DerDeterminant_scal = 1/(k*c1k*c2k)*DerDeterminant - (1/k^2)/(c1k*c2k)*Determinant + 1/k*dkCk*Determinant;

        #print("$l, $k, $Determinant_scal, $DerDeterminant_scal\n")
    end;
    k0, k, l, Determinant_scal, DerDeterminant_scal;
    AbsDeterminant_scal=abs(Determinant_scal);
    AbsDerDeterminant_scal=abs(DerDeterminant_scal);

    print("$m & $k0 & $l & $k & $AbsDeterminant_scal & $AbsDerDeterminant_scal \n")

end;

#test
xi=0.5;
n2(x)=1;
#example: special case with explicit solution (Luneburg lens) -> Table S-18 in Supplementary Material
n1(x)=sqrt(2-x^2);

#Plot
x=range(0,0.5,length=100);
plot1=plot(x,n1);
display(plot(x,n1))


lmax=2000;eps=1e-6;
#loop on m, k0 given by asymptotics
Nb=60; mmin=0; mmax=60;
for n in 1:Nb
    m=mmin+(mmax-mmin)*n/Nb;
    k0=m/(xi*n1(xi))
    Newton_var(xi,n1,n2,m,eps,lmax,k0)
end;