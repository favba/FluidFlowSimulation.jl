@par function calculate_u1u2_spectrum!(Ef,u,cplane,s::@par(AbstractSimulation))
  ux = u.cx
  uy = u.cy
  # Initialize the shells to zeros
  fill!(Ef,0)
  dk =  max(kx[2],ky[2]) 
  nshells = min(Nx,Ny÷2 + 1)
  @inbounds for j=1:Ny
    conjFactX=1.0
    for i=1:Nx
      n=getn2d(i,j,s);
      if n <= nshells
        #magsq = a1.c[i,j,0].r*b1.c[i,j,0].r + a1.c[i,j,0].i*b1.c[i,j,0].i;
        #magsq += a2.c[i,j,0].r*b2.c[i,j,0].r + a2.c[i,j,0].i*b2.c[i,j,0].i;
        magsq = abs2(ux[i,j,cplane]) + abs2(uy[i,j,cplane])
        ee = 0.5*conjFactX * magsq / dk;
        Ef[n]+=ee;
        conjFactX=2.0;
      end
    end
  end
end

@par function calculate_u1u2_spectrum(u,cplane,s::@par(AbstractSimulation))
  Ef = zeros(min(Nx, Ny÷2 + 1))
  calculate_u1u2_spectrum!(Ef,u,cplane,s)
  return Ef
end

@inline @inbounds @par function getn2d(i,j,s::@par(AbstractSimulation))
  dk = max(kx[2],ky[2])
  K = sqrt(kx[i]^2 + ky[j]^2)
  return trunc(Int,K/dk + 1.5)
end