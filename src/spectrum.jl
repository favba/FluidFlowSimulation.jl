@par function calculate_u1u2_spectrum!(Ef,u,s::@par(AbstractSimulation))
  ux = u.cx
  uy = u.cy
  # Initialize the shells to zeros
  fill!(Ef,0)
  dk = kx[2] - kx[1] 
  @inbounds for j=1:Ny
    conjFactX=1.0
    for i=1:Nx
      n=getn2d(i,j,s);
      if n <= Nx
        #magsq = a1.c[i,j,0].r*b1.c[i,j,0].r + a1.c[i,j,0].i*b1.c[i,j,0].i;
        #magsq += a2.c[i,j,0].r*b2.c[i,j,0].r + a2.c[i,j,0].i*b2.c[i,j,0].i;
        magsq = abs2(ux[i,j,0]) + abs2(uy[i,j,0])
        ee = 0.5*conjFactX * magsq / dk;
        Ef[n]+=ee;
        conjFactX=2.0;
      end
    end
  end
end

@par function calculate_u1u2_spectrum(u,s::@par(AbstractSimulation))
  Ef = zeros(Nx)
  calculate_u1u2_spectrum!(Ef,u,s)
  return Ef
end

@inline @par function getn2d(i,j,s::@par(AbstractSimulation))
  K = sqrt(kx[i]^2 + ky[j]^2)
  return trunc(Int,K/kx[2] + 0.5)
end