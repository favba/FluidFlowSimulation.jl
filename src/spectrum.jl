@par function calculate_u1u2_spectrum!(Ef,u,cplane,s::@par(AbstractSimulation))
    ux = u.cx
    uy = u.cy
    # Initialize the shells to zeros
    fill!(Ef,0)
    dk =  kx[2] 
    maxdk2d = max(kx[2],ky[2])
    nshells = min(Nx,Ny÷2)
    @inbounds for j=1:Ny
        conjFactX=1.0
        for i=1:Nx
            K = sqrt(kx[i]^2 + ky[j]^2)
            n = round(Int, K/maxdk2d) + 1
            if n <= nshells
                magsq = abs2(ux[i,j,cplane]) + abs2(uy[i,j,cplane])
                ee = 0.5*conjFactX * magsq / maxdk2d;
                Ef[n]+=ee;
                conjFactX=2.0;
            end
        end
    end
end

@par function calculate_u1u2_spectrum(u,cplane,s::@par(AbstractSimulation))
    Ef = zeros(min(Nx, Ny÷2))
    calculate_u1u2_spectrum!(Ef,u,cplane,s)
    return Ef
end
