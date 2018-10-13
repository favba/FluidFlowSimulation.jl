@par function calculate_u1u2_spectrum!(Ef,u,cplane,s::@par(AbstractSimulation))
    ux = u.cx
    uy = u.cy
    # Initialize the shells to zeros
    fill!(Ef,0)
    dk =  KX[2] 
    maxdk2d = max(KX[2],KY[2])
    nshells = min(NX,NY÷2)
    @inbounds for j in YRANGE
        conjFactX=1.0
        for i in XRANGE
            k = sqrt(KX[i]^2 + KY[j]^2)
            n = round(Int, k/maxdk2d) + 1
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
    Ef = zeros(min(NX, NY÷2))
    calculate_u1u2_spectrum!(Ef,u,cplane,s)
    return Ef
end
