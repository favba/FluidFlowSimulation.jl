function calculate_u1u2_spectrum!(Ef,u,cplane)
    ux = u.c.x
    uy = u.c.y
    # Initialize the shells to zeros
    fill!(Ef,0)
    maxdk2d = max(KX[2],KY[2])
    nshells = min(NX,NY÷2)
    @inbounds for j in YRANGE
        conjFactX=1.0
        for i in XRANGE
            k = @fastmath sqrt(KX[i]^2 + KY[j]^2)
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

function calculate_u1u2_spectrum(u,cplane,s::AbstractSimulation)
    Ef = zeros(min(NX, NY÷2))
    calculate_u1u2_spectrum!(Ef,u,cplane=1)
    return Ef
end
