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
            k = fsqrt(KX[i]^2 + KY[j]^2)
            n = round(Int, k/maxdk2d) + 1
            if n <= nshells
                magsq = abs2(ux[i,j,cplane]) + abs2(uy[i,j,cplane])
                ee = 0.5*conjFactX * magsq;
                Ef[n]+=ee;
            end
            conjFactX=2.0;
        end
    end
end

function calculate_u1u2_spectrum(u,cplane,s::AbstractSimulation)
    Ef = zeros(min(NX, NY÷2))
    calculate_u1u2_spectrum!(Ef,u,1)
    return Ef
end

function calculate_abs2!(Ef,u,cplane)
    ux = u.c.x
    uy = u.c.y
    # Initialize the shells to zeros
    fill!(Ef,0)
    @inbounds for j in YRANGE
        conjFactX=1.0
        for i in XRANGE
            magsq = abs2(ux[i,j,cplane]) + abs2(uy[i,j,cplane])
            ee = 0.5*conjFactX * magsq;
            Ef[i,j]=ee;
            conjFactX=2.0;
        end
    end
end

function calculate_u1u2_mag_spectrum!(Ef,u,cplane)
    ux = u.c.x
    uy = u.c.y
    # Initialize the shells to zeros
    fill!(Ef,0)
    maxdk2d = max(KX[2],KY[2])
    nshells = min(NX,NY÷2)
    @inbounds for j in YRANGE
        conjFactX=1.0
        for i in XRANGE
            k = fsqrt(KX[i]^2 + KY[j]^2)
            n = round(Int, k/maxdk2d) + 1
            if n <= nshells
                mag = fsqrt(abs2(ux[i,j,cplane]) + abs2(uy[i,j,cplane]))
                ee = conjFactX * mag;
                Ef[n]+=ee;
            end
            conjFactX=2.0;
        end
    end
end
