
@par function realspace_dynamic_les_calculation!(s::A) where {A<:@par(AbstractSimulation)}
    real!(s.lesmodel.û)
    real!(s.lesmodel.M)
    if is_dynP_les(A)
        real!(s.lesmodel.ŵ)
    end
    @mthreads for j in TRANGE
        realspace_dynamic_les_calculation!(s,j)
    end
    return nothing
end

@par function realspace_dynamic_les_calculation!(s::A,j::Integer) where {A<:@par(AbstractSimulation)} 
    u = s.u.rr
    L = s.lesmodel.L.rr
    tau = s.lesmodel.tau.rr
    Sa = s.lesmodel.S.rr
    Δ² = s.lesmodel.Δ²
    if is_dynP_les(A)
        w = s.rhs.rr
        P = s.lesmodel.P.rr
    end

    @inbounds @msimd for i in REAL_RANGES[j]
        S = tau[i]
        Sa[i] = Δ²*norm(S)*S 
        v = u[i]
        L[i] = symouter(v,v)
        if is_dynP_les(A)
            P[i] = Δ²*Lie(S,AntiSymTen(0.5*w[i]))
        end
    end

    return nothing
end

@par function fourierspace_dynamic_les_calculation!(s::A) where {A<:@par(AbstractSimulation)}
    myfourier!(s.lesmodel.S)
    myfourier!(s.lesmodel.L)
    is_dynP_les(A) && myfourier!(s.lesmodel.P)

    @mthreads for k in ZRANGE
        fourierspace_dynamic_les_calculation!(s,k)
    end

    real!(s.lesmodel.S)
    real!(s.lesmodel.L)
    is_dynP_les(A) && real!(s.lesmodel.P)
    return nothing
end

@par function fourierspace_dynamic_les_calculation!(s::A,k::Integer) where {A<:@par(AbstractSimulation)}
    S = s.lesmodel.S.c
    L = s.lesmodel.L.c
    Δ̂² = s.lesmodel.Δ̂² 
    if is_dynP_les(A)
        P = s.lesmodel.P.c
    end

    @inbounds for j in YRANGE
        @msimd for i in XRANGE 
            G = Gaussfilter(Δ̂²,i,j,k)
            S[i,j,k] *= G
            L[i,j,k] *= G
            if is_dynP_les(A)
                P[i,j,k] *= G
            end
        end
    end
end
