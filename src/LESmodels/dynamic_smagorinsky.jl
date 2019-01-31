
@par function realspace_dynamic_les_calculation!(s::A) where {A<:@par(AbstractSimulation)}
    real!(s.lesmodel.û)
    real!(s.lesmodel.M)
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

    @inbounds @msimd for i in REAL_RANGES[j]
        S = tau[i]
        Sa[i] = Δ²*norm(S)*S 
        v = u[i]
        L[i] = symouter(v,v)
    end

    return nothing
end

@par function fourierspace_dynamic_les_calculation!(s::A) where {A<:@par(AbstractSimulation)}
    myfourier!(s.lesmodel.S)
    dealias!(s.lesmodel.S)
    myfourier!(s.lesmodel.L)
    dealias!(s.lesmodel.L)
    @mthreads for k in ZRANGE
        fourierspace_dynamic_les_calculation!(s,k)
    end
    real!(s.lesmodel.S)
    real!(s.lesmodel.L)
    return nothing
end

@par function fourierspace_dynamic_les_calculation!(s::A,k::Integer) where {A<:@par(AbstractSimulation)}
    S = s.lesmodel.S.c
    L = s.lesmodel.L.c
    Δ̂² = s.lesmodel.Δ̂² 
    @inbounds for j in YRANGE
        @msimd for i in XRANGE 
            G = Gaussfilter(Δ̂²,i,j,k)
            S[i,j,k] *= G
            L[i,j,k] *= G
        end
    end
end
