function dynamic_les_coefficient_calculation!(s)
    dynamic_les_calculation_p1!(s)
    fourierspace_dynamic_les_calculation!(s)
    dynamic_les_calculation_p2!(s)
    dynamic_les_calculation_p3!(s)
    dynamic_les_calculation_p4!(s)
end

@par function dynamic_les_calculation_p1!(s::A) where {A<:@par(AbstractSimulation)}
    real!(s.lesmodel.û)
    real!(s.lesmodel.M)
    @mthreads for j in TRANGE
        dynamic_les_calculation_p1!(s,j)
    end
    return nothing
end

@par function dynamic_les_calculation_p1!(s::A,j::Integer) where {A<:@par(AbstractSimulation)} 
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

@par function dynamic_les_calculation_p2!(s::A) where {A<:@par(AbstractSimulation)}
    @mthreads for j in TRANGE
        dynamic_les_calculation_p2!(s,j)
    end
    return nothing
end

@par function dynamic_les_calculation_p2!(s::A,j::Integer) where {A<:@par(AbstractSimulation)} 
    ca = s.lesmodel.c.rr
    û = s.lesmodel.û.rr
    La = s.lesmodel.L.rr
    Ma = s.lesmodel.M.rr
    Sa = s.lesmodel.S.rr
    Δ̂² = s.lesmodel.Δ̂²
    Δ² = s.lesmodel.Δ²
    aux = s.lesmodel.S.rr.xx

    @inbounds @msimd for i in REAL_RANGES[j]
        L = symouter(û[i],û[i]) - La[i]
        M = Δ̂²*norm(Sa[i])*Sa[i] - Ma[i]
        c1 = 0.5*(traceless(L) : M)
        c2 = M:M
        ca[i] = c1
        aux[i] = c2
    end

    return nothing
end

@par function dynamic_les_calculation_p3!(s::A) where {A<:@par(AbstractSimulation)}
    myfourier!(s.lesmodel.c)
    dealias!(s.lesmodel.c)
    myfourier!(s.lesmodel.S.c.xx)
    dealias!(s.lesmodel.S.c.xx)
    @mthreads for k in ZRANGE
        dynamic_les_calculation_p3!(s,k)
    end
    real!(s.lesmodel.c)
    real!(s.lesmodel.S.c.xx)
    return nothing
end

function dynamic_les_calculation_p3!(s::A,k::Integer) where {A<:AbstractSimulation}
    c = s.lesmodel.c.c
    c2 = s.lesmodel.S.c.xx
    Δ̂² = 10*s.lesmodel.Δ̂² 
    @inbounds for j in YRANGE
        @msimd for i in XRANGE 
            G = boxfilter(Δ̂²,i,j,k)
            c[i,j,k] *= G
            c2[i,j,k] *= G
        end
    end
end

@par function dynamic_les_calculation_p4!(s::A) where {A<:@par(AbstractSimulation)}
    @mthreads for j in TRANGE
        dynamic_les_calculation_p4!(s,j)
    end
    myfourier!(s.lesmodel.tau)
    dealias!(s.lesmodel.tau)
    return nothing
end

function dynamic_les_calculation_p4!(s::A,j::Integer) where {A<:AbstractSimulation} 
    c = s.lesmodel.c.rr
    c2 = s.lesmodel.S.rr.xx
    τ = s.lesmodel.tau.rr
    Δ² = s.lesmodel.Δ²
    @inbounds @msimd for i in REAL_RANGES[j]
        S = τ[i]
        cs = max(0.0, -c[i]/c2[i])
        τ[i] = 2*cs*Δ²*norm(S)*S
        c[i] = cs
    end

    return nothing
end
