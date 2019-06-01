include("dynamic_piomelli.jl")

@par function realspace_dynamic_les_calculation!(s::A) where {A<:@par(AbstractSimulation)}
    real!(s.lesmodel.û)
    real!(s.lesmodel.M)
    if is_dynP_les(A)
        real!(s.lesmodel.ŵ)
    end
    if is_piomelliSmag_les(A)
        dt = get_dt(s)
        dtm1 = s.lesmodel.dtm1[]
        dtf = dt == 0.0 ? 1.0 : dt
        dtm1f = dtm1 == 0.0 ? dtf : dtm1

        if s.iteration[] == 0
            copyto!(s.lesmodel.cm1.rr,s.lesmodel.c.rr)
        end

        calc_cstar!(s.lesmodel.c.rr,s.lesmodel.cm1.rr,dtf,dtm1f)
        s.lesmodel.dtm1[] = dt
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

    if is_piomelliSmag_les(A)
        c = s.lesmodel.c.rr
    end

    if is_dynP_les(A)
        w = s.rhs.rr
        P = s.lesmodel.P.rr
    end

    @inbounds @msimd for i in REAL_RANGES[j]
        S = tau[i]
        if is_piomelliSmag_les(A)
            Sa[i] = 2*c[i]*Δ²*norm(S)*S 
        else
            Sa[i] = Δ²*norm(S)*S 
        end
        v = u[i]
        L[i] = symouter(v,v)
        if is_dynP_les(A)
            P[i] = Δ²*Lie(S,AntiSymTen(0.5*w[i]))
        end
    end

    return nothing
end

@par function fourierspace_dynamic_les_calculation!(s::A) where {A<:@par(AbstractSimulation)}
    fullfourier!(s.lesmodel.S)
    fullfourier!(s.lesmodel.L)
    is_dynP_les(A) && fullfourier!(s.lesmodel.P)

    @mthreads for k in Base.OneTo(NZ)
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

    @inbounds for j in Base.OneTo(NY)
        @msimd for i in Base.OneTo(NX) 
            G = Gaussfilter(Δ̂²,i,j,k)
            S[i,j,k] *= G
            L[i,j,k] *= G
            if is_dynP_les(A)
                P[i,j,k] *= G
            end
        end
    end
end
