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

    if is_piomelliP_les(A)
        dt = get_dt(s)
        dtm1 = s.lesmodel.dtpm1[]
        dtf = dt == 0.0 ? 1.0 : dt
        dtm1f = dtm1 == 0.0 ? dtf : dtm1

        if s.iteration[] == 0
            copyto!(s.lesmodel.cpm1.rr,s.lesmodel.cp.rr)
        end

        calc_cstar!(s.lesmodel.cp.rr,s.lesmodel.cpm1.rr,dtf,dtm1f)
        s.lesmodel.dtpm1[] = dt
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
        if is_piomelliP_les(A)
            cp = s.lesmodel.cp.rr
        end
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
        if is_piomelliP_les(A)
            P[i] = cp[i]*Δ²*Lie(S,AntiSymTen(0.5*w[i]))
        elseif is_dynSandP_les(A)
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

@par function coefficient_les_calculation!(s::A) where {A<:@par(AbstractSimulation)}
    @mthreads for j in TRANGE
        coefficient_les_calculation!(s,j)
    end

    s.lesmodel.avg && average_c!(s.lesmodel.c,s.lesmodel.Δ̂²)

    if is_dynP_les(s)
        s.lesmodel.avg && average_c!(s.lesmodel.cp,s.lesmodel.Δ̂²)
    end
 
    return nothing
end

@par function coefficient_les_calculation!(s::A,j::Integer) where {A<:@par(AbstractSimulation)}

    rhs = s.rhs.rr
    τ = s.lesmodel.tau.rr
    Δ² = s.lesmodel.Δ²

    if haspassivescalarles(A)
        φ = s.passivescalar.φ.field.data
        fφ = s.passivescalar.flux.rr
        has_les_scalar_vorticity_model(A) && (cφ = s.passivescalar.lesmodel.c*Δ²)
    end

    if hasdensityles(A)
        ρ = s.densitystratification.ρ.field.data
        f = s.densitystratification.flux.rr
        has_les_density_vorticity_model(A) && (cρ = s.densitystratification.lesmodel.c*Δ²)
    end

    ca = s.lesmodel.c.rr
    û = s.lesmodel.û.rr
    La = s.lesmodel.L.rr
    Ma = s.lesmodel.M.rr
    Sa = s.lesmodel.S.rr
    Δ̂² = s.lesmodel.Δ̂² + Δ²
    #cmin = s.lesmodel.cmin

    if is_dynP_les(A)
        Pa = s.lesmodel.P.rr
        ŵ = s.lesmodel.ŵ.rr
        cpa = s.lesmodel.cp.rr
    end

    @inbounds @msimd for i in REAL_RANGES[j]
        w = rhs[i]
        S = τ[i]

        L = traceless(symouter(û[i],û[i]) - La[i])
        La[i] = L
        Sh = Ma[i]

        if is_dynSmag_les(A)
            M = Δ̂²*norm(Sh)*Sh - Sa[i]
            #c = max(0.0,0.5*((L:M)/(M:M)))
            c = max(-0.4,min(0.4,0.5*((L:M)/(M:M))))
        elseif is_piomelliSmag_les(A)

            τ̂ = Sa[i]

            if is_piomelliP_les(A)
                τ̂ += Pa[i]
            end

            ah = norm(Sh)

            M = 2*Δ̂²*ah*Sh
            Mn = M/(M:M)
            c = (τ̂ + L):Mn
            c = max(-0.4,min(0.4,c))
            #c = ifelse(ah<10.0,0.0,max(0.0,c))
        end
        ca[i] = c

        if is_dynP_les(A)
            wh = ŵ[i]
            Ph = Lie(Sh,AntiSymTen(0.5*wh))
           
            if is_dynSandP_les(A)
                Mp = Δ̂²*Ph - Pa[i]
                cp = max(-0.8,min(0.8,(L:Mp)/(Mp:Mp))) # Should I clip negative values?
                #cp = max(0.0,cp)
            elseif is_piomelliP_les(A)
                if !is_piomelliSmag_les(A)
                    τ̂ = Pa[i]
                end
                Mp = Δ̂²*Ph
                Mpn = Mp/(Mp:Mp)
                cp = max(-0.8,min(.8,(τ̂ + L):Mpn))
            end
 
            cpa[i] = cp
        end

    end

    return nothing
end
