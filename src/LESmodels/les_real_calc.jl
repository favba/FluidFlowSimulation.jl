include("dynamic_smagorinsky.jl")

function realspace_LES_calculation!(s::AbstractSimulation)
    hasles(s) || return nothing
    if is_dynamic_les(s)
        realspace_dynamic_les_calculation!(s)
        fourierspace_dynamic_les_calculation!(s)
        #coefficient_dynamic_les_calculation!(s)
    end
    calc_les!(s)
end

@par function calc_les!(s::A) where {A<:@par(AbstractSimulation)}
    @mthreads for j in TRANGE
        calc_les!(s,j)
    end
    return nothing
end

@par function calc_les!(s::A,j::Integer) where {A<:@par(AbstractSimulation)}

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

    if is_dynamic_les(A)
        ca = s.lesmodel.c.rr
        û = s.lesmodel.û.rr
        La = s.lesmodel.L.rr
        Ma = s.lesmodel.M.rr
        Sa = s.lesmodel.S.rr
        Δ̂² = s.lesmodel.Δ̂²
        cmin = s.lesmodel.cmin
    elseif is_Smagorinsky(A) || is_Vreman(A) || is_production_model(A)
        c = s.lesmodel.c
        α = c*c*Δ²
    end

    if is_SandP(A)
        β = s.lesmodel.cb*Δ²
    end

    pr = s.lesmodel.pr.rr

    @inbounds @msimd for i in REAL_RANGES[j]
        w = rhs[i]
        S = τ[i]

        if is_dynamic_les(A)
            L = symouter(û[i],û[i]) - La[i]
            M = Δ̂²*norm(Ma[i])*Ma[i] - Sa[i]
            c = 0.5*((traceless(L) : M)/(M:M))
            c = max(cmin ,c) # to prevent underflow set minimum value for c
            νt = 2*c*Δ²*norm(S)
            t = νt*S
            ca[i] = c
        elseif is_Smagorinsky(A)
            νt = α*norm(S)
            t = 2*νt*S
        elseif is_Vreman(A)
            νt = Vreman_eddy_viscosity(S,w,c,Δ²)
            t = 2*νt*S
        elseif is_production_model(A)
            νt = production_eddy_viscosity(S,AntiSymTen(-0.5w),c,Δ²)
            t = 2*νt*S
        end

        if is_SandP(A)
            P = Lie(S,AntiSymTen(-0.5*w))
            t += β*P
        end

        pr[i] = t:S
        τ[i] = t

        if hasdensityles(A)
            ∇ρ = f[i]
            rhsden = νt*∇ρ
            if has_les_density_vorticity_model(A)
                rhsden -= cρ*w × ∇ρ
            end
            f[i] = rhsden
        end

        if haspassivescalarles(A)
            ∇φ = fφ[i]
            rhsp = νt*fφ[i]
            if has_les_scalar_vorticity_model(A)
                rhsp -= cφ*w × ∇φ
            end
            fφ[i] = rhsp
        end

    end

    return nothing
end
