include("dynamic_smagorinsky.jl")

function realspace_LES_calculation!(s::AbstractSimulation)
    hasles(s) || return nothing
    if is_dynamic_les(s)
        realspace_dynamic_les_calculation!(s)
        fourierspace_dynamic_les_calculation!(s)
        coefficient_les_calculation!(s)
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

    if is_dynP_les(A)
        Pa = s.lesmodel.P.rr
        ŵ = s.lesmodel.ŵ.rr
        cpa = s.lesmodel.cp.rr
    end

    if is_SandP(A)
        β = s.lesmodel.cb*Δ²
    end

    if is_Silvis(A)
        c = s.lesmodel.c
        cp = s.lesmodel.cp
    end

    if is_stable_nl_model(A)
        c = s.lesmodel.c
    end

    @inbounds @msimd for i in REAL_RANGES[j]
        w = rhs[i]
        S = τ[i]

        if is_dynamic_les(A)
            νt = max(cmin,ca[i])*Δ²*norm(S)
            t = 2*νt*S
        elseif is_Smagorinsky(A)
            νt = α*norm(S)
            t = 2*νt*S
        elseif is_Vreman(A)
            νt = Vreman_eddy_viscosity(S,w,c,Δ²)
            t = 2*νt*S
        elseif is_production_model(A)
            νt = production_eddy_viscosity(S,AntiSymTen(-0.5w),c,Δ²)
            t = 2*νt*S
        elseif is_Silvis(A)
            fv = fvs(S,w)
            νt = Silvis_eddy_viscosity(S,fv,c,Δ²)
            νp = Silvis_P_coeff(cp,fv,Δ²)
            t = 2*νt*S + νp*Lie(S,AntiSymTen(0.5*w))
        elseif is_stable_nl_model(A)
            W = AntiSymTen(-0.5w)
            T = traceless(square(W) - Lie(S,W) - square(S))
            νt, mnut = stable_nl_eddy_viscosity(S,W,c,Δ²)
            t = c*Δ²*T + 2*mnut*S
        end

        if is_SandP(A)
            P = Lie(S,AntiSymTen(0.5*w))
            if is_FakeSmagorinsky(A)
                t = β*P
            else
                t += β*P
            end
        end

        if is_dynP_les(A)
           t += cpa[i]*Δ²*Lie(S,AntiSymTen(0.5*w))
        end

        if !(is_FakeSmagorinsky(A) && !is_SandP(A))
            τ[i] = t
        end


        if hasdensityles(A)
            is_FakeSmagorinsky(A) && (νt = 0.0)
            ∇ρ = f[i]
            rhsden = νt*∇ρ
            if has_les_density_vorticity_model(A)
                rhsden += (cρ/2) * (w × ∇ρ)
            end
            f[i] = rhsden
        end

        if haspassivescalarles(A)
            is_FakeSmagorinsky(A) && (νt = 0.0)
            ∇φ = fφ[i]
            rhsp = νt*fφ[i]
            if has_les_scalar_vorticity_model(A)
                rhsp += (cφ/2) * (w × ∇φ)
            end
            fφ[i] = rhsp
        end

    end

    return nothing
end
