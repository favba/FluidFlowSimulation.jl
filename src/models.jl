@inline @par function advance_in_time!(s::A) where {A<:@par(AbstractSimulation)}
    calculate_rhs!(s)
    hasforcing(s) && s.forcing(s)
    time_step!(s)
    return nothing
end

@par function calculate_rhs!(s::A) where {A<:@par(AbstractSimulation)}
    fourierspacep1!(s)
    realspace!(s)
    has_variable_timestep(s) && set_dt!(s)
    fourierspacep2!(s)
    return nothing
end

@par function fourierspacep1!(s::A) where {A<:@par(AbstractSimulation)}
    @mthreads for k in Base.OneTo(Nz)
        fourierspacep1!(k,s)
    end
    return nothing
end

@inline @par function fourierspacep1!(k,s::@par(AbstractSimulation)) 
    aux = s.aux.c
    u = s.u.c
    if hasles(s)
        τ = s.lesmodel.tau.c
        if hasdensity(s) || haspassivescalar(s)
            ρ = hasdensity(s) ? s.densitystratification.ρ : s.passivescalar.ρ
            ∇ρ = s.lesmodel.scalar.gradρ.c
        end
    end
    @inbounds for j in Base.OneTo(Ny)
        @msimd for i in Base.OneTo(Nx) 
            v = u[i,j,k]
            ∇ = im*K[i,j,k]  
            aux[i,j,k] = ∇ × v
            if hasles(s)
                τ[i,j,k] = symouter(∇,v)
                if haspassivescalar(s) || hasdensity(s)
                    ∇ρ[i,j,k] = ∇*ρ[i,j,k]
                end
            end
        end
    end
end

@par function realspace!(s::A) where {A<:@par(AbstractSimulation)}
    brfft!(s.u)
    brfft!(s.aux)
  
    haspassivescalar(A) && brfft!(s.passivescalar.ρ)
    hasdensity(A) && brfft!(s.densitystratification.ρ)
 
    if hasles(s)
        brfft!(s.lesmodel.tau)
        (haspassivescalar(A) | hasdensity(A)) && brfft!(s.lesmodel.scalar.gradρ)
    end
  
    realspacecalculation!(s)

    rfft_and_scale!(s.rhs)
    # fix for erros on u×ω calculation
    s.rhs[1] = zero(Vec{ComplexF64})

    dealias!(s.rhs)
    rfft_and_scale!(s.u)
    if haspassivescalar(A) 
        rfft_and_scale!(s.aux)
        dealias!(s.aux)
        rfft_and_scale!(s.passivescalar.ρ)
    elseif hasdensity(A)
        rfft_and_scale!(s.aux)
        dealias!(s.aux)
        rfft_and_scale!(s.densitystratification.ρ)
    else
        dealias!(s.aux)
    end

    if hasles(s)
        rfft_and_scale!(s.lesmodel.tau)
        dealias!(s.lesmodel.tau)
        if (haspassivescalar(A) | hasdensity(A))
            dealias!(s.lesmodel.scalar.gradρ)
        end
    end

    return nothing
end

@par function realspacecalculation!(s::A) where {A<:@par(AbstractSimulation)}
    @assert !(hasdensity(A) & haspassivescalar(A))
    @mthreads for j in 1:Nt
        realspacecalculation!(s,j)
    end
    return nothing
end

@par function realspacecalculation!(s::A,j::Integer) where {A<:@par(AbstractSimulation)} 
    u = s.u.rr
    ω = s.aux.rr
    out = s.rhs.rr
    if has_variable_timestep(A)
        umaxhere = 0.0
        umax = 0.0
        if hasdensity(A)
            ρmax = 0.0
        end
        if hasles(A)
            vis = nu(s)
            numax = 0.
            # ep = eps()
        end
    end
    if haspassivescalar(A)
        ρ = s.passivescalar.ρ.field.data
    end
    if hasdensity(A)
        ρ = s.densitystratification.ρ.field.data
    end
    if hasles(A)
        τ = s.lesmodel.tau.rr
        c = cs(s.lesmodel)
        Δ = Delta(s.lesmodel)
        α = c*c*Δ*Δ 
        is_SandP(A) && (β = cbeta(s.lesmodel)*Δ*Δ)
        if haspassivescalar(A) | hasdensity(A)
            ∇ρ = s.lesmodel.scalar.gradρ.rr
        end
    end

    @inbounds @msimd for i in RealRanges[j]
        v = u[i]
        w = ω[i]
        out[i] = v × w
        if has_variable_timestep(A)
            umaxhere = ifelse(abs(v.x)>abs(v.y),abs(v.x),abs(v.y))
            umaxhere = ifelse(umaxhere>abs(v.z),umaxhere,abs(v.z))
            umax = ifelse(umaxhere>umax,umaxhere,umax)
        end
        if hasles(A)
            S = τ[i]
            νt = α*norm(S)
            if has_variable_timestep(A)
                nnu = (vis+νt)
                numax = ifelse(numax > nnu, numax, nnu)
            end
            if is_Smagorinsky(A)
                τ[i] = νt*S
            elseif is_SandP(A)
                P = Lie(S,0.5*w)
                τ[i] = νt*S + β*P
            end
        end
        if haspassivescalar(A) || hasdensity(A)
            ω[i] = (-ρ[i]) * u[i]
            if has_variable_timestep(A)
                ρmax = ifelse(ρmax > abs(ρ[i]),ρmax,abs(ρ[i]))
            end
            if hasles(A)
                ω[i] += νt*∇ρ[i]
            end
        end
    end
    if has_variable_timestep(A) 
        s.reduction[j] = umax
        hasdensity(A) && (s.densitystratification.reduction[j] = ρmax)
        if hasles(A)
            s.lesmodel.reduction[j] = numax
        end
    end
    return nothing
end

@par function fourierspacep2!(s::A) where {A<:@par(AbstractSimulation)}
    is_implicit(typeof(s.timestep)) || add_viscosity!(s.rhs,s.u,ν,s)
    hasles(s) && add_residual_tensor!(s.rhs,s.lesmodel.tau,s)
    if hasdensity(A)
        Gdirec = graddir(s.densitystratification)
        gdir = Gdirec === :x ? s.rhs.cx : Gdirec === :y ? s.rhs.cy : s.rhs.cz 
        addgravity!(gdir, complex(s.densitystratification.ρ), -gravity(s.densitystratification), s)
    end
    pressure_projection!(s.rhs.cx,s.rhs.cy,s.rhs.cz,s)
    if haspassivescalar(A)
        gdir = graddir(s.passivescalar) === :x ? s.u.cx : graddir(s.passivescalar) === :y ? s.u.cy : s.u.cz 
        div!(complex(s.passivescalar.ρrhs), s.aux.cx, s.aux.cy, s.aux.cz, gdir, -meangradient(s.passivescalar), s)
        is_implicit(typeof(s.passivescalar.timestep)) || add_scalar_difusion!(complex(s.passivescalar.ρrhs),complex(s.passivescalar.ρ),diffusivity(s.passivescalar),s)
    end
    if hasdensity(A)
        gdir = graddir(s.densitystratification) === :x ? s.u.cx : graddir(s.densitystratification) === :y ? s.u.cy : s.u.cz 
        div!(complex(s.densitystratification.ρrhs), s.aux.cx, s.aux.cy, s.aux.cz, gdir, -meangradient(s.densitystratification), s)
        is_implicit(typeof(s.densitystratification.timestep)) || add_scalar_difusion!(complex(s.densitystratification.ρrhs),complex(s.densitystratification.ρ),diffusivity(s.densitystratification),s)
    end
    return nothing
end

@par function add_residual_tensor!(rhs::VectorField,τ::SymmetricTracelessTensor,s::@par(AbstractSimulation))
    @mthreads for k in 1:Nz
        add_residual_tensor!(rhs,τ,k,s)
    end
end

@inline @par function add_residual_tensor!(rhs::VectorField,tau::SymmetricTracelessTensor,k::Int,s::@par(AbstractSimulation))
    rx = rhs.cx
    ry = rhs.cy
    rz = rhs.cz
    txx = tau.cxx
    txy = tau.cxy
    txz = tau.cxz
    tyy = tau.cyy
    tyz = tau.cyz
    @inbounds for j in 1:Ny
        @msimd for i in 1:Nx
            rx[i,j,k] += im*(kx[i]*txx[i,j,k] + ky[j]*txy[i,j,k] + kz[k]*txz[i,j,k])
            ry[i,j,k] += im*(kx[i]*txy[i,j,k] + ky[j]*tyy[i,j,k] + kz[k]*tyz[i,j,k])
            rz[i,j,k] += im*(kx[i]*txz[i,j,k] + ky[j]*tyz[i,j,k] + kz[k]*(-txx[i,j,k]-tyy[i,j,k]))
        end
    end
end


function add_viscosity!(rhs::VectorField,u::VectorField,ν::Real,s::AbstractSimulation)
    if hashyperviscosity(s)
         _add_hviscosity!(rhs.cx,u.cx,s)
         _add_hviscosity!(rhs.cy,u.cy,s)
         _add_hviscosity!(rhs.cz,u.cz,s)
    else
         _add_viscosity!(rhs.cx,u.cx,-ν,s)
         _add_viscosity!(rhs.cy,u.cy,-ν,s)
         _add_viscosity!(rhs.cz,u.cz,-ν,s)
    end
end

@par function _add_viscosity!(rhs::AbstractArray,u::AbstractArray,mν::Real,s::@par(AbstractSimulation))
    @mthreads for k in 1:Nz
        for j in 1:Ny
            @inbounds @msimd for i in 1:Nx
                #rhs[i,j,k] = (kx[i]*kx[i] + ky[j]*ky[j] + kz[k]*kz[k])*mŒΩ*u[i,j,k] + rhs[i,j,k]
                rhs[i,j,k] = muladd(muladd(kx[i], kx[i], muladd(ky[j], ky[j], kz[k]*kz[k])), mν*u[i,j,k], rhs[i,j,k])
            end
        end
    end
end

@par function _add_hviscosity!(rhs::AbstractArray,u::AbstractArray,s::@par(AbstractSimulation))
    mν = -nu(s)
    mνh = -nuh(s)
    M = get_hyperviscosity_exponent(s)
    @mthreads for k in 1:Nz
        for j in 1:Ny
            @inbounds @msimd for i in 1:Nx
                modk2 = muladd(kx[i], kx[i], muladd(ky[j], ky[j], kz[k]*kz[k]))
                rhs[i,j,k] = muladd(muladd(modk2, mν, modk2^M * mνh), u[i,j,k], rhs[i,j,k])
            end
        end
    end
end

function add_scalar_difusion!(rhs::AbstractArray,u::AbstractArray,ν::Real,s::AbstractSimulation)
    _add_viscosity!(rhs,u,-ν,s)
end

 @par function pressure_projection!(rhsx,rhsy,rhsz,s::@par(AbstractSimulation))
    @inbounds a = (rhsx[1],rhsy[1],rhsz[1])
    @mthreads for k in 1:Nz
        for j in 1:Ny
            @inbounds @msimd for i in 1:Nx
                #p1 = -(kx[i]*rhsx[i,j,k] + ky[j]*rhsy[i,j,k] + kz[k]*rhsz[i,j,k])/(kx[i]*kx[i] + ky[j]*ky[j] + kz[k]*kz[k])
                p1 = -muladd(kx[i], rhsx[i,j,k], muladd(ky[j], rhsy[i,j,k], kz[k]*rhsz[i,j,k]))/muladd(kx[i], kx[i], muladd(ky[j], ky[j],  kz[k]*kz[k]))
                rhsx[i,j,k] = muladd(kx[i],p1,rhsx[i,j,k])
                rhsy[i,j,k] = muladd(ky[j],p1,rhsy[i,j,k])
                rhsz[i,j,k] = muladd(kz[k],p1,rhsz[i,j,k])
            end
        end
    end
    @inbounds rhsx[1],rhsy[1],rhsz[1] = a
end

@par function addgravity!(rhs,ρ,g::Real,s::@par(AbstractSimulation))
    @mthreads for k in 1:Nz
        for j in 1:Ny
            @inbounds @msimd for i in 1:Nx
                rhs[i,j,k] = muladd(ρ[i,j,k],g,rhs[i,j,k])
            end
        end
    end
end

@par function time_step!(s::A) where {A<:@par(AbstractSimulation)}
    s.timestep(s.u,s.rhs,s)
    haspassivescalar(s) && s.passivescalar.timestep(s.passivescalar.ρ,s.passivescalar.ρrhs,s)
    hasdensity(s) && s.densitystratification.timestep(s.densitystratification.ρ,s.densitystratification.ρrhs,s)
end
