include("LESmodels/les_real_calc.jl")

@par function advance_in_time!(s::A) where {A<:@par(AbstractSimulation)}
    calculate_rhs!(s)
    hasforcing(A) && s.forcing(s)

    mod(s.iteration[],s.dtstats) == 0 && writestats(s)
    time_step!(s)

    s.iteration[] += 1
    s.time[] += get_dt(s)
    return nothing
end

@par function calculate_rhs!(s::A) where {A<:@par(AbstractSimulation)}
    # Set necessary fields to fourier
    setfourier!(s.rhs)
    hasdensity(A) && setfourier!(s.densitystratification.rhs)
    hasdensityles(A) && setfourier!(s.densitystratification.flux)
    haspassivescalar(A) && setfourier!(s.passivescalar.rhs)
    haspassivescalarles(A) && setfourier!(s.passivescalar.flux)
    hasles(A) && setfourier!(s.lesmodel.tau)
    is_dynamic_les(A) && (setfourier!(s.lesmodel.û); setfourier!(s.lesmodel.M))


    #actual calculation
    fourierspacep1!(s)
    realspace!(s)
    has_variable_timestep(s) && set_dt!(s)
    fourierspacep2!(s)

    return nothing
end

@par function fourierspacep1!(s::A) where {A<:@par(AbstractSimulation)}
    @mthreads for k in ZRANGE
        fourierspacep1!(k,s)
    end
    return nothing
end

@par function fourierspacep1!(k,s::A) where {A<:@par(AbstractSimulation)}
    rhs = s.rhs.c
    u = s.u.c
    if hasles(A)
        τ = s.lesmodel.tau.c
        if is_dynamic_les(A)
            û = s.lesmodel.û.c
            Δ̂² = s.lesmodel.Δ̂²
            M = s.lesmodel.M.c
        end
    end

    if hasdensity(A) && hasdensityles(A)
        ρ = s.densitystratification.ρ
        f = s.densitystratification.flux
    end

    if haspassivescalar(A) && haspassivescalarles(A)
        φ = s.passivescalar.φ
        fφ = s.passivescalar.flux
    end

    @inbounds for j in YRANGE
        @msimd for i in XRANGE 
            v = u[i,j,k]
            ∇ = im*K[i,j,k]  

            if need_vorticity(A)
                rhs[i,j,k] = ∇ × v
            end

            if hasles(A)
                S = symouter(∇,v)
                τ[i,j,k] = S
                if is_dynamic_les(A)
                    G = Gaussfilter(Δ̂²,i,j,k)
                    M[i,j,k] = G*S
                    û[i,j,k] = G*v
                end
            end

            if hasdensity(A) && hasdensityles(A)
                f[i,j,k] = ∇*ρ[i,j,k]
            end

            if haspassivescalar(A) && haspassivescalarles(A)
                fφ[i,j,k] = ∇*φ[i,j,k]
            end

        end
    end
end

@par function realspace!(s::A) where {A<:@par(AbstractSimulation)}
    real!(s.u)
    if need_vorticity(A)
        real!(s.rhs)
    end
    if !is_vorticityEquation(A)
        setreal!(s.equation.uu)
    end
  
    haspassivescalar(A) && real!(s.passivescalar.φ)
    hasdensity(A) && real!(s.densitystratification.ρ)
 
    hasles(s) && real!(s.lesmodel.tau)
  
    if hasdensity(A)
        if hasdensityles(A)
            real!(s.densitystratification.flux)
        else
            setreal!(s.densitystratification.flux)
        end
    end

    if haspassivescalar(A)
        if haspassivescalarles(A)
            real!(s.passivescalar.flux)
        else
            setreal!(s.passivescalar.flux)
        end
    end

    realspace_LES_calculation!(s)
    realspacecalculation!(s)

    if has_variable_timestep(s)
        find_max(s.reduction,s.u.r)
        if hasdensity(s)
            find_max(s.densitystratification.reduction,s.densitystratification.ρ.r)
        end
    end

    is_output_time(s) && writeoutput(s)

    if is_vorticityEquation(A)
        myfourier!(s.rhs)
        # fix for erros on u×ω calculation
        s.rhs.c[1] = zero(Vec{ComplexF64})
        dealias!(s.rhs)
    else
        myfourier!(s.equation.uu)
        dealias!(s.equation.uu)
        setfourier!(s.rhs)
        div!(s.rhs,s.equation.uu)
    end

    if is_output_time(s)
        real!(s.rhs)
        init = s.iteration[]
        write("nn1.$init",s.rhs.rr.x)
        write("nn2.$init",s.rhs.rr.y)
        write("nn3.$init",s.rhs.rr.z)
        myfourier!(s.rhs)
    end
 
    myfourier!(s.u)

    if haspassivescalar(A) 
        myfourier!(s.passivescalar.flux)
        dealias!(s.passivescalar.flux)
        myfourier!(s.passivescalar.φ)
    end
    if hasdensity(A)
        myfourier!(s.densitystratification.flux)
        dealias!(s.densitystratification.flux)
        myfourier!(s.densitystratification.ρ)
    end

    if hasles(s)
        if !(!is_SandP(A) && is_FakeSmagorinsky(A))
            myfourier!(s.lesmodel.tau)
            dealias!(s.lesmodel.tau)
        end
    end

    return nothing
end


@par function realspacecalculation!(s::A) where {A<:@par(AbstractSimulation)}
    #@assert !(hasdensity(A) & haspassivescalar(A))
    @mthreads for j in TRANGE
        realspacecalculation!(s,j)
    end
    return nothing
end

@par function realspacecalculation!(s::A,j::Integer) where {A<:@par(AbstractSimulation)} 

    u = s.u.rr
    rhs = s.rhs.rr

    if !is_vorticityEquation(A)
        uu = s.equation.uu.rr
    end

    if haspassivescalar(A)
        φ = s.passivescalar.φ.field.data
        fφ = s.passivescalar.flux.rr
    end

    if hasdensity(A)
        ρ = s.densitystratification.ρ.field.data
        f = s.densitystratification.flux.rr
    end

    @inbounds @msimd for i in REAL_RANGES[j]

        v = u[i]
        if is_vorticityEquation(A)
            w = rhs[i]
            rhs[i] = v × w
        else
            uu[i] = traceless(symouter(v,-v))
        end

        if hasdensity(A)
            rhsden = (-ρ[i]) * v
            hasdensityles(A) && (rhsden += f[i])
            f[i] = rhsden
        end

        if haspassivescalar(A)
            rhsp = (-φ[i]) * v
            haspassivescalarles(A) && (rhsp += fφ[i])
            fφ[i] = rhsp
        end

    end

    return nothing
end

@par function fourierspacep2_velocity!(s::A) where {A<:@par(AbstractSimulation)}
    @mthreads for k in ZRANGE
        fourierspacep2_velocity!(k,s)
    end
    return nothing
end

@inline @par function fourierspacep2_velocity!(k,s::A) where {A<:@par(AbstractSimulation)}
    u = s.u.c
    rhsv = s.rhs.c

    if !is_implicit(A)
        mν = -ν
        if hashyperviscosity(A)
            mνh = -nuh(A)
            M = get_hyperviscosity_exponent(A)
        end
    end

    if hasles(A)
        τ = s.lesmodel.tau.c
    end

    if hasdensity(A)
        ρ = s.densitystratification.ρ.field
        g = gravity(s.densitystratification)
    end
    
    @inbounds for j in YRANGE
        @msimd for i in XRANGE 

            kh = K[i,j,k]
            K2 = kh⋅kh
            v = u[i,j,k]
            rhs = rhsv[i,j,k]

            if !is_implicit(A)
                if hashyperviscosity(A)
                    rhs += (K2*mν + (K2^M)*mνh)*v
                else
                    rhs += (K2*mν)*v
                end
            end

            if hasles(A)
                if !(!is_SandP(A) && is_FakeSmagorinsky(A))
                    rhs += (im*kh)⋅τ[i,j,k]
                end
            end

            if hasdensity(A)
                rhs += ρ[i,j,k]*g
            end
            
            p1 = ifelse(k==j==i==1,0.0,-(kh⋅rhs)/K2)
            rhsv[i,j,k] = p1*kh + rhs
        end
    end

end

@par function fourierspacep2!(s::A) where {A<:@par(AbstractSimulation)}
    #is_implicit(typeof(s.timestep)) || add_viscosity!(s.rhs,s.u,ν,s)
    #hasles(s) && add_residual_tensor!(s.rhs,s.lesmodel.tau,s)
    #if hasdensity(A)
        #Gdirec = graddir(s.densitystratification)
        #gdir = Gdirec === :x ? s.rhs.cx : Gdirec === :y ? s.rhs.cy : s.rhs.cz 
        #addgravity!(gdir, complex(s.densitystratification.ρ), -gravity(s.densitystratification), s)
    #end
    #pressure_projection!(s.rhs.c.x,s.rhs.c.y,s.rhs.c.z,s)
    fourierspacep2_velocity!(s)
    if haspassivescalar(A)
        gdir = graddir(s.passivescalar) === :x ? s.u.c.x : graddir(s.passivescalar) === :y ? s.u.c.y : s.u.c.z 
        div!(complex(s.passivescalar.rhs), s.passivescalar.flux.c.x, s.passivescalar.flux.c.y, s.passivescalar.flux.c.z, gdir, -meangradient(s.passivescalar), s)
        is_implicit(typeof(s.passivescalar.timestep)) || add_scalar_difusion!(complex(s.passivescalar.rhs),complex(s.passivescalar.φ),diffusivity(s.passivescalar),s)
    end
    if hasdensity(A)
        gdir = graddir(s.densitystratification) === :x ? s.u.c.x : graddir(s.densitystratification) === :y ? s.u.c.y : s.u.c.z 
        div!(complex(s.densitystratification.rhs), s.densitystratification.flux.c.x, s.densitystratification.flux.c.y, s.densitystratification.flux.c.z, gdir, -meangradient(s.densitystratification), s)
        is_implicit(typeof(s.densitystratification.timestep)) || add_scalar_difusion!(complex(s.densitystratification.rhs),complex(s.densitystratification.ρ),diffusivity(s.densitystratification),s)
    end
    return nothing
end

function div!(out,input)
    @mthreads for k in ZRANGE
        @inbounds for j in YRANGE
            @msimd for i in XRANGE
                out[i,j,k] = (im*K[i,j,k]) ⋅ input[i,j,k]
            end
        end
    end
end

#@par function add_residual_tensor!(rhs::VectorField,τ::SymmetricTracelessTensor,s::@par(AbstractSimulation))
#    @mthreads for k in 1:Nz
#        add_residual_tensor!(rhs,τ,k,s)
#    end
#end
#
#@inline @par function add_residual_tensor!(rhs::VectorField,tau::SymmetricTracelessTensor,k::Int,s::@par(AbstractSimulation))
#    rx = rhs.cx
#    ry = rhs.cy
#    rz = rhs.cz
#    txx = tau.cxx
#    txy = tau.cxy
#    txz = tau.cxz
#    tyy = tau.cyy
#    tyz = tau.cyz
#    @inbounds for j in 1:Ny
#        @msimd for i in 1:Nx
#            rx[i,j,k] += im*(kx[i]*txx[i,j,k] + ky[j]*txy[i,j,k] + kz[k]*txz[i,j,k])
#            ry[i,j,k] += im*(kx[i]*txy[i,j,k] + ky[j]*tyy[i,j,k] + kz[k]*tyz[i,j,k])
#            rz[i,j,k] += im*(kx[i]*txz[i,j,k] + ky[j]*tyz[i,j,k] + kz[k]*(-txx[i,j,k]-tyy[i,j,k]))
#        end
#    end
#end


#function add_viscosity!(rhs::VectorField,u::VectorField,ν::Real,s::AbstractSimulation)
#    if hashyperviscosity(s)
#        _add_hviscosity!(rhs.c.x,u.c.x,s)
#        _add_hviscosity!(rhs.c.y,u.c.y,s)
#        _add_hviscosity!(rhs.c.z,u.c.z,s)
#    else
#        _add_viscosity!(rhs.c.x,u.c.x,-ν,s)
#        _add_viscosity!(rhs.c.y,u.c.y,-ν,s)
#        _add_viscosity!(rhs.c.z,u.c.z,-ν,s)
#    end
#end
#
#@par function _add_viscosity!(rhs::AbstractArray,u::AbstractArray,mν::Real,s::@par(AbstractSimulation))
#    @mthreads for k in ZRANGE
#        for j in YRANGE
#            @inbounds @msimd for i in XRANGE
#                #rhs[i,j,k] = (kx[i]*kx[i] + ky[j]*ky[j] + kz[k]*kz[k])*mŒΩ*u[i,j,k] + rhs[i,j,k]
#                rhs[i,j,k] = muladd(muladd(KX[i], KX[i], muladd(KY[j], KY[j], KZ[k]*KZ[k])), mν*u[i,j,k], rhs[i,j,k])
#            end
#        end
#    end
#end
#
#@par function _add_hviscosity!(rhs::AbstractArray,u::AbstractArray,s::@par(AbstractSimulation))
#    mν = -nu(s)
#    mνh = -nuh(s)
#    M = get_hyperviscosity_exponent(s)
#    @mthreads for k in ZRANGE
#        for j in YRANGE
#            @inbounds @msimd for i in XRANGE
#                modk2 = muladd(KX[i], KX[i], muladd(KY[j], KY[j], KZ[k]*KZ[k]))
#                rhs[i,j,k] = muladd(muladd(modk2, mν, modk2^M * mνh), u[i,j,k], rhs[i,j,k])
#            end
#        end
#    end
#end
#
#function add_scalar_difusion!(rhs::AbstractArray,u::AbstractArray,ν::Real,s::AbstractSimulation)
#    _add_viscosity!(rhs,u,-ν,s)
#end

@par function pressure_projection!(rhs,s::@par(AbstractSimulation))
    @mthreads for k in ZRANGE
        for j in YRANGE
            @inbounds @msimd for i in XRANGE
                rhsh = rhs[i,j,k]
                kh = K[i,j,k]
                K2 = kh⋅kh
                p1 = ifelse(k==j==i==1,0.0,-(kh⋅rhsh)/K2)
                rhs[i,j,k] = p1*kh + rhsh
            end
        end
    end
end

#@par function addgravity!(rhs,ρ,g::Real,s::@par(AbstractSimulation))
    #@mthreads for k in ZRANGE
        #for j in YRANGE
            #@inbounds @msimd for i in XRANGE
                #rhs[i,j,k] = muladd(ρ[i,j,k],g,rhs[i,j,k])
            #end
        #end
    #end
#end

@par function time_step!(s::A) where {A<:@par(AbstractSimulation)}
    s.timestep(s.u,s.rhs,s)
    haspassivescalar(s) && s.passivescalar.timestep(s.passivescalar.φ,s.passivescalar.rhs,s)
    hasdensity(s) && s.densitystratification.timestep(s.densitystratification.ρ,s.densitystratification.rhs,s)
end
