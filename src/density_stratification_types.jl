abstract type AbstractDensityStratification{L,TT,α,dρdz,g,Gdirec} end

    hasdensityles(a::Type{T}) where {L,TT,α,dρdz,g,Gdirec,T<:AbstractDensityStratification{L,TT,α,dρdz,g,Gdirec}} = 
        L === NoLESScalar ? false : true
    @inline hasdensityles(a::AbstractDensityStratification{L,TT,α,dρdz,g,Gdirec}) where {L,TT,α,dρdz,g,Gdirec} = hasdensityles(typeof(a))

    has_les_density_vorticity_model(a::Type{T}) where {L,TT,α,dρdz,g,Gdirec,T<:AbstractDensityStratification{L,TT,α,dρdz,g,Gdirec}} = 
        is_vorticity_model(L)
    @inline has_les_density_vorticity_model(a::AbstractDensityStratification{L,TT,α,dρdz,g,Gdirec}) where {L,TT,α,dρdz,g,Gdirec} = has_les_density_vorticity_model(typeof(a))
    @inline @par has_les_density_vorticity_model(s::Type{T}) where {T<:@par(AbstractSimulation)} = has_les_density_vorticity_model(DensityStratificationType)

    diffusivity(a::Type{T}) where {L,TT,α,dρdz,g,Gdirec,T<:AbstractDensityStratification{L,TT,α,dρdz,g,Gdirec}} = 
        α
    @inline diffusivity(a::AbstractDensityStratification{L,TT,α,dρdz,g,Gdirec}) where {L,TT,α,dρdz,g,Gdirec} = diffusivity(typeof(a))
    @inline @par diffusivity(s::Type{T}) where {T<:@par(AbstractSimulation)} = diffusivity(DensityStratificationType)
    @inline diffusivity(s::AbstractSimulation) = diffusivity(typeof(s))

    meangradient(a::Type{T}) where {L,TT,α,dρdz,g,Gdirec,T<:AbstractDensityStratification{L,TT,α,dρdz,g,Gdirec}} = 
        dρdz
    @inline meangradient(a::AbstractDensityStratification{L,TT,α,dρdz,g,Gdirec}) where {L,TT,α,dρdz,g,Gdirec} = meangradient(typeof(a))

    gravity(a::Type{T}) where {L,TT,α,dρdz,g,Gdirec,T<:AbstractDensityStratification{L,TT,α,dρdz,g,Gdirec}} = 
        g
    @inline gravity(a::AbstractDensityStratification{L,TT,α,dρdz,g,Gdirec}) where {L,TT,α,dρdz,g,Gdirec} = gravity(typeof(a))
    @inline @par gravity(s::Type{T}) where {T<:@par(AbstractSimulation)} = gravity(DensityStratificationType)
    @inline gravity(s::AbstractSimulation) = gravity(typeof(s))

    graddir(a::Type{T}) where {L,TT,α,dρdz,g,Gdirec,T<:AbstractDensityStratification{L,TT,α,dρdz,g,Gdirec}} = 
        Gdirec
    @inline graddir(a::AbstractDensityStratification{L,TT,α,dρdz,g,Gdirec}) where {L,TT,α,dρdz,g,Gdirec} = graddir(typeof(a))


    initialize!(a::AbstractDensityStratification,s::AbstractSimulation) = initialize!(a.timestep,parent(real(a.rhs)),diffusivity(a),s)

    statsheader(a::AbstractDensityStratification) = "rho,rhop2,drhodxp2,drhodyp2,drhodzp2"#,u.g*rho"

    stats(a::AbstractDensityStratification,s::A) where {A<:AbstractSimulation} = (scalar_stats(a.ρ,a,s)...,)# buoyancy_flux(a,s))

struct NoDensityStratification <: AbstractDensityStratification{NoLESScalar,nothing,nothing,nothing,nothing,nothing} end

    initialize!(a::NoDensityStratification,s::AbstractSimulation) = nothing

    statsheader(a::NoDensityStratification) = ""

    stats(a::NoDensityStratification,s::AbstractSimulation) = ()

    msg(a::NoDensityStratification) = "\nDensity Stratification: No density stratification\n"

struct BoussinesqApproximation{L,TTimeStep, α #=Difusitivity = ν/Pr =#,
                   dρdz #=Linear mean profile=#, g #=This is actually g/ρ₀ =#, 
                   Gdirec#=Gravity direction =#} <: AbstractDensityStratification{L,TTimeStep,α,dρdz,g,Gdirec}
    ρ::ScalarField{Float64,3,2,false}
    rhs::ScalarField{Float64,3,2,false}
    flux::VectorField{Float64,3,2,false}
    lesmodel::L
    timestep::TTimeStep
    reduction::Vector{Float64}

    function BoussinesqApproximation{TT,α,dρdz,g,Gdirec}(ρ,timestep,les) where {TT,α,dρdz,g,Gdirec}
        ρrhs = similar(ρ)
        flux = VectorField(size(ρ.field.r),(ρ.kx.l,ρ.ky.l,ρ.kz.l))
        reduction = zeros(THR ? Threads.nthreads() : 1)
        return new{typeof(les),TT,α,dρdz,g,Gdirec}(ρ,ρrhs,flux,les,timestep,reduction)
    end
end 

initialize!(a::BoussinesqApproximation,s::AbstractSimulation) = initialize!(a.timestep,parent(real(a.rhs)),diffusivity(a),s)

msg(a::BoussinesqApproximation{L,TT,α,dρdz,g,Gdirec}) where {L,TT,α,dρdz,g,Gdirec} = """

Density Stratification: Boussinesq Approximation
Density diffusivity: $(α)
g/ρ₀ : $(g)
Density mean gradient : $(dρdz)
Density mean gradient direction: $(Gdirec)
Density time-stepping method: $(TT)
Density LES model: $(L)

"""

function stratificationtype(d::AbstractDict,start,integrator,nx,ny,nz,lx,ly,lz,variableTimeStep,idt)
    
    s = NoDensityStratification()

    if haskey(d,:hyperViscosity)
        νh = parse(Float64,d[:hyperViscosity])
        m = haskey(d,:hyperViscosityM) ? parse(Int,d[:hyperViscosityM]) : 2
        hyperviscositytype = HyperViscosity{νh,m}()
    else
        hyperviscositytype = NoHyperViscosity()
    end

    if haskey(d,:densityStratification) 

        haskey(d,:gravityDirection) ? (gdir = Symbol(d[:gravityDirection])) : (gdir = :z)
        α = ν/Float64(eval(Meta.parse(d[:Pr])))
        dρdz = Float64(eval(Meta.parse(d[:densityGradient])))
        gval = Float64(eval(Meta.parse(d[:zAcceleration])))/Float64(eval(Meta.parse(d[:referenceDensity])))
        if gdir === :z
            g = Vec(0.0,0.0,gval)
        elseif gdir === :y
            g = Vec(0.0,gval,0.0)
        elseif gdir === :x
            g = Vec(gval,0.0,0.0)
        end

        @info("Reading initial density field rho.$start")
        rho = isfile("rho.$start") ? ScalarField{Float64}("rho.$start") : ScalarField{Float64}((nx,ny,nz),(lx,ly,lz)) 
        gdir = haskey(d,:gravityDirection) ? Symbol(d[:gravityDirection]) : :z
        densitytimestep = if integrator === :Euller
            Euller{variableTimeStep,idt}(Ref(idt))
        elseif integrator === :Adams_Bashforth3rdO
            Adams_Bashforth3rdO{variableTimeStep,idt}()
        else
            ETD3rdO(variableTimeStep,idt,hyperviscositytype,α)
        end 

        lestypedensity = if haskey(d,:lesModel)
            if haskey(d,:densityLesModel)
                if d[:densityLesModel] == "vorticityDiffusion"
                    c = haskey(d,:densityVorticityCoefficient) ? parse(Float64,d[:densityVorticityCoefficient]) : 1.0
                    VorticityDiffusion(c)
                else
                    EddyDiffusion()
                end
            else
                EddyDiffusion()
            end
        else
            NoLESScalar()
        end
        s = BoussinesqApproximation{typeof(densitytimestep),α,dρdz,g,gdir}(rho,densitytimestep,lestypedensity)
    end

    return s
end

function buoyancy_flux(a::AbstractDensityStratification,s::AbstractSimulation) 
    gdir = graddir(a)
    if gdir === :z
       return buoyancy_flux(s.u.c.z.rr ,a.ρ.rr,gravity(s).z,s.reduction)
    elseif gdir === :y
       return buoyancy_flux(s.u.c.y.rr ,a.ρ.rr,gravity(s).y,s.reduction)
    elseif gdir === :x
       return buoyancy_flux(s.u.c.x.rr ,a.ρ.rr,gravity(s).x,s.reduction)
    end
end

function buoyancy_flux(u::AbstractArray{T,3},ρ::AbstractArray{T,3},g::T,reduction::Vector{T}) where {T}
    result = fill!(reduction,0.0)
    @mthreads for k in ZRANGE
        for j in YRANGE
            @inbounds @msimd for i in RXRANGE
                result[Threads.threadid()] += ρ[i,j,k]*u[i,j,k]*g
            end
        end
    end
    return sum(result)/(NRX*NY*NZ)
end