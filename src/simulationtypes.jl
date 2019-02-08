#Parent type of all simulations
abstract type @par(AbstractSimulation) end
#=
Simulation will encapsule different structs for: 
  Time-stepping, PassiveScalar, DensityStratification, LESModel, Forcing Scheme
=#

#Traits
@par haspassivescalar(s::Type{T}) where {T<:@par(AbstractSimulation)} =
    PassiveScalarType !== NoPassiveScalar
haspassivescalar(s::AbstractSimulation) = haspassivescalar(typeof(s))

@par hasdensity(s::Type{T}) where {T<:@par(AbstractSimulation)}  =
    DensityStratificationType !== NoDensityStratification
hasdensity(s::AbstractSimulation) = hasdensity(typeof(s))

@par hasles(s::Type{T}) where {T<:@par(AbstractSimulation)}  =
    LESModelType !== NoLESModel
hasles(s::AbstractSimulation) = hasles(typeof(s))

@par hasdensityles(s::Type{T}) where {T<:@par(AbstractSimulation)}  =
    (hasdensity(T) && hasdensityles(DensityStratificationType))
hasdensityles(s::AbstractSimulation) = hasdensityles(typeof(s))

@par haspassivescalarles(s::Type{T}) where {T<:@par(AbstractSimulation)}  =
    (haspassivescalar(T) && haspassivescalarles(PassiveScalarType))
haspassivescalarles(s::AbstractSimulation) = haspassivescalarles(typeof(s))

@par hasforcing(s::Type{T}) where {T<:@par(AbstractSimulation)}  =
    ForcingType !== NoForcing
hasforcing(s::AbstractSimulation) = hasforcing(typeof(s))

@par hashyperviscosity(s::Type{T}) where {T<:@par(AbstractSimulation)} =
    HyperViscosityType !== NoHyperViscosity
hashyperviscosity(s::AbstractSimulation) = hashyperviscosity(typeof(s))

struct @par(Simulation) <: @par(AbstractSimulation)
    u::VectorField{Float64,3,2,false}
    rhs::VectorField{Float64,3,2,false}
    reduction::Vector{Float64}
    timestep::VelocityTimeStepType
    passivescalar::PassiveScalarType
    densitystratification::DensityStratificationType
    lesmodel::LESModelType
    forcing::ForcingType
    hyperviscosity::HyperViscosityType
    iteration::Base.RefValue{Int}
    time::Base.RefValue{Float64}
    dtoutput::Int
    dtstats::Int
  
    @par function @par(Simulation)(u::VectorField,timestep,passivescalar,densitystratification,lesmodel,forcing,hv,iteration,time,dtout,dtstats) 

        rhs = similar(u)
  
        reduction = zeros(THR ? Threads.nthreads() : 1)

        return @par(new)(u,rhs,reduction,timestep,passivescalar,densitystratification,lesmodel,forcing,hv,iteration,time,dtout,dtstats)
    end

end

@inline @par nuh(s::Type{T}) where {T<:@par(AbstractSimulation)} = nuh(HyperViscosityType)
@inline nuh(s::AbstractSimulation) = nuh(typeof(s))

@inline @par get_hyperviscosity_exponent(s::Type{T}) where {T<:@par(AbstractSimulation)} = get_hyperviscosity_exponent(HyperViscosityType)
@inline get_hyperviscosity_exponent(s::AbstractSimulation) = get_hyperviscosity_exponent(typeof(s))

@par function Base.show(io::IO,s::@par(Simulation))
smsg = """
Fluid Flow Simulation

nx: $(NRX)
ny: $NY
nz: $NZ
x domain size: $(LX)*2π
y domain size: $(LY)*2π
z domain size: $(LZ)*2π

Kinematic Viscosity: $(ν)

Velocity time-stepping method: $(typeof(s.timestep.x))
Dealias type: $(DEALIAS_TYPE[1]) $(DEALIAS_TYPE[2])
Threaded: $THR
"""
smsg = join((smsg,msg.(getfield.(Ref(s),sim_fields))...))#msg(s.passivescalar),
#  msg(s.densitystratification),
#  msg(s.lesmodel),
#  msg(s.forcing)))

print(io,smsg)
end

#LES models
include("LESmodels/les_types.jl")

# Simulaiton with Scalar fields ===================================================================================================================================================


abstract type AbstractPassiveScalar{L,TT,α,dρdz,Gdirec} end

    haspassivescalarles(a::Type{T}) where {L,TT,α,dρdz,Gdirec,T<:AbstractPassiveScalar{L,TT,α,dρdz,Gdirec}} = 
        L === NoLESScalar ? false : true
    @inline haspassivescalarles(a::AbstractPassiveScalar) = haspassivescalarles(typeof(a)) 

    has_les_scalar_vorticity_model(a::Type{T}) where {L,TT,α,dρdz,Gdirec,T<:AbstractPassiveScalar{L,TT,α,dρdz,Gdirec}} = 
        is_vorticity_model(L)
    @inline has_les_scalar_vorticity_model(a::AbstractPassiveScalar) = has_les_scalar_vorticity_model(typeof(a)) 
    @inline @par has_les_scalar_vorticity_model(s::Type{T}) where {T<:@par(AbstractSimulation)} = has_les_scalar_vorticity_model(PassiveScalarType)


    diffusivity(a::Type{T}) where {L,TT,α,dρdz,Gdirec,T<:AbstractPassiveScalar{L,TT,α,dρdz,Gdirec}} = 
        α
    @inline diffusivity(a::AbstractPassiveScalar) = diffusivity(typeof(a)) 

    meangradient(a::Type{T}) where {L,TT,α,dρdz,Gdirec,T<:AbstractPassiveScalar{L,TT,α,dρdz,Gdirec}} = 
        dρdz
    @inline meangradient(a::AbstractPassiveScalar{L,TT,α,dρdz,Gdirec}) where {L,TT,α,dρdz,Gdirec} = meangradient(typeof(a))

    graddir(a::Type{T}) where {L,TT,α,dρdz,Gdirec,T<:AbstractPassiveScalar{L,TT,α,dρdz,Gdirec}} = 
        Gdirec
    @inline graddir(a::AbstractPassiveScalar{L,TT,α,dρdz,Gdirec}) where {L,TT,α,dρdz,Gdirec} = graddir(typeof(a))

    initialize!(a::AbstractPassiveScalar,s::AbstractSimulation) = initialize!(a.timestep,parent(real(a.rhs)),diffusivity(a),s)

    statsheader(a::AbstractPassiveScalar) = "scalar,scalarp2,dscalardxp2,dscalardyp2,dscalardzp2"

    stats(a::AbstractPassiveScalar,s::AbstractSimulation) = scalar_stats(a.φ,a,s)

struct NoPassiveScalar <: AbstractPassiveScalar{NoLESScalar,nothing,nothing,nothing,nothing} end

    initialize!(a::NoPassiveScalar,s::AbstractSimulation) = nothing

    statsheader(a::NoPassiveScalar) = ""

    stats(a::NoPassiveScalar,s::AbstractSimulation) = ()

    msg(a::NoPassiveScalar) = "\nPassive Scalar: No passive scalar\n"


struct PassiveScalar{L,TTimeStep, α #=Difusitivity = ν/Pr =#,
                  dρdz #=Linear mean profile=#, Gdirec #=Axis of mean profile =#} <: AbstractPassiveScalar{L,TTimeStep,α,dρdz,Gdirec}
    φ::ScalarField{Float64,3,2,false}
    rhs::ScalarField{Float64,3,2,false}
    flux::VectorField{Float64,3,2,false}
    lesmodel::L
    timestep::TTimeStep
    reduction::Vector{Float64}

    function PassiveScalar{TT,α,dρdz,Gdirec}(ρ,les,timestep) where {TT,α,dρdz,Gdirec}
        ρrhs = similar(ρ)
        flux = VectorField(size(ρ.field.r),(ρ.kx.l,ρ.ky.l,ρ.kz.l))
        reduction = zeros(THR ? Threads.nthreads() : 1)
        return new{typeof(les),TT,α,dρdz,Gdirec}(ρ,ρrhs,flux,les,timestep,reduction)
    end
end 

msg(a::PassiveScalar{L,TT,α,dρdz,Gdirec}) where {L,TT,α,dρdz,Gdirec} = """

Passive Scalar: true
Scalar diffusivity: $(α)
Scalar mean gradient: $(dρdz)
Scalar mean gradient direction: $(Gdirec)
Scalar time-stepping method: $(TT)
Scalar LES model: $(L)

"""
# ==========================================================================================================

include("density_stratification_types.jl")

# ==========================================================================================
# Forcing Scheme

include("Forcing_methods/forcing_types.jl")

# Hyper viscosity Type

abstract type AbstractHyperViscosity end

#statsheader(a::AbstractForcing) = ""

struct NoHyperViscosity <: AbstractHyperViscosity end

    statsheader(a::AbstractHyperViscosity) = ""

    stats(a::AbstractHyperViscosity,s::AbstractSimulation) = ()

    msg(a::NoHyperViscosity) = "\nHyper viscosity: no hyperviscosity\n\n"

    @inline nuh(::Type{NoHyperViscosity}) = nothing
    @inline nuh(a::AbstractHyperViscosity) = nuh(typeof(a))

    @inline get_hyperviscosity_exponent(::Type{NoHyperViscosity}) = nothing
    @inline get_hyperviscosity_exponent(a::AbstractHyperViscosity) = get_hyperviscosity_exponent(typeof(a))

struct HyperViscosity{νh,M} <: AbstractHyperViscosity
end

    @inline nuh(::Type{<:HyperViscosity{n,M}}) where {n,M} = n
    @inline get_hyperviscosity_exponent(::Type{<:HyperViscosity{n,M}}) where {n,M} = M

    msg(a::HyperViscosity{nh,M}) where {nh,M} = "\nHyper viscosity: νh = $(nh), m = $(M)\n\n"

# Initializan function =========================================================================================================================================================================================================

function parameters(d::Dict)

    nx = parse(Int,d[:nx])
    ny = parse(Int,d[:ny])
    nz = parse(Int,d[:nz])

    ncx = div(nx,2)+1
    lcs = ncx*ny*nz
    lcv = 3*lcs
    lrs = 2*lcs
    lrv = 2*lcv

    lx = Float64(eval(Meta.parse(d[:xDomainSize])))
    ly = Float64(eval(Meta.parse(d[:yDomainSize])))
    lz = Float64(eval(Meta.parse(d[:zDomainSize])))
    ν = Float64(eval(Meta.parse(d[:kinematicViscosity])))

    start = haskey(d,:start) ? parse(Int,d[:start]) : 0
    starttime = haskey(d,:startTime) ? parse(Float64,d[:startTime]) : 0.0
    dtstat = parse(Int,d[:dtStat])
    dtout = parse(Int,d[:writeTime])

    @info("Reading initial velocity field u1.$start u2.$start u3.$start")
    u = VectorField{Float64}("u1.$start","u2.$start","u3.$start")

    kxp = reshape(rfftfreq(nx,lx),(ncx,1,1))
    kyp = reshape(fftfreq(ny,ly),(1,ny,1))
    kzp = reshape(fftfreq(nz,lz),(1,1,nz))

    haskey(d,:threaded) ? (tr = parse(Bool,d[:threaded])) : (tr = true)

    tr || FFTW.set_num_threads(1)
    nt = tr ? Threads.nthreads() : 1 

    b = Globals.splitrange(lrs, nt)

    haskey(d,:dealias) ? (Dealiastype = Symbol(d[:dealias])) : (Dealiastype = :sphere)
    haskey(d,:cutoff) ? (cutoffr = Float64(eval(Meta.parse(d[:cutoff])))) : (cutoffr = 15/16)

    cutoff = (cutoffr*kxp[end])^2

#    dealias = BitArray(undef,(ncx,ny,nz))
    #if Dealiastype == :sphere
        #@. dealias = (kxp^2 + kyp^2 + kzp^2) > cutoff
    #elseif Dealiastype == :cube
        #@. dealias = (kxp^2 > cutoff) | (kyp^2 > cutoff) | (kzp^2 > cutoff)
    #end
  
    integrator = haskey(d,:velocityTimeStep) ? Symbol(d[:velocityTimeStep]) : :Adams_Bashforth3rdO
    variableTimeStep = haskey(d,:variableTimestepFlag) ? parse(Bool,d[:variableTimestepFlag]) : true
    cfl = haskey(d,:cfl) ? parse(Float64,d[:cfl]) : 0.18
    idt = haskey(d,:dt) ? Float64(eval(Meta.parse(d[:dt]))) : 0.0
    vtimestep = if integrator === :Euller
        VectorTimeStep{cfl}(Euller{variableTimeStep,idt}(Ref(idt)),Euller{variableTimeStep,idt}(Ref(idt)),Euller{variableTimeStep,idt}(Ref(idt)))
    elseif integrator === :Adams_Bashforth3rdO
        VectorTimeStep{cfl}(Adams_Bashforth3rdO{variableTimeStep,idt}(),Adams_Bashforth3rdO{variableTimeStep,idt}(),Adams_Bashforth3rdO{variableTimeStep,idt}())
    else
        VectorTimeStep{cfl}(ETD3rdO{variableTimeStep,idt,haskey(d,:hyperViscosity) ? true : false}(),ETD3rdO{variableTimeStep,idt,haskey(d,:hyperViscosity) ? true : false}(),ETD3rdO{variableTimeStep,idt, haskey(d,:hyperViscosity) ? true : false}())
    end

    if haskey(d,:passiveScalar)
        α = ν/Float64(eval(Meta.parse(d[:scalarPr])))
        dρdz = Float64(eval(Meta.parse(d[:scalarGradient])))
        @info("Reading initial scalar field scalar.$start")
        rho = isfile("scalar.$start") ? ScalarField{Float64}("scalar.$start") : ScalarField{Float64}((nx,ny,nz),(lx,ly,lz)) 
        scalardir = haskey(d,:scalarDirection) ? Symbol(d[:scalarDirection]) : :z
        scalartimestep = if integrator === :Euller
            Euller{variableTimeStep,idt}(Ref(idt))
        elseif integrator === :Adams_Bashforth3rdO
            Adams_Bashforth3rdO{variableTimeStep,idt}()
        else
            ETD3rdO{variableTimeStep,idt,false}()
        end 

        lestypescalar = if haskey(d,:lesModel)
            if haskey(d,:scalarLesModel)
                if d[:scalarLesModel] == "vorticityDiffusion"
                    c = haskey(d,:scalarVorticityCoefficient) ? parse(Float64,d[:scalarVorticityCoefficient]) : 1.0
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
 
        scalartype = PassiveScalar{typeof(scalartimestep),α,dρdz,scalardir}(rho,lestypescalar,scalartimestep)
    else
        scalartype = NoPassiveScalar()
    end

    densitytype = stratificationtype(d,start,integrator,nx,ny,nz,lx,ly,lz,variableTimeStep,idt)

    if haskey(d,:lesModel)
        if d[:lesModel] == "Smagorinsky"
            c = haskey(d,:smagorinskyConstant) ? Float64(eval(Meta.parse(d[:smagorinskyConstant]))) : 0.17 
            Δ = haskey(d,:filterWidth) ? Float64(eval(Meta.parse(d[:filterWidth]))) : 2*(lx*2π/nx)  
            lestype = Smagorinsky(c,Δ,(nx,ny,nz))
        elseif d[:lesModel] == "Vreman"
            c = haskey(d,:vremanConstant) ? Float64(eval(Meta.parse(d[:vremanConstant]))) : 2*(0.17)^2
            Δ = haskey(d,:filterWidth) ? Float64(eval(Meta.parse(d[:filterWidth]))) : 2*(lx*2π/nx)  
            lestype = VremanLESModel(c,Δ,(nx,ny,nz))
        elseif d[:lesModel] == "productionViscosity"
            c = haskey(d,:productionConstant) ? Float64(eval(Meta.parse(d[:productionConstant]))) : 0.4
            Δ = haskey(d,:filterWidth) ? Float64(eval(Meta.parse(d[:filterWidth]))) : 2*(lx*2π/nx)  
            lestype = ProductionViscosityLESModel(c,Δ,(nx,ny,nz))
        elseif d[:lesModel] == "dynamicSmagorinsky"
            Δ = haskey(d,:filterWidth) ? Float64(eval(Meta.parse(d[:filterWidth]))) : 2*(lx*2π/nx)
            tΔ = haskey(d,:TestFilterWidth) ? Float64(eval(Meta.parse(d[:TestFilterWidth]))) : 2*Δ
            backscatter = haskey(d,:cmin) ? parse(Float64,d[:cmin]) : 0.0
            lestype = DynamicSmagorinsky(Δ,tΔ,(nx,ny,nz), backscatter)
        elseif d[:lesModel] == "SandP"
            cb = haskey(d,:pConstant) ? Float64(eval(Meta.parse(d[:pConstant]))) : 0.1
            Slestype = if d[:sLesModel] == "Smagorinsky"
                c = haskey(d,:smagorinskyConstant) ? Float64(eval(Meta.parse(d[:smagorinskyConstant]))) : 0.17 
                Δ = haskey(d,:filterWidth) ? Float64(eval(Meta.parse(d[:filterWidth]))) : 2*(lx*2π/nx)  
                Smagorinsky(c,Δ,(nx,ny,nz))
            elseif d[:sLesModel] == "Vreman"
                c = haskey(d,:vremanConstant) ? Float64(eval(Meta.parse(d[:vremanConstant]))) : 2*(0.17)^2
                Δ = haskey(d,:filterWidth) ? Float64(eval(Meta.parse(d[:filterWidth]))) : 2*(lx*2π/nx)  
                lestype = VremanLESModel(c,Δ,(nx,ny,nz))
            elseif d[:sLesModel] == "productionViscosity"
                c = haskey(d,:productionConstant) ? Float64(eval(Meta.parse(d[:productionConstant]))) : 0.4
                Δ = haskey(d,:filterWidth) ? Float64(eval(Meta.parse(d[:filterWidth]))) : 2*(lx*2π/nx)  
                lestype = ProductionViscosityLESModel(c,Δ,(nx,ny,nz))
            elseif d[:sLesModel] == "dynamicSmagorinsky"
                Δ = haskey(d,:filterWidth) ? Float64(eval(Meta.parse(d[:filterWidth]))) : 2*(lx*2π/nx)
                tΔ = haskey(d,:TestFilterWidth) ? Float64(eval(Meta.parse(d[:TestFilterWidth]))) : 2*Δ
                backscatter = haskey(d,:cmin) ? parse(Float64,d[:cmin]) : 0.0
                DynamicSmagorinsky(Δ,tΔ,(nx,ny,nz), backscatter)
            end
            lestype = SandP(cb,Slestype)
        end
    else
        lestype = NoLESModel()
    end

    forcingtype = forcing_model(d,nx,ny,nz,ncx)

    if haskey(d,:hyperViscosity)
        νh = parse(Float64,d[:hyperViscosity])
        m = haskey(d,:hyperViscosityM) ? parse(Int,d[:hyperViscosityM]) : 2
        hyperviscositytype = HyperViscosity{νh,m}()
    else
        hyperviscositytype = NoHyperViscosity()
    end

    s = Simulation{typeof(vtimestep),
        typeof(scalartype),typeof(densitytype),typeof(lestype),typeof(forcingtype),
        typeof(hyperviscositytype)}(u,vtimestep,scalartype,densitytype,lestype,forcingtype,hyperviscositytype,Ref(start),Ref(starttime),dtout,dtstat)
  #

    @info(s)
    return s
end

parameters() = parameters(readglobal())
