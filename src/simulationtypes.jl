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

@par is_vorticityEquation(s::Type{T}) where {T<:@par(AbstractSimulation)} =
    EquationType === VorticityEquation
is_vorticityEquation(s::AbstractSimulation) = is_vorticityEquation(typeof(s))

@par need_vorticity(s::Type{T}) where {T<:@par(AbstractSimulation)} =
    is_vorticityEquation(T) | hasles(T)
need_vorticity(s::AbstractSimulation) = need_vorticity(typeof(s))

struct @par(Simulation) <: @par(AbstractSimulation)
    u::VectorField{Float64,3,2,false}
    rhs::VectorField{Float64,3,2,false}
    reductionh::Vector{Float64}
    reductionv::Vector{Float64}
    xspec::PaddedArray{Float64,1,0,true}
    yspec::PaddedArray{Float64,1,0,true}
    equation::EquationType
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
    dtspec::Int
    hspec::Array{Float64,3}
    vspec::Array{Float64,3}
    specH::Array{Float64,1}
    tspecH::Array{Float64,1}
    specV::Array{Float64,1}
    kspec::ScalarField{Float64,3,2,false}
    nlstats::Base.RefValue{Tuple{Float64,Float64}}
    pressstats::Base.RefValue{Tuple{Float64,Float64}}
  
    @par function @par(Simulation)(u::VectorField,equation,timestep,passivescalar,densitystratification,lesmodel,forcing,hv,iteration,time,dtout,dtstats,dtspecs) 

        rhs = similar(u)
        kspec = ScalarField{Float64}(size(u.r),(LX,LY,LZ))
  
        reductionh = zeros(THR ? Threads.nthreads() : 1)
        reductionv = zeros(THR ? Threads.nthreads() : 1)

        xspec = PaddedArray(NRX)
        yspec = PaddedArray(NY)

        hspec = zeros(Float64,size(u))
        vspec = zeros(Float64,size(u))
        specH = zeros(Float64,length(KH))
        tspecH = zeros(Float64,length(KH))
        specV = zeros(Float64,length(KRZ))
        nlstats = Ref((1.0,1.0))
        pressstats = Ref((1.0,1.0))

        return @par(new)(u,rhs,reductionh,reductionv,xspec,yspec,equation,timestep,passivescalar,densitystratification,lesmodel,forcing,hv,iteration,time,dtout,dtstats,dtspecs,hspec,vspec,specH,tspecH,specV,kspec,nlstats,pressstats)
    end

end

is_output_time(s) = (s.timestep.x.iteration[] != 0) & (mod(s.iteration[],s.dtoutput) == 0)
is_stats_time(s) = (mod(s.iteration[],s.dtstats) == 0)
is_spec_time(s) = (mod(s.iteration[],s.dtspec) == 0)

##################################### Equation type #########################################

struct VorticityEquation end

struct NavierStokesEquation{T}
    uu::SymTrTenField{T,3,2,false}
end

msg(::VorticityEquation) = "Navier-Stokes in rotational form ((∇×u)×u)"
msg(::NavierStokesEquation) = "Navier-Stokes in conservation form (∇⋅(uu))"

##################################### Equation type #########################################

@inline @par nuh(s::Type{T}) where {T<:@par(AbstractSimulation)} = nuh(HyperViscosityType)
@inline nuh(s::AbstractSimulation) = nuh(typeof(s))

@inline @par get_hyperviscosity_exponent(s::Type{T}) where {T<:@par(AbstractSimulation)} = get_hyperviscosity_exponent(HyperViscosityType)
@inline get_hyperviscosity_exponent(s::AbstractSimulation) = get_hyperviscosity_exponent(typeof(s))

@par function Base.show(io::IO,s::@par(Simulation))
smsg = """
Fluid Flow Simulation
$(msg(s.equation))

nx: $(NRX)
ny: $NY
nz: $NZ
x domain size: $(LX)*2π
y domain size: $(LY)*2π
z domain size: $(LZ)*2π

Kinematic Viscosity: $(ν)

Velocity time-stepping method: $(typeof(s.timestep.x))
Dealias cutoff: $(DEALIAS_TYPE)
Number of Threads: $NT
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

    statsheader(a::AbstractPassiveScalar) = "scalar,scalarp2,dscalardxp2,dscalardyp2,dscalardzp2,scalardiss"

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
# Hyper viscosity Type

include("Hyperviscosity/hyperviscosity_types.jl")

# ==========================================================================================================
# Density Stratification Types

include("density_stratification_types.jl")

# ==========================================================================================
# Forcing Scheme

include("Forcing_methods/forcing_types.jl")

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
    dtspecs = parse(Int,d[:dtSpec])
    dtout = parse(Int,d[:writeTime])

    if haskey(d,:equation)
        if d[:equation] == "NS"
            equation = NavierStokesEquation{Float64}(SymTrTenField((nx,ny,nz),(lx,ly,lz)))
        else
            equation = VorticityEquation()
        end
    else
        equation = VorticityEquation()
    end

    @info("Reading initial velocity field u1.$start u2.$start u3.$start")
    u = VectorField{Float64}("u1.$start","u2.$start","u3.$start")

    kxp = reshape(rfftfreq(nx,lx),(ncx,1,1))
    kyp = reshape(fftfreq(ny,ly),(1,ny,1))
    kzp = reshape(fftfreq(nz,lz),(1,1,nz))

    haskey(d,:threaded) ? (tr = parse(Bool,d[:threaded])) : (tr = true)

    tr || FFTW.set_num_threads(1)
    nt = tr ? Threads.nthreads() : 1 

    b = Globals.splitrange(lrs, nt)

    haskey(d,:cutoff) ? (cutoffr = Float64(eval(Meta.parse(d[:cutoff])))) : (cutoffr = 2/3)

    cutoff = (cutoffr*kxp[end])^2

    if haskey(d,:hyperViscosity)
        if d[:hyperViscosity] == "spectralBarrier"
            initkp = haskey(d,:initk) ? parse(Float64,d[:initk]) : cutoffr*2KX[end] / 3
            endK = haskey(d,:endk) ? parse(Float64,d[:endk]) : sqrt(cutoffr*3)*KX[end]
            hyperviscositytype = SpectralBarrier(initkp,endK)
        else
            νh = parse(Float64,d[:hyperViscosityCoefficient])
            m = haskey(d,:hyperViscosityM) ? parse(Int,d[:hyperViscosityM]) : 2
            hyperviscositytype = HyperViscosity{νh,m}()
        end
    else
        hyperviscositytype = NoHyperViscosity()
    end

 
    integrator = haskey(d,:velocityTimeStep) ? Symbol(d[:velocityTimeStep]) : :ETD3rdO
    variableTimeStep = haskey(d,:variableTimestepFlag) ? parse(Bool,d[:variableTimestepFlag]) : true
    cfl = haskey(d,:cfl) ? parse(Float64,d[:cfl]) : 0.18
    idt = haskey(d,:dt) ? Float64(eval(Meta.parse(d[:dt]))) : 0.0
    vtimestep = if integrator === :Euller
        VectorTimeStep{cfl}(Euller{variableTimeStep,idt}(Ref(idt)),Euller{variableTimeStep,idt}(Ref(idt)),Euller{variableTimeStep,idt}(Ref(idt)))
    elseif integrator === :Adams_Bashforth3rdO
        VectorTimeStep{cfl}(Adams_Bashforth3rdO{variableTimeStep,idt}(),Adams_Bashforth3rdO{variableTimeStep,idt}(),Adams_Bashforth3rdO{variableTimeStep,idt}())
    else
        VectorTimeStep{cfl}(ETD3rdO(variableTimeStep,idt,hyperviscositytype,ν),ETD3rdO(variableTimeStep,idt,hyperviscositytype,ν),ETD3rdO(variableTimeStep,idt,hyperviscositytype,ν))
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
            ETD3rdO(variableTimeStep,idt,hyperviscositytype,α)
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

    lestype = les_types(d,nx,ny,nz,lx,ly,lz)

    forcingtype = forcing_model(d,nx,ny,nz,ncx,lx)

    s = Simulation{typeof(equation),typeof(vtimestep),
        typeof(scalartype),typeof(densitytype),typeof(lestype),typeof(forcingtype),
        typeof(hyperviscositytype)}(u,equation,vtimestep,scalartype,densitytype,lestype,forcingtype,hyperviscositytype,Ref(start),Ref(starttime),dtout,dtstat,dtspecs)
  #
    return s
end

parameters() = parameters(readglobal())
