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
    iteration::Ref{Int}
    time::Ref{Float64}
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

# Simulaiton with Scalar fields ===================================================================================================================================================

abstract type AbstractLESScalar end

struct NoLESScalar <: AbstractLESScalar end

struct EddyDiffusion <: AbstractLESScalar end

abstract type AbstractPassiveScalar{L,TT,α,dρdz,Gdirec} end

    haspassivescalarles(a::Type{T}) where {L,TT,α,dρdz,Gdirec,T<:AbstractPassiveScalar{L,TT,α,dρdz,Gdirec}} = 
        L === NoLESScalar ? false : true
    @inline haspassivescalarles(a::AbstractPassiveScalar) = haspassivescalarles(typeof(a)) 

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
        flux = VectorField(size(ρ.field.r),(ρ.space.lx,ρ.space.ly,ρ.space.lz))
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

"""
# ==========================================================================================================

abstract type AbstractDensityStratification{L,TT,α,dρdz,g,Gdirec} end

    hasdensityles(a::Type{T}) where {L,TT,α,dρdz,g,Gdirec,T<:AbstractDensityStratification{L,TT,α,dρdz,g,Gdirec}} = 
        L === NoLESScalar ? false : true
    @inline hasdensityles(a::AbstractDensityStratification{L,TT,α,dρdz,g,Gdirec}) where {L,TT,α,dρdz,g,Gdirec} = hasdensityles(typeof(a))

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

    statsheader(a::AbstractDensityStratification) = "rho,rhop2,drhodxp2,drhodyp3,drhodzp2"

    stats(a::AbstractDensityStratification,s::AbstractSimulation) = scalar_stats(a.ρ,a,s)

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

    function BoussinesqApproximation{TT,α,dρdz,g,Gdirec}(ρ,timestep,tr,les) where {TT,α,dρdz,g,Gdirec}
        ρrhs = similar(ρ)
        flux = VectorField(size(ρ.field.r),(ρ.space.lx,ρ.space.ly,ρ.space.lz))
        reduction = zeros(tr ? Threads.nthreads() : 1)
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

"""
# ==========================================================================================
# LES Model

abstract type AbstractLESModel end

    is_Smagorinsky(a) = false
    is_SandP(a) = false
    lesscalarmodel(a) = NoLESScalar

struct NoLESModel <: AbstractLESModel end

    statsheader(a::NoLESModel) = ""

    stats(a::NoLESModel,s::AbstractSimulation) = ()

    msg(a::NoLESModel) = "\nLES model: No LES model\n"


# Smagorinsky Model Start ======================================================

abstract type EddyViscosityModel <: AbstractLESModel end

struct Smagorinsky{T} <: EddyViscosityModel
    c::T
    Δ²::T
    tau::SymTrTenField{T,3,2,false}
    reduction::Vector{T}
    pr::ScalarField{T,3,2,false}
end

function Smagorinsky(c::T,Δ::Real,dim::NTuple{3,Integer}) where {T<:Real}
    data = SymTrTenField{T}(dim,(LX,LY,LZ))
    pr = ScalarField{T}(dim,(LX,LY,LZ))
    #fill!(data,0)
    reduction = zeros(THR ? Threads.nthreads() : 1)
    return Smagorinsky{T}(c,Δ^2, data,reduction,pr)
end

is_Smagorinsky(a::Union{<:Smagorinsky,Type{<:Smagorinsky}}) = true
@inline @par is_Smagorinsky(s::Type{T}) where {T<:@par(AbstractSimulation)} = is_Smagorinsky(LESModelType)
@inline is_Smagorinsky(s::T) where {T<:AbstractSimulation} = is_Smagorinsky(T)
is_Smagorinsky(a) = false

statsheader(a::Smagorinsky) = "pr"

stats(a::Smagorinsky,s::AbstractSimulation) = (tmean(a.pr.rr,s),)

msg(a::Smagorinsky) = "\nLES model: Smagorinsky\nConstant: $(a.c)\nFilter Width: $(sqrt(a.Δ²))\n"

# Smagorinsky Model End ======================================================

struct DynamicSmagorinsky{T<:Real} <: EddyViscosityModel
    Δ²::T
    Δ̂²::T
    c::ScalarField{T,3,2,false}
    cmin::T
    L::SymTenField{T,3,2,false}
    M::SymTrTenField{T,3,2,false}
    tau::SymTrTenField{T,3,2,false}
    S::SymTrTenField{T,3,2,false}
    û::VectorField{T,3,2,false}
    reduction::Vector{T}
    pr::ScalarField{T,3,2,false}
end

function DynamicSmagorinsky(Δ::T, d2::T, dim::NTuple{3,Integer}, cmin::Real=0.0) where {T<:Real}
    c = ScalarField{T}(dim,(LX,LY,LZ))
    L = SymTenField(dim,(LX,LY,LZ))
    tau = SymTrTenField(dim,(LX,LY,LZ))
    M = similar(tau)
    S = similar(tau)
    u = VectorField{T}(dim,(LX,LY,LZ))
    #fill!(data,0)
    reduction = zeros(THR ? Threads.nthreads() : 1)
    return DynamicSmagorinsky{T}(Δ^2, d2^2, c, cmin, L, M, tau, S, u, reduction, similar(c))
end

DynamicSmagorinsky(Δ::T, dim::NTuple{3,Integer},b::Bool=false) where {T<:Real} = DynamicSmagorinsky(Δ, 2Δ, dim,b)

is_dynamic_les(a::Union{<:DynamicSmagorinsky,<:Type{<:DynamicSmagorinsky}}) = true
@inline @par is_dynamic_les(s::Type{T}) where {T<:@par(AbstractSimulation)} = is_dynamic_les(LESModelType)
@inline is_dynamic_les(s::T) where {T<:AbstractSimulation} = is_dynamic_les(T)
is_dynamic_les(a) = false

statsheader(a::DynamicSmagorinsky) = "pr"

stats(a::DynamicSmagorinsky,s::AbstractSimulation) = (tmean(a.pr.rr,s),)

msg(a::DynamicSmagorinsky) = "\nLES model: Dynamic Smagorinsky\nFilter Width: $(sqrt(a.Δ²))\nTest Filter Width: $(sqrt(a.Δ̂²))\nMinimum coefficient permitted: $(a.cmin)\n"

# Smagorinsky+P Model Start ======================================================

struct SandP{T<:Real,Smodel<:EddyViscosityModel} <: AbstractLESModel
    cb::T # default: 0.1
    Smodel::Smodel
end

@inline function Base.getproperty(a::SandP,s::Symbol)
    if s === :Smodel || s === :cb
        return getfield(a,s)
    else
        return getfield(getfield(a,:Smodel),s)
    end
end

is_SandP(a::Union{<:SandP,Type{<:SandP}}) = true
@inline @par is_SandP(s::Type{T}) where {T<:@par(AbstractSimulation)} = is_SandP(LESModelType)
@inline is_SandP(s::T) where {T<:AbstractSimulation} = is_SandP(T)

@inline is_Smagorinsky(a::Union{<:SandP{t,S},<:Type{SandP{t,S}}}) where {t,S} = is_Smagorinsky(S)
@inline is_dynamic_les(a::Union{<:SandP{t,S},<:Type{SandP{t,S}}}) where {t,S} = is_dynamic_les(S)

statsheader(a::SandP) = "pr"

stats(a::SandP,s::AbstractSimulation) = (tmean(a.pr.rr,s),)

msg(a::SandP) = "\nLES model: SandP\nP tensor constant: $(a.cb)\nEddy Viscosity model: [$(msg(a.Smodel))]\n"


# ==========================================================================================
# Forcing Scheme

abstract type AbstractForcing end

#statsheader(a::AbstractForcing) = ""

    struct NoForcing <: AbstractForcing end

    statsheader(a::NoForcing) = ""

    stats(a::NoForcing,s::AbstractSimulation) = ()

    msg(a::NoForcing) = "\nForcing: No forcing\n"

struct RfForcing{Tf,α,Kf,MaxDk,avgK, Zf} #= Tf = 1.0 , α = 1.0  =# <: AbstractForcing
    Ef::Vector{Float64} # Velocity Field Spectrum
    Em::Vector{Float64} # Target Spectrum
    R::Vector{Float64} # Solution to ODE
  #  Zf::Vector{Float64} # Cutoff function, using as parameter
    #dRdt::Vector{Float64} # Not needed if I use Euller timestep
    factor::Vector{Float64} # Factor to multiply velocity Field
    forcex::PaddedArray{Float64,3,2,false} # Final force
    forcey::PaddedArray{Float64,3,2,false} # Final force
    init::Bool # Tell if the initial condition spectra should be used instead of from data
end

    statsheader(a::RfForcing) = ""

    stats(a::RfForcing,s::AbstractSimulation) = ()

    msg(a::RfForcing) = "\nForcing:  Rf forcing\nTf: $(getTf(a))\nalphac: $(getalpha(a))\nKf: $(getKf(a))\n"

    getTf(f::Type{RfForcing{Tf,α,Kf,MaxDk,avgK, Zf}}) where {Tf,α,Kf,MaxDk,avgK, Zf} = Tf
    @inline getTf(f::RfForcing{Tf,α,Kf,MaxDk,avgK, Zf}) where {Tf,α,Kf,MaxDk,avgK, Zf} = getTf(typeof(f))

    getalpha(f::Type{RfForcing{Tf,α,Kf,MaxDk,avgK, Zf}}) where {Tf,α,Kf,MaxDk,avgK, Zf} = α
    @inline getalpha(f::RfForcing{Tf,α,Kf,MaxDk,avgK, Zf}) where {Tf,α,Kf,MaxDk,avgK, Zf} = getalpha(typeof(f))

    getKf(f::Type{RfForcing{Tf,α,Kf,MaxDk,avgK, Zf}}) where {Tf,α,Kf,MaxDk,avgK, Zf} = Kf
    @inline getKf(f::RfForcing{Tf,α,Kf,MaxDk,avgK, Zf}) where {Tf,α,Kf,MaxDk,avgK, Zf} = getKf(typeof(f))

    getmaxdk(f::Type{RfForcing{Tf,α,Kf,MaxDk,avgK, Zf}}) where {Tf,α,Kf,MaxDk,avgK, Zf} = MaxDk
    @inline getmaxdk(f::RfForcing{Tf,α,Kf,MaxDk,avgK, Zf}) where {Tf,α,Kf,MaxDk,avgK, Zf} = getmaxdk(typeof(f))

    getavgk(f::Type{RfForcing{Tf,α,Kf,MaxDk,AvgK, Zf}}) where {Tf,α,Kf,MaxDk,AvgK, Zf} = AvgK
    @inline getavgk(f::RfForcing{Tf,α,Kf,MaxDk,AvgK, Zf}) where {Tf,α,Kf,MaxDk,AvgK, Zf} = getavgk(typeof(f))

    getZf(f::Type{RfForcing{Tf,α,Kf,MaxDk,AvgK, Zf}}) where {Tf,α,Kf,MaxDk,AvgK, Zf} = Zf
    @inline getZf(f::RfForcing{Tf,α,Kf,MaxDk,AvgK, Zf}) where {Tf,α,Kf,MaxDk,AvgK, Zf} = getZf(typeof(f))

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

    tr && FFTW.set_num_threads(Threads.nthreads())
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

    isfile("fftw_wisdom") && FFTW.import_wisdom("fftw_wisdom")
  
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
            EddyDiffusion()
        else
            NoLESScalar()
        end
        scalartype = PassiveScalar{typeof(scalartimestep),α,dρdz,scalardir}(rho,lestypescalar,scalartimestep)
    else
        scalartype = NoPassiveScalar()
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
            ETD3rdO{variableTimeStep,idt,false}()
        end 
        lestypedensity = if haskey(d,:lesModel)
                EddyDiffusion()
            else
                NoLESScalar()
            end
        densitytype = BoussinesqApproximation{typeof(densitytimestep),α,dρdz,g,gdir}(rho,densitytimestep,tr,lestypedensity)

    else
        densitytype = NoDensityStratification()
    end

    if haskey(d,:lesModel)
        if d[:lesModel] == "Smagorinsky"
            c = haskey(d,:smagorinskyConstant) ? Float64(eval(Meta.parse(d[:smagorinskyConstant]))) : 0.17 
            Δ = haskey(d,:filterWidth) ? Float64(eval(Meta.parse(d[:filterWidth]))) : 2*(lx*2π/nx)  
            lestype = Smagorinsky(c,Δ,(nx,ny,nz))
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
    if haskey(d,:forcing)
        if d[:forcing] == "rfForcing"
            TF = parse(Float64,d[:TF])
            alphac = parse(Float64,d[:alphac])
            kf = parse(Float64,d[:kf])
            nShells2D, maxdk2D, numPtsInShell2D, kh = compute_shells2D(KX,KY,ncx,ny)
            Ef = zeros(length(kh))
            Em = zeros(length(kh))
            R = zeros(length(kh))
            factor = zeros(length(kh))
            forcex = PaddedArray((nx,ny,nz))
            forcey = PaddedArray((nx,ny,nz))
            Zf = calculate_Zf(kf,kh)
            if !isfile("targSpectrum.dat")
                forcingtype = RfForcing{TF, alphac, kf, maxdk2D, (kh...,),Zf}(Ef,Em,R,factor,forcex,forcey,true)
            else
            #todo read spectrum.dat
            end
        end
    else
        forcingtype = NoForcing()
    end

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

    FFTW.export_wisdom("fftw_wisdom")
    @info(s)
    return s
end

parameters() = parameters(readglobal())
