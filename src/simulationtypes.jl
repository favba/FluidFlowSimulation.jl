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

@par hasforcing(s::Type{T}) where {T<:@par(AbstractSimulation)}  =
    ForcingType !== NoForcing
hasforcing(s::AbstractSimulation) = hasforcing(typeof(s))

@par hashyperviscosity(s::Type{T}) where {T<:@par(AbstractSimulation)} =
    HyperViscosityType !== NoHyperViscosity
hashyperviscosity(s::AbstractSimulation) = hashyperviscosity(typeof(s))

struct @par(Simulation) <: @par(AbstractSimulation)
    u::VectorField{Float64,3,2,false}
    rhs::VectorField{Float64,3,2,false}
    aux::VectorField{Float64,3,2,false}
    reduction::Vector{Float64}
    timestep::VelocityTimeStepType
    passivescalar::PassiveScalarType
    densitystratification::DensityStratificationType
    lesmodel::LESModelType
    forcing::ForcingType
    hyperviscosity::HyperViscosityType
  
    @par function @par(Simulation)(u::VectorField,timestep,passivescalar,densitystratification,lesmodel,forcing,hv) 

        rhs = similar(u)
        aux = similar(u)
  
        reduction = zeros(THR ? Threads.nthreads() : 1)

        return @par(new)(u,rhs,aux,reduction,timestep,passivescalar,densitystratification,lesmodel,forcing,hv)
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
abstract type AbstractPassiveScalar{TT,α,dρdz,Gdirec} end

    diffusivity(a::Type{T}) where {TT,α,dρdz,Gdirec,T<:AbstractPassiveScalar{TT,α,dρdz,Gdirec}} = 
        α
    @inline diffusivity(a::AbstractPassiveScalar) = diffusivity(typeof(a)) 

    meangradient(a::Type{T}) where {TT,α,dρdz,Gdirec,T<:AbstractPassiveScalar{TT,α,dρdz,Gdirec}} = 
        dρdz
    @inline meangradient(a::AbstractPassiveScalar{TT,α,dρdz,Gdirec}) where {TT,α,dρdz,Gdirec} = meangradient(typeof(a))

    graddir(a::Type{T}) where {TT,α,dρdz,Gdirec,T<:AbstractPassiveScalar{TT,α,dρdz,Gdirec}} = 
        Gdirec
    @inline graddir(a::AbstractPassiveScalar{TT,α,dρdz,Gdirec}) where {TT,α,dρdz,Gdirec} = graddir(typeof(a))

    initialize!(a::AbstractPassiveScalar,s::AbstractSimulation) = initialize!(a.timestep,parent(real(a.ρrhs)),diffusivity(a),s)

    statsheader(a::AbstractPassiveScalar) = "scalar,scalar^2,dscalardx^2,dscalardy^2,dscalardz^2"

    stats(a::AbstractPassiveScalar,s::AbstractSimulation) = scalar_stats(a,s)

struct NoPassiveScalar <: AbstractPassiveScalar{nothing,nothing,nothing,nothing} end

    initialize!(a::NoPassiveScalar,s::AbstractSimulation) = nothing

    statsheader(a::NoPassiveScalar) = ""

    stats(a::NoPassiveScalar,s::AbstractSimulation) = ()

    msg(a::NoPassiveScalar) = "\nPassive Scalar: No passive scalar\n"

struct PassiveScalar{TTimeStep, α #=Difusitivity = ν/Pr =#,
                  dρdz #=Linear mean profile=#, Gdirec #=Axis of mean profile =#} <: AbstractPassiveScalar{TTimeStep,α,dρdz,Gdirec}
    ρ::ScalarField{Float64,3,2,false}
    ρrhs::ScalarField{Float64,3,2,false}
    timestep::TTimeStep

    function PassiveScalar{TT,α,dρdz,Gdirec}(ρ,timestep) where {TT,α,dρdz,Gdirec}
        ρrhs = similar(ρ)
        return new{TT,α,dρdz,Gdirec}(ρ,ρrhs,timestep)
    end
end 

msg(a::PassiveScalar{TT,α,dρdz,Gdirec}) where {TT,α,dρdz,Gdirec} = """

Passive Scalar: true
Scalar diffusivity: $(α)
Scalar mean gradient: $(dρdz)
Scalar mean gradient direction: $(Gdirec)
Scalar time-stepping method: $(TT)

"""
# ==========================================================================================================

abstract type AbstractDensityStratification{TT,α,dρdz,g,Gdirec} end

    diffusivity(a::Type{T}) where {TT,α,dρdz,g,Gdirec,T<:AbstractDensityStratification{TT,α,dρdz,g,Gdirec}} = 
        α
    @inline diffusivity(a::AbstractDensityStratification{TT,α,dρdz,g,Gdirec}) where {TT,α,dρdz,g,Gdirec} = diffusivity(typeof(a))
    @inline @par diffusivity(s::Type{T}) where {T<:@par(AbstractSimulation)} = diffusivity(DensityStratificationType)
    @inline diffusivity(s::AbstractSimulation) = diffusivity(typeof(s))

    meangradient(a::Type{T}) where {TT,α,dρdz,g,Gdirec,T<:AbstractDensityStratification{TT,α,dρdz,g,Gdirec}} = 
        dρdz
    @inline meangradient(a::AbstractDensityStratification{TT,α,dρdz,g,Gdirec}) where {TT,α,dρdz,g,Gdirec} = meangradient(typeof(a))

    gravity(a::Type{T}) where {TT,α,dρdz,g,Gdirec,T<:AbstractDensityStratification{TT,α,dρdz,g,Gdirec}} = 
        g
    @inline gravity(a::AbstractDensityStratification{TT,α,dρdz,g,Gdirec}) where {TT,α,dρdz,g,Gdirec} = gravity(typeof(a))
    @inline @par gravity(s::Type{T}) where {T<:@par(AbstractSimulation)} = gravity(DensityStratificationType)
    @inline gravity(s::AbstractSimulation) = gravity(typeof(s))

    graddir(a::Type{T}) where {TT,α,dρdz,g,Gdirec,T<:AbstractDensityStratification{TT,α,dρdz,g,Gdirec}} = 
        Gdirec
    @inline graddir(a::AbstractDensityStratification{TT,α,dρdz,g,Gdirec}) where {TT,α,dρdz,g,Gdirec} = graddir(typeof(a))


    initialize!(a::AbstractDensityStratification,s::AbstractSimulation) = initialize!(a.timestep,parent(real(a.ρrhs)),diffusivity(a),s)

    statsheader(a::AbstractDensityStratification) = "rho,rho^2,drhodx^2,drhody^3,drhodz^2"

    stats(a::AbstractDensityStratification,s::AbstractSimulation) = scalar_stats(a,s)

struct NoDensityStratification <: AbstractDensityStratification{nothing,nothing,nothing,nothing,nothing} end

    initialize!(a::NoDensityStratification,s::AbstractSimulation) = nothing

    statsheader(a::NoDensityStratification) = ""

    stats(a::NoDensityStratification,s::AbstractSimulation) = ()

    msg(a::NoDensityStratification) = "\nDensity Stratification: No density stratification\n"

struct BoussinesqApproximation{TTimeStep, α #=Difusitivity = ν/Pr =#,
                   dρdz #=Linear mean profile=#, g #=This is actually g/ρ₀ =#, 
                   Gdirec#=Gravity direction =#} <: AbstractDensityStratification{TTimeStep,α,dρdz,g,Gdirec}
    ρ::ScalarField{Float64,3,2,false}
    ρrhs::ScalarField{Float64,3,2,false}
    timestep::TTimeStep
    reduction::Vector{Float64}

    function BoussinesqApproximation{TT,α,dρdz,g,Gdirec}(ρ,timestep,tr) where {TT,α,dρdz,g,Gdirec}
        ρrhs = similar(ρ)
        reduction = zeros(tr ? Threads.nthreads() : 1)
        return new{TT,α,dρdz,g,Gdirec}(ρ,ρrhs,timestep,reduction)
    end
end 

initialize!(a::BoussinesqApproximation,s::AbstractSimulation) = initialize!(a.timestep,parent(real(a.ρrhs)),diffusivity(a),s)

msg(a::BoussinesqApproximation{TT,α,dρdz,g,Gdirec}) where {TT,α,dρdz,g,Gdirec} = """

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

abstract type AbstractLESScalar end

struct NoLESScalar <: AbstractLESScalar end

struct EddyDiffusion{VecType} <: AbstractLESScalar 
    gradρ::VecType
end

    EddyDiffusion(nx,ny,nz) = EddyDiffusion(VectorField(nx,ny,nz))

# Smagorinsky Model Start ======================================================

abstract type EddyViscosityModel <: AbstractLESModel end

struct Smagorinsky{c,Δ,ScalarType<:AbstractLESScalar,TensorType} <: EddyViscosityModel
    tau::TensorType
    scalar::ScalarType
    reduction::Vector{Float64}
end

function Smagorinsky(c::Real,Δ::Real,scalar::Bool,dim::NTuple{3,Integer},tr) 
    data = SymTrTenField(dim...)
    #fill!(data,0)
    scalart = scalar ? EddyDiffusion(dim...) : NoLESScalar()
    reduction = zeros(tr ? Threads.nthreads() : 1)
    return Smagorinsky{c,Δ,typeof(scalart),typeof(data)}(data,scalart,reduction)
end

Smagorinsky(c::Real,Δ::Real,dim::NTuple{3,Integer}) = Smagorinsky(c,Δ,false,dim)

is_Smagorinsky(a::Union{<:Smagorinsky,Type{<:Smagorinsky}}) = true
@inline @par is_Smagorinsky(s::Type{T}) where {T<:@par(AbstractSimulation)} = is_Smagorinsky(LESModelType)
@inline is_Smagorinsky(s::T) where {T<:AbstractSimulation} = is_Smagorinsky(T)

cs(s::Type{T}) where {c,Δ,scalar,T<:Smagorinsky{c,Δ,scalar}} = c
@inline cs(s::T) where {c,Δ,scalar,T<:Smagorinsky{c,Δ,scalar}} = cs(T)
Delta(s::Type{T}) where {c,Δ,scalar,T<:Smagorinsky{c,Δ,scalar}} = Δ
@inline Delta(s::T) where {c,Δ,scalar,T<:Smagorinsky{c,Δ,scalar}} = Delta(T)

lesscalarmodel(s::Type{T}) where {c,Δ,scalar,T<:Smagorinsky{c,Δ,scalar}} = scalar
@inline lesscalarmodel(s::T) where {c,Δ,scalar,T<:Smagorinsky{c,Δ,scalar}} = lesscalarmodel(T)
@inline @par lesscalarmodel(s::Type{T}) where {T<:@par(AbstractSimulation)} = lesscalarmodel(LESModelType)
@inline lesscalarmodel(s::AbstractSimulation) = lesscalarmodel(typeof(s))

statsheader(a::Smagorinsky) = ""

stats(a::Smagorinsky,s::AbstractSimulation) = ()

msg(a::Smagorinsky) = "\nLES model: Smagorinsky\nConstant: $(cs(a))\nFilter Width: $(Delta(a))\n"

# Smagorinsky Model End ======================================================

# Smagorinsky+P Model Start ======================================================

struct SandP{cs,cβ,Δ,ScalarType<:AbstractLESScalar,TensorType} <: AbstractLESModel
    tau::TensorType
    scalar::ScalarType
end

function SandP(c::Real,cb::Real,Δ::Real,scalar::Bool,dim::NTuple{3,Integer}) 
    data = SymmetricTracelessTensor(dim)
    fill!(data,0)
    scalart = scalar ? EddyDiffusion(dim...) : NoLESScalar()
    return SandP{c,cb,Δ,typeof(scalart),typeof(data)}(data,scalart)
end

is_SandP(a::Union{<:SandP,Type{<:SandP}}) = true
@inline @par is_SandP(s::Type{T}) where {T<:@par(AbstractSimulation)} = is_SandP(LESModelType)
@inline is_SandP(s::T) where {T<:AbstractSimulation} = is_SandP(T)

cs(s::Type{T}) where {c,cb,Δ,T<:SandP{c,cb,Δ}} = c
@inline cs(s::T) where {c,cb,Δ,T<:SandP{c,cb,Δ}} = cs(T)
cbeta(s::Type{T}) where {c,cb,Δ,T<:SandP{c,cb,Δ}} = cb
@inline cbeta(s::T) where {c,cb,Δ,T<:SandP{c,cb,Δ}} = cbeta(T)
Delta(s::Type{T}) where {c,cb,Δ,T<:SandP{c,cb,Δ}} = Δ
@inline Delta(s::T) where {c,cb,Δ,T<:SandP{c,cb,Δ}} = Delta(T)

lesscalarmodel(s::Type{T}) where {c,cb,Δ,scalar,T<:SandP{c,cb,Δ,scalar}} = scalar
@inline lesscalarmodel(s::T) where {c,cb,Δ,scalar,T<:SandP{c,cb,Δ,scalar}} = lesscalarmodel(T)

statsheader(a::SandP) = ""

stats(a::SandP,s::AbstractSimulation) = ()

msg(a::SandP) = "\nLES model: Smagorinsky + P tensor\nSmagorinsky Constant: $(cs(a))\nP tensor constant: $(cbeta(a))\nFilter Width: $(Delta(a))\n"


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

    start = haskey(d,:start) ? d[:start] : "0"

    @info("Reading initial velocity field u1.$start u2.$start u3.$start")
    u = VectorField(nx,ny,nz)
    read!("u1.$start",u.rr.x)
    read!("u2.$start",u.rr.y)
    read!("u3.$start",u.rr.z)

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
        rho = isfile("scalar.$start") ? PaddedArray("scalar.$start",(nx,ny,nz),true) : PaddedArray(zeros(nx,ny,nz)) 
        scalardir = haskey(d,:scalarDirection) ? Symbol(d[:scalarDirection]) : :z
        scalartimestep = if integrator === :Euller
            Euller{variableTimeStep,idt}(Ref(idt))
        elseif integrator === :Adams_Bashforth3rdO
            Adams_Bashforth3rdO{variableTimeStep,idt}()
        else
            ETD3rdO{variableTimeStep,idt,false}()
        end 
        scalartype = PassiveScalar{typeof(scalartimestep),α,dρdz,scalardir}(rho,scalartimestep)
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
        rho = isfile("rho.$start") ? ScalarField("rho.$start",(nx,ny,nz)) : ScalarField(PaddedArray(nx,ny,nz)) 
        gdir = haskey(d,:gravityDirection) ? Symbol(d[:gravityDirection]) : :z
        densitytimestep = if integrator === :Euller
            Euller{variableTimeStep,idt}(Ref(idt))
        elseif integrator === :Adams_Bashforth3rdO
            Adams_Bashforth3rdO{variableTimeStep,idt}()
        else
            ETD3rdO{variableTimeStep,idt,false}()
        end 
        densitytype = BoussinesqApproximation{typeof(densitytimestep),α,dρdz,g,gdir}(rho,densitytimestep,tr)

    else
        densitytype = NoDensityStratification()
    end

    if haskey(d,:lesModel)
        if d[:lesModel] == "Smagorinsky"
            c = haskey(d,:smagorinskyConstant) ? Float64(eval(Meta.parse(d[:smagorinskyConstant]))) : 0.17 
            Δ = haskey(d,:filterWidth) ? Float64(eval(Meta.parse(d[:filterWidth]))) : 2*(lx*2π/nx)  
            lesscalar = (haskey(d,:passiveScalar) | haskey(d,:densityStratification)) ? true : false
            lestype = Smagorinsky(c,Δ,lesscalar,(nx,ny,nz),tr)
        elseif d[:lesModel] == "Smagorinsky+P"
            c = haskey(d,:smagorinskyConstant) ? Float64(eval(Meta.parse(d[:smagorinskyConstant]))) : 0.17 
            cb = haskey(d,:pTensorConstant) ? Float64(eval(Meta.parse(d[:pTensorConstant]))) : 0.17 
            Δ = haskey(d,:filterWidth) ? Float64(eval(Meta.parse(d[:filterWidth]))) : lx*2π/nx  
            lesscalar = (haskey(d,:passiveScalar) | haskey(d,:densityStratification)) ? true : false
            lestype = SandP(c,cb,Δ,lesscalar,(nx,ny,nz))
        end
    else
        lestype = NoLESModel()
    end
    if haskey(d,:forcing)
        if d[:forcing] == "rfForcing"
            TF = parse(Float64,d[:TF])
            alphac = parse(Float64,d[:alphac])
            kf = parse(Float64,d[:kf])
            nShells2D, maxdk2D, numPtsInShell2D, kh = compute_shells2D(kx,ky,ncx,ny)
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
        typeof(hyperviscositytype)}(u,vtimestep,scalartype,densitytype,lestype,forcingtype,hyperviscositytype)
  #

    FFTW.export_wisdom("fftw_wisdom")
    @info(s)
    return s
end

parameters() = parameters(readglobal())
