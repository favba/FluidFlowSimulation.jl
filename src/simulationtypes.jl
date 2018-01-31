abstract type @par(BoussinesqSimulation) end
abstract type @par(ScalarSimulation) end

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

struct @par(Simulation) <: @par(AbstractSimulation)
  u::VectorField{PaddedArray{Float64,4,false}}
  rhs::VectorField{PaddedArray{Float64,4,false}}
  aux::VectorField{PaddedArray{Float64,4,false}}
  p::FFTW.rFFTWPlan{Float64,-1,true,4}
  pb::FFTW.rFFTWPlan{Complex{Float64},1,true,4}
  reduction::Vector{Float64}
  dealias::BitArray{3}
  timestep::VelocityTimeStepType
  passivescalar::PassiveScalarType
  densitystratification::DensityStratificationType
  lesmodel::LESModelType
  forcing::ForcingType
  
  @par function @par(Simulation)(u::VectorField{PaddedArray{Float64,4,false}},dealias::BitArray{3},timestep,passivescalar,densitystratification,lesmodel,forcing) 

    rhs = similar(u)
    aux = similar(u)
  
    aux = VectorField(PaddedArray(Nrx,Ny,Nz,3))
    info("Calculating FFTW in-place forward plan for velocity field")
    p = plan_rfft!(aux,1:3,flags=FFTW.MEASURE)
    info("Calculating FFTW in-place backward plan for velocity field")
    pb = plan_brfft!(aux,1:3,flags=FFTW.MEASURE)

    reduction = zeros(Thr ? Threads.nthreads() : 1)

    return @par(new)(u,rhs,aux,p,pb,reduction,dealias,timestep,passivescalar,densitystratification,lesmodel,forcing)
  end

end

@par nu(s::@par(Simulation)) = ν
@par ngridpoints(s::@par(Simulation)) = (Nrx,Ny,Nz)
@par domainlength(s::@par(Simulation)) = (Lx,Ly,Lz)

@par function Base.show(io::IO,s::@par(Simulation))
smsg = """
Fluid Flow Simulation

nx: $(Nrx)
ny: $Ny
nz: $Nz
x domain size: $(Lx)*2π
y domain size: $(Ly)*2π
z domain size: $(Lz)*2π

Kinematic Viscosity: $(ν)

Velocity time-stepping method: $(typeof(s.timestep.x))
Dealias type: $(Dealias[1]) $(Dealias[2])
Threaded: $Thr
"""
smsg = join((smsg,msg(s.passivescalar),
  msg(s.densitystratification),
  msg(s.lesmodel),
  msg(s.forcing)))

print(io,smsg)
end

# Simulaiton with Scalar fields ===================================================================================================================================================
abstract type AbstractPassiveScalar{TT,α,dρdz,Gdirec} end

  diffusivity(a::AbstractPassiveScalar{TT,α,dρdz,Gdirec}) where {TT,α,dρdz,Gdirec} = 
    α
  meangradient(a::AbstractPassiveScalar{TT,α,dρdz,Gdirec}) where {TT,α,dρdz,Gdirec} = 
    dρdz
  graddir(a::AbstractPassiveScalar{TT,α,dρdz,Gdirec}) where {TT,α,dρdz,Gdirec} = 
    Gdirec

  initialize!(a::AbstractPassiveScalar,s::AbstractSimulation) = initialize!(a.timestep,parent(real(a.ρrhs)),s)

  statsheader(a::AbstractPassiveScalar) = "scalar,scalar^2,dscalardx^2,dscalardy^2,dscalardz^2"

  stats(a::AbstractPassiveScalar,s::AbstractSimulation) = scalar_stats(a,s)

struct NoPassiveScalar <: AbstractPassiveScalar{nothing,nothing,nothing,nothing} end

  initialize!(a::NoPassiveScalar,s::AbstractSimulation) = nothing

  statsheader(a::NoPassiveScalar) = ""

  stats(a::NoPassiveScalar,s::AbstractSimulation) = ()

  msg(a::NoPassiveScalar) = "\nPassive Scalar: No passive scalar\n"

struct PassiveScalar{TTimeStep, α #=Difusitivity = ν/Pr =#,
                  dρdz #=Linear mean profile=#, Gdirec #=Axis of mean profile =#} <: AbstractPassiveScalar{TTimeStep,α,dρdz,Gdirec}
  ρ::PaddedArray{Float64,3,false}
  ps::FFTW.rFFTWPlan{Float64,-1,true,3}
  pbs::FFTW.rFFTWPlan{Complex{Float64},1,true,3}
  ρrhs::PaddedArray{Float64,3,false}
  timestep::TTimeStep

  function PassiveScalar{TT,α,dρdz,Gdirec}(ρ,timestep) where {TT,α,dρdz,Gdirec}
    ρrhs = similar(ρ)
    info("Calculating FFTW in-place forward plan for scalar field")
    ps = plan_rfft!(ρrhs,flags=FFTW.MEASURE)
    info("Calculating FFTW in-place backward plan for scalar field")
    pbs = plan_brfft!(ρrhs,flags=FFTW.MEASURE) 
    return new{TT,α,dρdz,Gdirec}(ρ,ps,pbs,ρrhs,timestep)
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

  diffusivity(a::AbstractDensityStratification{TT,α,dρdz,g,Gdirec}) where {TT,α,dρdz,g,Gdirec} = 
    α
  meangradient(a::AbstractDensityStratification{TT,α,dρdz,g,Gdirec}) where {TT,α,dρdz,g,Gdirec} = 
    dρdz
  gravity(a::AbstractDensityStratification{TT,α,dρdz,g,Gdirec}) where {TT,α,dρdz,g,Gdirec} = 
    g
  graddir(a::AbstractDensityStratification{TT,α,dρdz,g,Gdirec}) where {TT,α,dρdz,g,Gdirec} = 
    Gdirec


  initialize!(a::AbstractDensityStratification,s::AbstractSimulation) = initialize!(a.timestep,parent(real(a.ρrhs)),s)

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
  ρ::PaddedArray{Float64,3,false}
  ps::FFTW.rFFTWPlan{Float64,-1,true,3}
  pbs::FFTW.rFFTWPlan{Complex{Float64},1,true,3}
  ρrhs::PaddedArray{Float64,3,false}
  timestep::TTimeStep

  function BoussinesqApproximation{TT,α,dρdz,g,Gdirec}(ρ,timestep) where {TT,α,dρdz,g,Gdirec}
    ρrhs = similar(ρ)
    info("Calculating FFTW in-place forward plan for scalar field")
    ps = plan_rfft!(ρrhs,flags=FFTW.MEASURE)
    info("Calculating FFTW in-place backward plan for scalar field")
    pbs = plan_brfft!(ρrhs,flags=FFTW.MEASURE) 
    return new{TT,α,dρdz,g,Gdirec}(ρ,ps,pbs,ρrhs,timestep)
  end
end 

initialize!(a::BoussinesqApproximation,s::AbstractSimulation) = initialize!(a.timestep,parent(real(a.ρrhs)),s)

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

is_Smagorinsky(a::AbstractLESModel) = false
is_SandP(a::AbstractLESModel) = false
scalarmodel(a::AbstractLESModel) = NoLESScalar

struct NoLESModel <: AbstractLESModel end

statsheader(a::NoLESModel) = ""

stats(a::NoLESModel,s::AbstractSimulation) = ()

msg(a::NoLESModel) = "\nLES model: No LES model\n"

abstract type AbstractLESScalar end

struct NoLESScalar <: AbstractLESScalar end

struct EddyDiffusion <: AbstractLESScalar 
  gradρ::VectorField{PaddedArray{Float64,4,false}}
end

EddyDiffusion(nx,ny,nz) = EddyDiffusion(VectorField(PaddedArray(zeros(nx,ny,nz,3))))

# Smagorinsky Model Start ======================================================

abstract type EddyViscosityModel <: AbstractLESModel end

struct Smagorinsky{c,Δ,ScalarType<:AbstractLESScalar} <: EddyViscosityModel
  tau::SymmetricTracelessTensor
  pt::FFTW.rFFTWPlan{Float64,-1,true,4}
  pbt::FFTW.rFFTWPlan{Complex{Float64},1,true,4}
  scalar::ScalarType
end

function Smagorinsky(c::Real,Δ::Real,scalar::Bool,dim::NTuple{3,Integer}) 
  data = SymmetricTracelessTensor(dim)
  info("Calculating FFTW in-place forward plan for symmetric traceless tensor field")
  pt = plan_rfft!(data,1:3,flags=FFTW.MEASURE)
  info("Calculating FFTW in-place backward plan for symmetric traceless tensor field")
  pbt = plan_brfft!(data,1:3,flags=FFTW.MEASURE)
  fill!(data,0)
  scalart = scalar ? EddyDiffusion(dim...) : NoLESScalar()
  return Smagorinsky{c,Δ,typeof(scalart)}(data,pt,pbt,scalart)
end

Smagorinsky(c::Real,Δ::Real,dim::NTuple{3,Integer}) = Smagorinsky(c,Δ,false,dim)

is_Smagorinsky(a::Smagorinsky) = true

cs(s::Union{T,Type{T}}) where {c,Δ,scalar,T<:Smagorinsky{c,Δ,scalar}} = c
Delta(s::Union{T,Type{T}}) where {c,Δ,scalar,T<:Smagorinsky{c,Δ,scalar}} = Δ

scalarmodel(s::Union{T,Type{T}}) where {c,Δ,scalar,T<:Smagorinsky{c,Δ,scalar}} = scalar

statsheader(a::Smagorinsky) = ""

stats(a::Smagorinsky,s::AbstractSimulation) = ()

msg(a::Smagorinsky) = "\nLES model: Smagorinsky\nConstant: $(cs(a))\nFilter Width: $(Delta(a))\n"

# Smagorinsky Model End ======================================================

# Smagorinsky+P Model Start ======================================================

struct SandP{cs,cβ,Δ,ScalarType<:AbstractLESScalar} <: AbstractLESModel
  tau::SymmetricTracelessTensor
  pt::FFTW.rFFTWPlan{Float64,-1,true,4}
  pbt::FFTW.rFFTWPlan{Complex{Float64},1,true,4}
  scalar::ScalarType
end

function SandP(c::Real,cb::Real,Δ::Real,scalar::Bool,dim::NTuple{3,Integer}) 
  data = SymmetricTracelessTensor(dim)
  info("Calculating FFTW in-place forward plan for symmetric traceless tensor field")
  pt = plan_rfft!(data,1:3,flags=FFTW.MEASURE)
  info("Calculating FFTW in-place backward plan for symmetric traceless tensor field")
  pbt = plan_brfft!(data,1:3,flags=FFTW.MEASURE)
  fill!(data,0)
  scalart = scalar ? EddyDiffusion(dim...) : NoLESScalar()
  return SandP{c,cb,Δ,typeof(scalart)}(data,pt,pbt,scalart)
end

is_SandP(a::SandP) = true

cs(s::Union{T,Type{T}}) where {c,cb,Δ,T<:SandP{c,cb,Δ}} = c
cbeta(s::Union{T,Type{T}}) where {c,cb,Δ,T<:SandP{c,cb,Δ}} = cb
Delta(s::Union{T,Type{T}}) where {c,cb,Δ,T<:SandP{c,cb,Δ}} = Δ

scalarmodel(s::Union{T,Type{T}}) where {c,cb,Δ,scalar,T<:SandP{c,cb,Δ,scalar}} = scalar

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

  lx = Float64(eval(parse(d[:xDomainSize])))
  ly = Float64(eval(parse(d[:yDomainSize])))
  lz = Float64(eval(parse(d[:zDomainSize])))
  ν = Float64(eval(parse(d[:kinematicViscosity])))

  start = haskey(d,:start) ? d[:start] : "0"

  info("Reading initial velocity field u1.$start u2.$start u3.$start")
  u = VectorField("u1.$start","u2.$start","u3.$start",nx,ny,nz)

  kxp = reshape(rfftfreq(nx,lx),(ncx,1,1))
  kyp = reshape(fftfreq(ny,ly),(1,ny,1))
  kzp = reshape(fftfreq(nz,lz),(1,1,nz))

  haskey(d,:threaded) ? (tr = parse(Bool,d[:threaded])) : (tr = true)

  tr && FFTW.set_num_threads(Threads.nthreads())
  nt = tr ? Threads.nthreads() : 1 

  b = splitrange(lrs, nt)

  haskey(d,:dealias) ? (Dealiastype = Symbol(d[:dealias])) : (Dealiastype = :sphere)
  haskey(d,:cutoff) ? (cutoffr = Float64(eval(parse(d[:cutoff])))) : (cutoffr = 2/3)

  cutoff = (cutoffr*kxp[end])^2

  dealias = BitArray(ncx,ny,nz)
  if Dealiastype == :sphere
    @. dealias = (kxp^2 + kyp^2 + kzp^2) > cutoff
  elseif Dealiastype == :cube
    @. dealias = (kxp^2 > cutoff) | (kyp^2 > cutoff) | (kzp^2 > cutoff)
  end

  kx = (kxp...)
  ky = (kyp...)
  kz = (kzp...)

  kxr = 1:(findfirst(x->x^2>cutoff,kx)-1)
  wly = (findfirst(x->x^2>cutoff,ky)-1)
  kyr = (1:wly,(ny-wly+2):ny)
  wlz = (findfirst(x->x^2>cutoff,kz)-1)
  rz = vcat(1:wlz,(ny-wlz+2):ny)
  #rz = vcat(1:(div(nz,3)+1),(nz-div(nz,3)+1):nz)
  kzr = (rz...)
  
  isfile("fftw_wisdom") && FFTW.import_wisdom("fftw_wisdom")
  
  integrator = haskey(d,:velocityTimeStep) ? parse(d[:velocityTimeStep]) : Adams_Bashforth3rdO
  vtimestep = if integrator === :Euller
      VectorTimeStep(Euller(),Euller(),Euller())
    else
      VectorTimeStep(Adams_Bashforth3rdO(kxr,kyr,kzr),Adams_Bashforth3rdO(kxr,kyr,kzr),Adams_Bashforth3rdO(kxr,kyr,kzr))
  end

  if haskey(d,:passiveScalar)
    α = ν/Float64(eval(parse(d[:scalarPr])))
    dρdz = Float64(eval(parse(d[:scalarGradient])))
    info("Reading initial scalar field scalar.$start")
    rho = isfile("scalar.$start") ? PaddedArray("scalar.$start",(nx,ny,nz),true) : PaddedArray(zeros(nx,ny,nz)) 
    scalardir = haskey(d,:scalarDirection) ? Symbol(d[:scalarDirection]) : :z
    scalartimestep = Adams_Bashforth3rdO(kxr,kyr,kzr) 
    scalartype = PassiveScalar{typeof(scalartimestep),α,dρdz,scalardir}(rho,scalartimestep)
  else
    scalartype = NoPassiveScalar()
  end

  if haskey(d,:densityStratification) 

    haskey(d,:gravityDirection) ? (gdir = Symbol(d[:gravityDirection])) : (gdir = :z)
    α = ν/Float64(eval(parse(d[:Pr])))
    dρdz = Float64(eval(parse(d[:densityGradient])))
    g = Float64(eval(parse(d[:zAcceleration])))/Float64(eval(parse(d[:referenceDensity])))
    info("Reading initial density field rho.$start")
    rho = isfile("rho.$start") ? PaddedArray("rho.$start",(nx,ny,nz),true) : PaddedArray(zeros(nx,ny,nz)) 
    gdir = haskey(d,:gravityDirection) ? Symbol(d[:gravityDirection]) : :z
    densitytimestep = Adams_Bashforth3rdO(kxr,kyr,kzr) 
    densitytype = BoussinesqApproximation{typeof(densitytimestep),α,dρdz,g,gdir}(rho,densitytimestep)

  else
    densitytype = NoDensityStratification()
  end

  if haskey(d,:lesModel)
    if d[:lesModel] == "Smagorinsky"
      c = haskey(d,:smagorinskyConstant) ? Float64(eval(parse(d[:smagrisnkyConstant]))) : 0.17 
      Δ = haskey(d,:filterWidth) ? Float64(eval(parse(d[:smagrisnkyConstant]))) : lx*2π/nx  
      lesscalar = (haskey(d,:passiveScalar) | haskey(d,:densityStratification)) ? true : false
      lestype = Smagorinsky(c,Δ,lesscalar,(nx,ny,nz))
    elseif d[:lesModel] == "Smagorinsky+P"
      c = haskey(d,:smagorinskyConstant) ? Float64(eval(parse(d[:smagrisnkyConstant]))) : 0.17 
      cb = haskey(d,:pTensorConstant) ? Float64(eval(parse(d[:pTensorConstant]))) : 0.17 
      Δ = haskey(d,:filterWidth) ? Float64(eval(parse(d[:smagrisnkyConstant]))) : lx*2π/nx  
      lesscalar = (haskey(d,:passiveScalar) | haskey(d,:densityStratification)) ? true : false
      lestype = SandP(c,cb,Δ,lesscalar,(nx,ny,nz))
    end
  else
  lestype = NoLESModel()
  end
  forcingtype = NoForcing()

  s = Simulation{lx,ly,lz,ncx,ny,nz,lcs,lcv,nx,lrs,lrv,ν,
      typeof(vtimestep),
      typeof(scalartype),typeof(densitytype),typeof(lestype),typeof(forcingtype),
      (Dealiastype,cutoffr),
      kxr,kyr,kzr,kx,ky,kz,tr,nt,b}(u,dealias,vtimestep,scalartype,densitytype,lestype,forcingtype)
  #

  FFTW.export_wisdom("fftw_wisdom")
  info(s)
  return s
end

parameters() = parameters(readglobal())

@par sizecomp(s::@par(AbstractSimulation)) = (Kxr,Kyr,Kzr)
@par wavenumber(s::@par(AbstractSimulation)) = (kx,ky,kz)