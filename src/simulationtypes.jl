#abstract type @par(BoussinesqSimulation) end
#abstract type @par(ScalarSimulation) end

#Parent type of all simulations
abstract type @par(AbstractSimulation) end
#=
Simulation will encapsule different structs for: 
  Time-stepping, PassiveScalar, DensityStratification, LESModel, Forcing Scheme
=#

#Traits
@par haspassivescalar(s::Type{T}) where {T<:@par(AbstractSimulation)} =
  PassiveScalarType !== NoPassiveScalar

@par hasdensity(s::Type{T}) where {T<:@par(AbstractSimulation)}  =
  DensityStratificationType !== NoDensityStratification

@par hasles(s::Type{T}) where {T<:@par(AbstractSimulation)}  =
  LESModelType !== NoLESModel

@par hasforcing(s::Type{T}) where {T<:@par(AbstractSimulation)}  =
  ForcingType !== NoForcing

struct @par(Simulation) <: @par(AbstractSimulation)
  u::VectorField{PaddedArray{Float64,4,false}}
  rhs::VectorField{PaddedArray{Float64,4,false}}
  aux::VectorField{PaddedArray{Float64,4,false}}
  p::FFTW.rFFTWPlan{Float64,-1,true,4}
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
    p.pinv = plan_irfft!(aux,1:3,flags=FFTW.MEASURE)

    reduction = zeros(Thr ? Threads.nthreads() : 1)

    return @par(new)(u,rhs,aux,p,reduction,dealias,timestep,passivescalar,densitystratification,lesmodel,forcing)
  end

end

@par ν(s::@par(Simulation)) = ν

@par function Base.show(io::IO,s::@par(Simulation))
msg = """
Fluid Flow Simulation

nx: $(Nrx)
ny: $Ny
nz: $Nz
x domain size: $(Lx)*2π
y domain size: $(Ly)*2π
z domain size: $(Lz)*2π

Viscosity: $(ν)

Time Step method: $(typeof(s.timestep.x))
Dealias type: $Dealias
Threaded: $Thr
"""
print(io,msg)
end

# Simulaiton with Scalar fields ===================================================================================================================================================
abstract type AbstractPassiveScalar end

struct NoPassiveScalar <: AbstractPassiveScalar end

struct PassiveScalar{TTimeStep, α #=Difusitivity = ν/Pr =#,
                  dρdz #=Linear mean profile=#, Gdirec #=Axis of mean profile =#} <: AbstractPassiveScalar
  s::PaddedArray{Float64,3,false}
  ps::FFTW.rFFTWPlan{Float64,-1,true,3}
  srhs::PaddedArray{Float64,3,false}
  timestep::TTimeStep

  function PassiveScalar{TT,α,dρdz,Gdirec}(ρ,timestep) where {TT,α,dρdz,Gdirec}
    ρrhs = similar(ρ)
    info("Calculating FFTW in-place forward plan for scalar field")
    ps = plan_rfft!(ρrhs,flags=FFTW.MEASURE)
    info("Calculating FFTW in-place backward plan for scalar field")
    ps.pinv = plan_irfft!(ρrhs,flags=FFTW.MEASURE) 
    return new{TT,α,dρdz,Gdirec}(ρ,ps,ρrhs,timestep)
  end
end 

# ==========================================================================================================

abstract type AbstractDensityStratification end

struct NoDensityStratification <: AbstractDensityStratification end

struct BoussinesqApproximation{TTimeStep, α #=Difusitivity = ν/Pr =#,
                   dρdz #=Linear mean profile=#, g #=This is actually g/ρ₀ =#, 
                   Gdirec#=Gravity direction =#} <: AbstractDensityStratification
  ρ::PaddedArray{Float64,3,false}
  ps::FFTW.rFFTWPlan{Float64,-1,true,3}
  ρrhs::PaddedArray{Float64,3,false}
  timestep::TTimeStep

  function BoussinesqApproximation{TT,α,dρdz,g,Gdirec}(ρ,timestep) where {TT,α,dρdz,g,Gdirec}
    ρrhs = similar(ρ)
    info("Calculating FFTW in-place forward plan for scalar field")
    ps = plan_rfft!(ρrhs,flags=FFTW.MEASURE)
    info("Calculating FFTW in-place backward plan for scalar field")
    ps.pinv = plan_irfft!(ρrhs,flags=FFTW.MEASURE) 
    return new{TT,α,dρdz,g,Gdirec}(ρ,ps,ρrhs,timestep)
  end
end 

# ==========================================================================================
# LES Model

abstract type AbstractLESModel end

struct NoLESModel <: AbstractLESModel end

# ==========================================================================================
# Forcing Scheme

abstract type AbstractForcing end

struct NoForcing <: AbstractForcing end

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

  lx = parse(Float64,d[:xDomainSize])
  ly = parse(Float64,d[:yDomainSize])
  lz = parse(Float64,d[:zDomainSize])
  ν = parse(Float64,d[:kinematicViscosity])
  info("Reading initial velocity field")
  u = VectorField("u1.0","u2.0","u3.0",nx,ny,nz)

  kxp = reshape(rfftfreq(nx,lx),(ncx,1,1))
  kyp = reshape(fftfreq(ny,ly),(1,ny,1))
  kzp = reshape(fftfreq(nz,lz),(1,1,nz))

  haskey(d,:threaded) ? (tr = parse(Bool,d[:threaded])) : (tr = true)

  tr && FFTW.set_num_threads(Threads.nthreads())
 
  if tr
    nt = Threads.nthreads()
    lr = ncx*2*ny*nz
    a = UnitRange{Int}[]
    if lr%nt == 0
      init = 1
      n = lr÷nt
      for i=1:nt
        stop=init+n-1
        push!(a,init:stop)
        init = stop+1
      end
    else
      # TODO 
    end
    b = (a...,)
  else
    lr = ncx*2*ny*nz
    b = (1:lr,)
  end


  haskey(d,:dealias) ? (Dealiastype = Symbol(d[:dealias])) : (Dealiastype = :sphere)


  cutoff = (2kxp[end]/3)^2

  dealias = BitArray(ncx,ny,nz)
  if Dealiastype == :sphere
    @. dealias = (kxp^2 + kyp^2 + kzp^2) > cutoff
  elseif Dealiastype == :cube
    @. dealias = (kxp^2 > cutoff) | (kyp^2 > cutoff) | (kzp^2 > cutoff)
  end

  kx = (kxp...)
  ky = (kyp...)
  kz = (kzp...)

  kxr = 1:div(2ncx,3)
  kyr = (1:(div(ny,3)+1),(ny-div(ny,3)+1):ny)
  rz = vcat(1:(div(nz,3)+1),(nz-div(nz,3)+1):nz)
  kzr = (rz...)
  
  isfile("fftw_wisdom") && FFTW.import_wisdom("fftw_wisdom")
  
  integrator = haskey(d,:velocityTimeStep) ? parse(d[:velocityTimeStep]) : Adams_Bashforth3rdO
  vtimestep = if integrator === :Euller
      VectorTimeStep(Euller(),Euller(),Euller())
    else
      VectorTimeStep(Adams_Bashforth3rdO(kxr,kyr,kzr),Adams_Bashforth3rdO(kxr,kyr,kzr),Adams_Bashforth3rdO(kxr,kyr,kzr))
    end

  if haskey(d,:passiveScalar)
    α = ν/parse(Float64,d[:scalarPr])
    dρdz = parse(Float64,d[:scalarGradient])
    info("Reading initial scalar field")
    rho = isfile("scalar.0") ? PaddedArray("scalar.0",(nx,ny,nz),true) : PaddedArray(zeros(nx,ny,nz)) 
    scalardir = haskey(d,:scalarDirection) ? Symbol(d[:scalarDirection]) : :z
    scalartimestep = Adams_Bashforth3rdO(kxr,kyr,kzr) 
    scalartype = PassiveScalar{typeof(scalartimestep),α,dρdz,scalardir}(rho,scalartimestep)
  else
    scalartype = NoPassiveScalar()
  end

  if haskey(d,:densityStratification) 

    haskey(d,:gravityDirection) ? (gdir = Symbol(d[:gravityDirection])) : (gdir = :z)
    α = ν/parse(Float64,d[:Pr])
    dρdz = parse(Float64,d[:densityGradient])
    g = parse(Float64,d[:zAcceleration])/parse(Float64,d[:referenceDensity])
    info("Reading initial density field")
    rho = isfile("rho.0") ? PaddedArray("rho.0",(nx,ny,nz),true) : PaddedArray(zeros(nx,ny,nz)) 
    gdir = haskey(d,:gravityDirection) ? Symbol(d[:gravityDirection]) : :z
    densitytimestep = Adams_Bashforth3rdO(kxr,kyr,kzr) 
    densitytype = BoussinesqApproximation{typeof(densitytimestep),α,dρdz,g,gdir}(rho,densitytimestep)

  else
    densitytype = NoDensityStratification()
  end

  lestype = NoLESModel()
  forcingtype = NoForcing()

  s = Simulation{lx,ly,lz,ncx,ny,nz,lcs,lcv,nx,lrs,lrv,ν,
      typeof(vtimestep),
      typeof(scalartype),typeof(densitytype),typeof(lestype),typeof(forcingtype),
      Dealiastype,
      kxr,kyr,kzr,kx,ky,kz,tr,b}(u,dealias,vtimestep,scalartype,densitytype,lestype,forcingtype)
  #

  FFTW.export_wisdom("fftw_wisdom")
  info(s)
  return s
end

parameters() = parameters(readglobal())

@par sizecomp(s::@par(AbstractSimulation)) = (Kxr,Kyr,Kzr)
@par wavenumber(s::@par(AbstractSimulation)) = (kx,ky,kz)