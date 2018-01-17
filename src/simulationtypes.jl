# Temporary
abstract type @par(AbstractParameters) end
abstract type @par(ScalarParameters) <: @par(AbstractParameters) end
abstract type @par(BoussinesqParameters) <: @par(ScalarParameters) end
#Parent type of all simulations
abstract type @par(AbstractSimulation) end

#Traits
Base.@pure isscalar(s::Union{AbstractSimulation,Type{<:AbstractSimulation}}) = false
Base.@pure isbuoyant(s::Union{AbstractSimulation,Type{<:AbstractSimulation}}) = false
Base.@pure isles(s::Union{AbstractSimulation,Type{<:AbstractSimulation}}) = false
Base.@pure isforced(s::Union{AbstractSimulation,Type{<:AbstractSimulation}}) = false

@def GenericParameters begin
  u::VectorField{PaddedArray{Float64,4,false}}
  rhs::VectorField{PaddedArray{Float64,4,false}}
  aux::VectorField{PaddedArray{Float64,4,false}}
  nx::Int64
  ny::Int64
  nz::Int64
  lx::Float64
  ly::Float64
  lz::Float64
  ν::Float64
  p::FFTW.rFFTWPlan{Float64,-1,true,4}
  rm1x::Array{Float64,3}
  rm1y::Array{Float64,3}
  rm1z::Array{Float64,3}
  rm2x::Array{Float64,3}
  rm2y::Array{Float64,3}
  rm2z::Array{Float64,3}
  reduction::Vector{Float64}
  dealias::BitArray{3}
end

@def GenericParametersCalculation begin
  rhs = similar(u)
  aux = similar(u)

  aux = VectorField(PaddedArray(nx,ny,nz,3))
  info("Calculating FFTW in-place forward plan for velocity field")
  p = plan_rfft!(aux,1:3,flags=FFTW.MEASURE)
  info("Calculating FFTW in-place backward plan for velocity field")
  p.pinv = plan_irfft!(aux,1:3,flags=FFTW.MEASURE)
  rm1x = Array{Float64}(2length(Kxr),length(Kyr[1])+length(Kyr[2]),length(Kzr))
  rm1y = Array{Float64}(2length(Kxr),length(Kyr[1])+length(Kyr[2]),length(Kzr))
  rm1z = Array{Float64}(2length(Kxr),length(Kyr[1])+length(Kyr[2]),length(Kzr))
  rm2x = Array{Float64}(2length(Kxr),length(Kyr[1])+length(Kyr[2]),length(Kzr))
  rm2y = Array{Float64}(2length(Kxr),length(Kyr[1])+length(Kyr[2]),length(Kzr))
  rm2z = Array{Float64}(2length(Kxr),length(Kyr[1])+length(Kyr[2]),length(Kzr))

  reduction = Vector{Float64}(Thr ? Threads.nthreads() : 1)
end

struct @par(SimpleSimulation) <: @par(AbstractSimulation)
  @GenericParameters
  
  @par function @par(SimpleSimulation)(u::VectorField{PaddedArray{Float64,4,false}},nx::Int64,ny::Int64,nz::Int64,lx::Float64,ly::Float64,lz::Float64,ν::Float64,dealias) 

    @GenericParametersCalculation    

    return @par(new)(u,rhs,aux,nx,ny,nz,lx,ly,lz,ν,p,rm1x,rm1y,rm1z,rm2x,rm2y,rm2z,reduction,dealias)
  end

end

@par function Base.show(io::IO,s::@par(SimpleSimulation))
msg = """
Fluid Flow Simulation

nx: $(Nrx)
ny: $Ny
nz: $Nz
x domain size: $(s.lx)*2π
y domain size: $(s.ly)*2π
z domain size: $(s.lz)*2π

Viscosity: $(s.ν)

Time Step method: $Integrator
Dealias type: $Dealias
Threaded: $Thr
"""
print(io,msg)
end

# Simulaiton with Scalar fields ===================================================================================================================================================

@def GenericScalarParameters begin
  ρ::PaddedArray{Float64,3,false}
  ps::FFTW.rFFTWPlan{Float64,-1,true,3}
  α::Float64 #Difusitivity = ν/Pr  
  dρdz::Float64  
  ρrhs::PaddedArray{Float64,3,false}
  rrm1::Array{Float64,3}
  rrm2::Array{Float64,3}
end

@def GenericScalarParametersCalculation begin
  ρrhs = similar(ρ)
  info("Calculating FFTW in-place forward plan for scalar field")
  ps = plan_rfft!(ρrhs,flags=FFTW.MEASURE)
  info("Calculating FFTW in-place backward plan for scalar field")
  ps.pinv = plan_irfft!(ρrhs,flags=FFTW.MEASURE)
  rrm1 = Array{Float64}(2length(Kxr),length(Kyr[1])+length(Kyr[2]),length(Kzr))
  rrm2 = Array{Float64}(2length(Kxr),length(Kyr[1])+length(Kyr[2]),length(Kzr))
end

struct @par(PassiveScalarSimulation) <: @par(AbstractSimulation)
  @GenericParameters
  @GenericScalarParameters

  @par function @par(PassiveScalarSimulation)(u::VectorField, nx::Integer, ny::Integer, nz::Integer, lx::Real, ly::Real, lz::Real, ν::Real,dealias, ρ::PaddedArray, α::Real,dρdz::Real) 
    
    @GenericParametersCalculation
    @GenericScalarParametersCalculation

    return @par(new)(u,rhs,aux,nx,ny,nz,lx,ly,lz,ν,p,rm1x,rm1y,rm1z,rm2x,rm2y,rm2z,reduction,dealias,ρ,ps,α,dρdz, ρrhs, rrm1,rrm2)
  end

end

Base.@pure isscalar(s::Union{PassiveScalarSimulation,Type{<:PassiveScalarSimulation}}) = true

@par function Base.show(io::IO,s::@par(PassiveScalarSimulation))
  msg = """
  Fluid Flow Simulation with Passive Scalar
  
  nx: $(Nrx)
  ny: $Ny
  nz: $Nz
  x domain size: $(s.lx)*2π
  y domain size: $(s.ly)*2π
  z domain size: $(s.lz)*2π
  
  Viscosity: $(s.ν)
  Scalar Diffusivity: $(s.α)
  Scalar mean gradient: $(s.dρdz)
  Scalar mean gradient direction: $GDirec
  
  Time Step method: $Integrator
  Dealias type: $Dealias
  Threaded: $Thr
  """
  print(io,msg)
end

# Boussinesq Fluid Flow ================================================================================================================================================================================

struct @par(BoussinesqSimulation) <: @par(AbstractSimulation)
  @GenericParameters
  @GenericScalarParameters
  g::Float64 #This is actually g/ρ₀

  @par function @par(BoussinesqSimulation)(u::VectorField, nx::Integer, ny::Integer, nz::Integer, lx::Real, ly::Real, lz::Real, ν::Real,dealias, ρ::PaddedArray, α::Real, dρdz::Real, g::Real)
    
    @GenericParametersCalculation
    @GenericScalarParametersCalculation

    return @par(new)(u,rhs,aux,nx,ny,nz,lx,ly,lz,ν,p,rm1x,rm1y,rm1z,rm2x,rm2x,rm2z,reduction,dealias,ρ,ps,α,dρdz,ρrhs,rrm1,rrm2,g)
  end

end

Base.@pure isscalar(s::Union{BoussinesqSimulation,Type{<:BoussinesqSimulation}}) = true
Base.@pure isbuoyant(s::Union{BoussinesqSimulation,Type{<:BoussinesqSimulation}}) = true

@par function Base.show(io::IO,s::@par(BoussinesqSimulation))
  msg = """
  Boussinesq Fluid Flow Simulation
  
  nx: $(Nrx)
  ny: $Ny
  nz: $Nz
  x domain size: $(s.lx)*2π
  y domain size: $(s.ly)*2π
  z domain size: $(s.lz)*2π
  
  Viscosity: $(s.ν)
  Density Diffusivity: $(s.α)
  Density mean gradient: $(s.dρdz)
  Gravity acceleration / reference density: $(s.g)
  Gravity direction: $GDirec
  
  Time Step method: $Integrator
  Dealias type: $Dealias
  Threaded: $Thr
  """
  print(io,msg)
end

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

  haskey(d,:timeIntegrator) ? (integrator = Symbol(d[:timeIntegrator])) : (integrator = :Adams_Bashforth3rdO)
  integrator in (:Euller,:Adams_Bashforth3rdO) || error("Unkown time integration method in global file: $integrator")

  haskey(d,:dealias) ? (Dealiastype = Symbol(d[:dealias])) : (Dealiastype = :sphere)

  haskey(d,:gravityDirection) ? (gdir = Symbol(d[:gravityDirection])) : (gdir = :z)

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

  if haskey(d,:model)

    model = Symbol(d[:model])

    if model == :PassiveScalar

      α = ν/parse(Float64,d[:Pr])
      dρdz = parse(Float64,d[:densityGradient])
      info("Reading initial scalar field")
      rho = isfile("rho.0") ? PaddedArray("rho.0",(nx,ny,nz),true) : PaddedArray(zeros(nx,ny,nz)) 

      s = PassiveScalarSimulation{ncx,ny,nz,lcs,lcv,nx,lrs,lrv,
        integrator,Dealiastype,
        kxr,kyr,kzr,kx,ky,kz,gdir,tr,b}(u,nx,ny,nz,lx,ly,lz,ν,dealias,rho,α,dρdz)

    elseif model == :Boussinesq 

      α = ν/parse(Float64,d[:Pr])
      dρdz = parse(Float64,d[:densityGradient])
      g = parse(Float64,d[:zAcceleration])/parse(Float64,d[:referenceDensity])
      info("Reading initial density field")
      rho = isfile("rho.0") ? PaddedArray("rho.0",(nx,ny,nz),true) : PaddedArray(zeros(nx,ny,nz)) 

      s = BoussinesqSimulation{ncx,ny,nz,lcs,lcv,nx,lrs,lrv,
        integrator,Dealiastype,
        kxr,kyr,kzr,kx,ky,kz,gdir,tr,b}(u,nx,ny,nz,lx,ly,lz,ν,dealias,rho,α,dρdz,g)

    else

      error("Unkown Model in global file: $model")

    end

  else

    s = SimpleSimulation{ncx,ny,nz,lcs,lcv,nx,lrs,lrv,
      integrator,Dealiastype,
      kxr,kyr,kzr,kx,ky,kz,:z,tr,b}(u,nx,ny,nz,lx,ly,lz,ν,dealias)

  end

  FFTW.export_wisdom("fftw_wisdom")
  info(s)
  return s
end

parameters() = parameters(readglobal())

@par sizecomp(s::@par(AbstractSimulation)) = (Kxr,Kyr,Kzr)
@par wavenumber(s::@par(AbstractSimulation)) = (kx,ky,kz)