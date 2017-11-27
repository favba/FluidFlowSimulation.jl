
struct VectorField{A<:AbstractPaddedArray{Float64, 4, false}} <: AbstractPaddedArray{Float64,4,false}
  data::A
  cx::Array{Complex128,3}
  cy::Array{Complex128,3}
  cz::Array{Complex128,3}
  rx::Array{Float64,3}
  ry::Array{Float64,3}
  rz::Array{Float64,3}

  function VectorField{A}(data::A) where {A<:AbstractPaddedArray{Float64, 4,false}}
    cdims = size(data)
    cnx, cny, cnz, _ = cdims
    cx = unsafe_wrap(Array{Complex128,3},pointer(complex(data)),(cnx,cny,cnz))
    cy = unsafe_wrap(Array{Complex128,3},pointer(complex(data),sub2ind(cdims,1,1,1,2)),(cnx,cny,cnz))
    cz = unsafe_wrap(Array{Complex128,3},pointer(complex(data),sub2ind(cdims,1,1,1,3)),(cnx,cny,cnz))

    rnx,rny,rnz,_ = size(rawreal(data))
    rdims = (rnx,rny,rnz)
    rx = reinterpret(Float64,cx,rdims)
    ry = reinterpret(Float64,cy,rdims)
    rz = reinterpret(Float64,cz,rdims)
    return new{A}(data,cx,cy,cz,rx,ry,rz)
  end
end

VectorField(data::AbstractPaddedArray{Float64,4,false}) = VectorField{typeof(data)}(data)

function VectorField(ux::AbstractString,uy::AbstractString,uz::AbstractString,nx::Integer,ny::Integer,nz::Integer)
  field = VectorField(PaddedArray(nx,ny,nz,3))
  read!(ux,field.rx)
  read!(uy,field.ry)
  read!(uz,field.rz)
  return field
end

@inline Base.real(V::VectorField) = real(V.data)
@inline Base.complex(V::VectorField) = complex(V.data) 
@inline InplaceRealFFTW.rawreal(V::VectorField) = rawreal(V.data)
Base.similar(V::VectorField{A}) where {A} = VectorField{A}(similar(V.data))
Base.copy(V::VectorField{A}) where {A} = VectorField{A}(copy(V.data))

InplaceRealFFTW.rfft!(V::VectorField{A}) where {A} = rfft!(V,1:3) 
InplaceRealFFTW.irfft!(V::VectorField{A}) where {A} = irfft!(V,1:3) 

#------------------------------------------------------------------------------------------------------

abstract type @par(AbstractParameters) end

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
  p::Base.DFT.FFTW.rFFTWPlan{Float64,-1,true,4}
  ip::Base.DFT.ScaledPlan{Complex{Float64},Base.DFT.FFTW.rFFTWPlan{Complex{Float64},1,false,4},Float64}
  rm1x::Array{Float64,3}
  rm1y::Array{Float64,3}
  rm1z::Array{Float64,3}
  rm2x::Array{Float64,3}
  rm2y::Array{Float64,3}
  rm2z::Array{Float64,3}
  reduction::Vector{Float64}
  dealias::BitArray{3}
end

struct @par(Parameters) <: @par(AbstractParameters)
  @GenericParameters
  
  @par function @par(Parameters)(u::VectorField{PaddedArray{Float64,4,false}},nx::Int64,ny::Int64,nz::Int64,lx::Float64,ly::Float64,lz::Float64,ν::Float64,dealias) 
    
    rhs = similar(u)
    aux = similar(u)

    aux = VectorField(PaddedArray(nx,ny,nz,3))
    p = plan_rfft!(aux,1:3,flags=FFTW.MEASURE)
    p.pinv = plan_irfft!(aux,1:3,flags=FFTW.MEASURE)
    ip = Base.DFT.ScaledPlan(FFTW.rFFTWPlan{Complex{Float64},FFTW.BACKWARD,false,4}(complex(aux), real(aux), 1:3, FFTW.MEASURE&FFTW.DESTROY_INPUT,FFTW.NO_TIMELIMIT),Base.DFT.normalization(Float64, size(real(aux)), 1:3))
    rm1x = Array{Float64}(2length(Kxr),length(Kyr),length(Kzr))
    rm1y = Array{Float64}(2length(Kxr),length(Kyr),length(Kzr))
    rm1z = Array{Float64}(2length(Kxr),length(Kyr),length(Kzr))
    rm2x = Array{Float64}(2length(Kxr),length(Kyr),length(Kzr))
    rm2y = Array{Float64}(2length(Kxr),length(Kyr),length(Kzr))
    rm2z = Array{Float64}(2length(Kxr),length(Kyr),length(Kzr))

    reduction = Vector{Float64}(Threads.nthreads())
    return @par(new)(u,rhs,aux,nx,ny,nz,lx,ly,lz,ν,p,ip,rm1x,rm1y,rm1z,rm2x,rm2y,rm2z,reduction,dealias)
  end

end

@par function Base.show(io::IO,s::@par(Parameters))
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
"""
print(io,msg)
return nothing
end

function Parameters(u::VectorField,nx::Integer,ny::Integer,nz::Integer,lx::Real,ly::Real,lz::Real,ν::Real,integrator::Symbol,Deal::Symbol,deat,kx,ky,kz) 
  ncx = div(nx,2)+1
  lcs = ncx*ny*nz
  lcv = 3*lcs
  lrs = 2*lcs
  lrv = 2*lcv
  rx = 1:div(2ncx,3)
  kxr = ((UInt32.(rx))...)
  ry = vcat(UInt32(1):(UInt32(div(ny,3))+1),UInt32(ny-div(ny,3)-1):UInt32(ny))
  kyr = (ry...)
  rz = vcat(1:(div(nz,3)+1),(nz-div(nz,3)-1):nz)
  kzr = (rz...)
  return Parameters{ncx,ny,nz,lcs,lcv,nx,lrs,lrv,integrator,Deal,kxr,kyr,kzr,kx,ky,kz}(u,nx,ny,nz,lx,ly,lz,ν,deat)
end

abstract type @par(ScalarParameters) <: @par(AbstractParameters) end

struct @par(PassiveScalarParameters) <: @par(ScalarParameters)
  @GenericParameters
  ρ::PaddedArray{Float64,3,false}
  ps::Base.DFT.FFTW.rFFTWPlan{Float64,-1,true,3}
  α::Float64 #Difusitivity = ν/Pr  
  dρdz::Float64  
  ρrhs::PaddedArray{Float64,3,false}
  rrm1::Array{Float64,3}
  rrm2::Array{Float64,3}

  @par function @par(PassiveScalarParameters)(u::VectorField, nx::Integer, ny::Integer, nz::Integer, lx::Real, ly::Real, lz::Real, ν::Real, ρ::PaddedArray, α::Real,dρdz::Real,dealias) 
    
    rhs = similar(u)
    aux = similar(u)

    aux = VectorField(PaddedArray(nx,ny,nz,3))
    p = plan_rfft!(aux,1:3,flags=FFTW.MEASURE)
    p.pinv = plan_irfft!(aux,1:3,flags=FFTW.MEASURE)
    ip = Base.DFT.ScaledPlan(FFTW.rFFTWPlan{Complex{Float64},FFTW.BACKWARD,false,4}(complex(aux), real(aux), 1:3, FFTW.MEASURE&FFTW.DESTROY_INPUT,FFTW.NO_TIMELIMIT),Base.DFT.normalization(Float64, size(real(aux)), 1:3))
    rm1x = Array{Float64}(2length(Kxr),length(Kyr),length(Kzr))
    rm1y = Array{Float64}(2length(Kxr),length(Kyr),length(Kzr))
    rm1z = Array{Float64}(2length(Kxr),length(Kyr),length(Kzr))
    rm2x = Array{Float64}(2length(Kxr),length(Kyr),length(Kzr))
    rm2y = Array{Float64}(2length(Kxr),length(Kyr),length(Kzr))
    rm2z = Array{Float64}(2length(Kxr),length(Kyr),length(Kzr))
    ρrhs = similar(ρ)
    ps = plan_rfft!(ρrhs,flags=FFTW.MEASURE)
    ps.pinv = plan_irfft!(ρrhs,flags=FFTW.MEASURE)
    rrm1 = Array{Float64}(2length(Kxr),length(Kyr),length(Kzr))
    rrm2 = Array{Float64}(2length(Kxr),length(Kyr),length(Kzr))

    reduction = Vector{Float64}(Threads.nthreads())

    return @par(new)(u,rhs,aux,nx,ny,nz,lx,ly,lz,ν,p,ip,rm1x,rm1y,rm1z,rm2x,rm2y,rm2z,reduction,dealias,ρ,ps,α,dρdz, ρrhs, rrm1,rrm2)
  end

end

@par function Base.show(io::IO,s::@par(PassiveScalarParameters))
  msg = """
  Fluid Flow Simulation with Passive Scalar
  
  nx: $(Nrx)
  ny: $Ny
  nz: $Nz
  x domain size: $(s.lx)*2π
  y domain size: $(s.ly)*2π
  z domain size: $(s.lz)*2π
  
  Viscosity: $(s.ν)
  Scalar Difusivity: $(s.α)
  Scalar mean gradient: $(s.dρdz)
  
  Time Step method: $Integrator
  Dealias type: $Dealias
  """
  print(io,msg)
  return nothing
end

function PassiveScalarParameters(u::VectorField,nx::Integer,ny::Integer,nz::Integer,lx::Real,ly::Real,lz::Real,ν::Real,ρ::PaddedArray, α::Real,dρdz::Real,integrator::Symbol,Deal::Symbol,deat,kx,ky,kz) 
  ncx = div(nx,2)+1
  lcs = ncx*ny*nz
  lcv = 3*lcs
  lrs = 2*lcs
  lrv = 2*lcv
  rx = 1:div(2ncx,3)
  kxr = ((UInt32.(rx))...)
  ry = vcat(UInt32(1):(UInt32(div(ny,3))+1),UInt32(ny-div(ny,3)-1):UInt32(ny))
  kyr = (ry...)
  rz = vcat(1:(div(nz,3)+1),(nz-div(nz,3)-1):nz)
  kzr = (rz...)
  return PassiveScalarParameters{ncx,ny,nz,lcs,lcv,nx,lrs,lrv,integrator,Deal,kxr,kyr,kzr,kx,ky,kz}(u,nx,ny,nz,lx,ly,lz,ν,ρ,α,dρdz,deat)
end

struct @par(BoussinesqParameters) <: @par(ScalarParameters)
  @GenericParameters
  ρ::PaddedArray{Float64,3,false}
  ps::Base.DFT.FFTW.rFFTWPlan{Float64,-1,true,3}
  α::Float64 #Difusitivity = ν/Pr  
  dρdz::Float64 #This is actually dρsdz/ρ₀
  g::Float64
  ρrhs::PaddedArray{Float64,3,false}
  rrm1::Array{Float64,3}
  rrm2::Array{Float64,3}

  @par function @par(BoussinesqParameters)(u::VectorField, nx::Integer, ny::Integer, nz::Integer, lx::Real, ly::Real, lz::Real, ν::Real, ρ::PaddedArray, α::Real, dρdz::Real, g::Real,dealias)
    
    rhs = similar(u)
    aux = similar(u)

    aux = VectorField(PaddedArray(nx,ny,nz,3))
    p = plan_rfft!(aux,1:3,flags=FFTW.MEASURE)
    p.pinv = plan_irfft!(aux,1:3,flags=FFTW.MEASURE)
    ip = Base.DFT.ScaledPlan(FFTW.rFFTWPlan{Complex{Float64},FFTW.BACKWARD,false,4}(complex(aux), real(aux), 1:3, FFTW.MEASURE&FFTW.DESTROY_INPUT,FFTW.NO_TIMELIMIT),Base.DFT.normalization(Float64, size(real(aux)), 1:3))
    rm1x = Array{Float64}(2length(Kxr),length(Kyr),length(Kzr))
    rm1y = Array{Float64}(2length(Kxr),length(Kyr),length(Kzr))
    rm1z = Array{Float64}(2length(Kxr),length(Kyr),length(Kzr))
    rm2x = Array{Float64}(2length(Kxr),length(Kyr),length(Kzr))
    rm2y = Array{Float64}(2length(Kxr),length(Kyr),length(Kzr))
    rm2z = Array{Float64}(2length(Kxr),length(Kyr),length(Kzr))
    ρrhs = similar(ρ)
    ps = plan_rfft!(ρrhs,flags=FFTW.MEASURE)
    ps.pinv = plan_irfft!(ρrhs,flags=FFTW.MEASURE)
    rrm1 = Array{Float64}(2length(Kxr),length(Kyr),length(Kzr))
    rrm2 = Array{Float64}(2length(Kxr),length(Kyr),length(Kzr))

    reduction = Vector{Float64}(Threads.nthreads())

    return @par(new)(u,rhs,aux,nx,ny,nz,lx,ly,lz,ν,p,ip,rm1x,rm1y,rm1z,rm2x,rm2x,rm2z,reduction,dealias,ρ,ps,α,dρdz,g, ρrhs, rrm1,rrm2)
  end

end

function BoussinesqParameters(u::VectorField,nx::Integer,ny::Integer,nz::Integer,lx::Real,ly::Real,lz::Real,ν::Real,ρ::PaddedArray, dρdz::Real,α::Real,g::Real,integrator::Symbol,Deal::Symbol,deat,kx,ky,kz) 
  ncx = div(nx,2)+1
  lcs = ncx*ny*nz
  lcv = 3*lcs
  lrs = 2*lcs
  lrv = 2*lcv
  rx = 1:div(2ncx,3)
  kxr = ((UInt32.(rx))...)
  ry = vcat(UInt32(1):(UInt32(div(ny,3))+1),UInt32(ny-div(ny,3)-1):UInt32(ny))
  kyr = (ry...)
  rz = vcat(1:(div(nz,3)+1),(nz-div(nz,3)-1):nz)
  kzr = (rz...)
  return BoussinesqParameters{ncx,ny,nz,lcs,lcv,nx,lrs,lrv,integrator,Deal,kxr,kyr,kzr,kx,ky,kz}(u,nx,ny,nz,lx,ly,lz,ν,ρ,α,dρdz,g,deat)
end

function parameters(d::Dict)

  nx = parse(Int,d[:nx])
  ny = parse(Int,d[:ny])
  nz = parse(Int,d[:nz])
  lx = parse(Float64,d[:xDomainSize])
  ly = parse(Float64,d[:yDomainSize])
  lz = parse(Float64,d[:zDomainSize])
  ν = parse(Float64,d[:kinematicViscosity])
  u = VectorField("u1.0","u2.0","u3.0",nx,ny,nz)

  ncx = div(nx,2)+1

  kxp = 2π .* reshape(rfftfreq(nx,lx),(ncx,1,1))
  kyp = 2π .* reshape(fftfreq(ny,ly),(1,ny,1))
  kzp = 2π .* reshape(fftfreq(nz,lz),(1,1,nz))

  FFTW.set_num_threads(Threads.nthreads())
  
  haskey(d,:timeIntegrator) ? (integrator = Symbol(d[:timeIntegrator])) : (integrator = :Adams_Bashforth3rdO)
  integrator in (:Euller,:Adams_Bashforth3rdO) || error("Unkown time integration method in global file: $integrator")

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
 
  isfile("fftw_wisdom") && FFTW.import_wisdom("fftw_wisdom")

  if haskey(d,:model)
    model = Symbol(d[:model])
    if model == :PassiveScalar
      α = ν/parse(Float64,d[:Pr])
      dρdz = parse(Float64,d[:densityGradient])/parse(Float64,d[:referenceDensity])
      rho = isfile("rho.0") ? PaddedArray("rho.0",(nx,ny,nz),padded=true) : PaddedArray(zeros(nx,ny,nz)) 
      s = PassiveScalarParameters(u,nx,ny,nz,lx,ly,lz,ν,rho,α,dρdz,integrator,Dealiastype,dealias,kx,ky,kz)
    elseif model == :Boussinesq 
      α = ν/parse(Float64,d[:Pr])
      dρdz = parse(Float64,d[:densityGradient])/parse(Float64,d[:referenceDensity])
      g = parse(Float64,d[:zAcceleration])
      rho = isfile("rho.0") ? PaddedArray("rho.0",(nx,ny,nz),padded=true) : PaddedArray(zeros(nx,ny,nz)) 
      s = BoussinesqParameters(u,nx,ny,nz,lx,ly,lz,ν,rho,α,dρdz,g,integrator,Dealiastype,dealias,kx,ky,kz)
    else
      error("Unkown Model in global file: $model")
    end
  else
    s = Parameters(u,nx,ny,nz,lx,ly,lz,ν,integrator,Dealiastype,dealias,kx,ky,kz)
  end

  FFTW.export_wisdom("fftw_wisdom")

  return s
end

parameters() = parameters(readglobal())

@par sizecomp(s::@par(AbstractParameters)) = (Kxr,Kyr,Kzr)
@par wavenumber(s::@par(AbstractParameters)) = (kx,ky,kz)