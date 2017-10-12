
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
  kx::SArray{Tuple{Nx,1,1},Float64,3,Nx}
  ky::SArray{Tuple{1,Ny,1},Float64,3,Ny}
  kz::SArray{Tuple{1,1,Nz},Float64,3,Nz}
  p::Base.DFT.FFTW.rFFTWPlan{Float64,-1,true,4}
  ip::Base.DFT.ScaledPlan{Complex{Float64},Base.DFT.FFTW.rFFTWPlan{Complex{Float64},1,false,4},Float64}
  rm1x::Array{Complex128,3}
  rm1y::Array{Complex128,3}
  rm1z::Array{Complex128,3}
  rm2x::Array{Complex128,3}
  rm2y::Array{Complex128,3}
  rm2z::Array{Complex128,3}
  dealias::BitArray{4}
end

struct @par(Parameters) <: @par(AbstractParameters)
  @GenericParameters
  
  @par function @par(Parameters)(u::VectorField{PaddedArray{Float64,4,false}},nx::Int64,ny::Int64,nz::Int64,lx::Float64,ly::Float64,lz::Float64,ν::Float64) 
    
    rhs = similar(u)
    aux = similar(u)

    kx = SArray{Tuple{Nx,1,1}}(reshape(rfftfreq(nx,lx),(Nx,1,1)))
    ky = SArray{Tuple{1,Ny,1}}(reshape(fftfreq(ny,ly),(1,Ny,1)))
    kz = SArray{Tuple{1,1,Nz}}(reshape(fftfreq(nz,lz),(1,1,Nz)))
    
    aux = VectorField(PaddedArray(nx,ny,nz,3))
    p = plan_rfft!(aux,1:3,flags=FFTW.MEASURE)
    p.pinv = plan_irfft!(aux,1:3,flags=FFTW.MEASURE)
    ip = Base.DFT.ScaledPlan(FFTW.rFFTWPlan{Complex{Float64},FFTW.BACKWARD,false,4}(complex(aux), real(aux), 1:3, FFTW.MEASURE&FFTW.DESTROY_INPUT,FFTW.NO_TIMELIMIT),Base.DFT.normalization(Float64, size(real(aux)), 1:3))
    rm1x = Array{Complex128}(length(Kxr),length(Kyr),length(Kzr))
    rm1y = Array{Complex128}(length(Kxr),length(Kyr),length(Kzr))
    rm1z = Array{Complex128}(length(Kxr),length(Kyr),length(Kzr))
    rm2x = Array{Complex128}(length(Kxr),length(Kyr),length(Kzr))
    rm2y = Array{Complex128}(length(Kxr),length(Kyr),length(Kzr))
    rm2z = Array{Complex128}(length(Kxr),length(Kyr),length(Kzr))

    dealias = BitArray(Nx,Ny,Nz,3)
    cutoff = (2kx[end]/3)^2

    if Dealiastype == :sphere
      @. dealias[:,:,:,1] = (kx^2 + ky^2 + kz^2) > cutoff
      @. dealias[:,:,:,2] = (kx^2 + ky^2 + kz^2) > cutoff
      @. dealias[:,:,:,3] = (kx^2 + ky^2 + kz^2) > cutoff
    elseif Dealiastype == :cube
      @. dealias[:,:,:,1] = (kx^2 > cutoff) | (ky^2 > cutoff) | (kz^2 > cutoff)
      @. dealias[:,:,:,2] = (kx^2 > cutoff) | (ky^2 > cutoff) | (kz^2 > cutoff)
      @. dealias[:,:,:,3] = (kx^2 > cutoff) | (ky^2 > cutoff) | (kz^2 > cutoff)
    end

    return @par(new)(u,rhs,aux,nx,ny,nz,lx,ly,lz,ν,kx,ky,kz,p,ip,rm1x,rm1y,rm1z,rm2x,rm2y,rm2z,dealias)
  end

end

function Parameters(u::VectorField,nx::Integer,ny::Integer,nz::Integer,lx::Real,ly::Real,lz::Real,ν::Real,integrator::Symbol,deat::Symbol) 
  ncx = div(nx,2)+1
  lcs = ncx*ny*nz
  lcv = 3*lcs
  lrs = 2*lcs
  lrv = 2*lcv
  rx = 1:div(2ncx,3)
  kxr = SVector{length(rx),UInt32}(rx)
  ry = vcat(UInt32(1):(UInt32(div(ny,3))+1),UInt32(ny-div(ny,3)-1):UInt32(ny))
  kyr = SVector{length(ry),UInt32}(ry)
  rz = vcat(1:(div(nz,3)+1),(nz-div(nz,3)-1):nz)
  kzr = SVector{length(rz),UInt32}(rz)
  return Parameters{ncx,ny,nz,lcs,lcv,nx,lrs,lrv,integrator,deat,kxr,kyr,kzr}(u,nx,ny,nz,lx,ly,lz,ν)
end

abstract type @par(ScalarParameters) <: @par(AbstractParameters) end

struct @par(PassiveScalarParameters) <: @par(ScalarParameters)
  @GenericParameters
  ρ::PaddedArray{Float64,3,false}
  ps::Base.DFT.FFTW.rFFTWPlan{Float64,-1,true,3}
  α::Float64 #Difusitivity = ν/Pr  
  dρdz::Float64  
  ρrhs::PaddedArray{Float64,3,false}
  rrm1::Array{Complex128,3}
  rrm2::Array{Complex128,3}

  @par function @par(PassiveScalarParameters)(u::VectorField, nx::Integer, ny::Integer, nz::Integer, lx::Real, ly::Real, lz::Real, ν::Real, ρ::PaddedArray, α::Real,dρdz::Real) 
    
    rhs = similar(u)
    aux = similar(u)

    kx = SArray{Tuple{Nx,1,1}}(reshape(rfftfreq(nx,lx),(Nx,1,1)))
    ky = SArray{Tuple{1,Ny,1}}(reshape(fftfreq(ny,ly),(1,Ny,1)))
    kz = SArray{Tuple{1,1,Nz}}(reshape(fftfreq(nz,lz),(1,1,Nz)))
    
    aux = VectorField(PaddedArray(nx,ny,nz,3))
    p = plan_rfft!(aux,1:3,flags=FFTW.MEASURE)
    p.pinv = plan_irfft!(aux,1:3,flags=FFTW.MEASURE)
    ip = Base.DFT.ScaledPlan(FFTW.rFFTWPlan{Complex{Float64},FFTW.BACKWARD,false,4}(complex(aux), real(aux), 1:3, FFTW.MEASURE&FFTW.DESTROY_INPUT,FFTW.NO_TIMELIMIT),Base.DFT.normalization(Float64, size(real(aux)), 1:3))
    rm1x = Array{Complex128}(length(Kxr),length(Kyr),length(Kzr))
    rm1y = Array{Complex128}(length(Kxr),length(Kyr),length(Kzr))
    rm1z = Array{Complex128}(length(Kxr),length(Kyr),length(Kzr))
    rm2x = Array{Complex128}(length(Kxr),length(Kyr),length(Kzr))
    rm2y = Array{Complex128}(length(Kxr),length(Kyr),length(Kzr))
    rm2z = Array{Complex128}(length(Kxr),length(Kyr),length(Kzr))
    ρrhs = similar(ρ)
    ps = plan_rfft!(ρrhs,flags=FFTW.MEASURE)
    ps.pinv = plan_irfft!(ρrhs,flags=FFTW.MEASURE)
    rrm1 = Array{Complex128}(length(Kxr),length(Kyr),length(Kzr))
    rrm2 = Array{Complex128}(length(Kxr),length(Kyr),length(Kzr))

    dealias = BitArray(Nx,Ny,Nz,3)
    cutoff = (2kx[end]/3)^2

    if Dealiastype == :sphere
      @. dealias[:,:,:,1] = (kx^2 + ky^2 + kz^2) > cutoff
      @. dealias[:,:,:,2] = (kx^2 + ky^2 + kz^2) > cutoff
      @. dealias[:,:,:,3] = (kx^2 + ky^2 + kz^2) > cutoff
    elseif Dealiastype == :cube
      @. dealias[:,:,:,1] = (kx^2 > cutoff) | (ky^2 > cutoff) | (kz^2 > cutoff)
      @. dealias[:,:,:,2] = (kx^2 > cutoff) | (ky^2 > cutoff) | (kz^2 > cutoff)
      @. dealias[:,:,:,3] = (kx^2 > cutoff) | (ky^2 > cutoff) | (kz^2 > cutoff)
    end


    return @par(new)(u,rhs,aux,nx,ny,nz,lx,ly,lz,ν,kx,ky,kz,p,ip,rm1x,rm1y,rm1z,rm2x,rm2y,rm2z,dealias,ρ,ps,α,dρdz, ρrhs, rrm1,rrm2)
  end

end

function PassiveScalarParameters(u::VectorField,nx::Integer,ny::Integer,nz::Integer,lx::Real,ly::Real,lz::Real,ν::Real,ρ::PaddedArray, α::Real,dρdz::Real,integrator::Symbol,deat::Symbol) 
  ncx = div(nx,2)+1
  lcs = ncx*ny*nz
  lcv = 3*lcs
  lrs = 2*lcs
  lrv = 2*lcv
  rx = 1:div(2ncx,3)
  kxr = SVector{length(rx),UInt32}(rx)
  ry = vcat(UInt32(1):(UInt32(div(ny,3))+1),UInt32(ny-div(ny,3)-1):UInt32(ny))
  kyr = SVector{length(ry),UInt32}(ry)
  rz = vcat(1:(div(nz,3)+1),(nz-div(nz,3)-1):nz)
  kzr = SVector{length(rz),UInt32}(rz)
  return PassiveScalarParameters{ncx,ny,nz,lcs,lcv,nx,lrs,lrv,integrator,deat,kxr,kyr,kzr}(u,nx,ny,nz,lx,ly,lz,ν,ρ,α,dρdz)
end

struct @par(BoussinesqParameters) <: @par(ScalarParameters)
  @GenericParameters
  ρ::PaddedArray{Float64,3,false}
  ps::Base.DFT.FFTW.rFFTWPlan{Float64,-1,true,3}
  α::Float64 #Difusitivity = ν/Pr  
  dρdz::Float64 #This is actually dρsdz/ρ₀
  g::Float64
  ρrhs::PaddedArray{Float64,3,false}
  rrm1::Array{Complex128,3}
  rrm2::Array{Complex128,3}

  @par function @par(BoussinesqParameters)(u::VectorField, nx::Integer, ny::Integer, nz::Integer, lx::Real, ly::Real, lz::Real, ν::Real, ρ::PaddedArray, α::Real, dρdz::Real, g::Real)
    
    rhs = similar(u)
    aux = similar(u)

    kx = SArray{Tuple{Nx,1,1}}(reshape(rfftfreq(nx,lx),(Nx,1,1)))
    ky = SArray{Tuple{1,Ny,1}}(reshape(fftfreq(ny,ly),(1,Ny,1)))
    kz = SArray{Tuple{1,1,Nz}}(reshape(fftfreq(nz,lz),(1,1,Nz)))
    
    aux = VectorField(PaddedArray(nx,ny,nz,3))
    p = plan_rfft!(aux,1:3,flags=FFTW.MEASURE)
    p.pinv = plan_irfft!(aux,1:3,flags=FFTW.MEASURE)
    ip = Base.DFT.ScaledPlan(FFTW.rFFTWPlan{Complex{Float64},FFTW.BACKWARD,false,4}(complex(aux), real(aux), 1:3, FFTW.MEASURE&FFTW.DESTROY_INPUT,FFTW.NO_TIMELIMIT),Base.DFT.normalization(Float64, size(real(aux)), 1:3))
    rm1x = Array{Complex128}(length(Kxr),length(Kyr),length(Kzr))
    rm1y = Array{Complex128}(length(Kxr),length(Kyr),length(Kzr))
    rm1z = Array{Complex128}(length(Kxr),length(Kyr),length(Kzr))
    rm2x = Array{Complex128}(length(Kxr),length(Kyr),length(Kzr))
    rm2y = Array{Complex128}(length(Kxr),length(Kyr),length(Kzr))
    rm2z = Array{Complex128}(length(Kxr),length(Kyr),length(Kzr))
    ρrhs = similar(ρ)
    ps = plan_rfft!(ρrhs,flags=FFTW.MEASURE)
    ps.pinv = plan_irfft!(ρrhs,flags=FFTW.MEASURE)
    rrm1 = Array{Complex128}(length(Kxr),length(Kyr),length(Kzr))
    rrm2 = Array{Complex128}(length(Kxr),length(Kyr),length(Kzr))

    dealias = BitArray(Nx,Ny,Nz,3)
    cutoff = (2kx[end]/3)^2
    if Dealiastype == :sphere
      @. dealias[:,:,:,1] = (kx^2 + ky^2 + kz^2) > cutoff
      @. dealias[:,:,:,2] = (kx^2 + ky^2 + kz^2) > cutoff
      @. dealias[:,:,:,3] = (kx^2 + ky^2 + kz^2) > cutoff
    elseif Dealiastype == :cube
      @. dealias[:,:,:,1] = (kx^2 > cutoff) | (ky^2 > cutoff) | (kz^2 > cutoff)
      @. dealias[:,:,:,2] = (kx^2 > cutoff) | (ky^2 > cutoff) | (kz^2 > cutoff)
      @. dealias[:,:,:,3] = (kx^2 > cutoff) | (ky^2 > cutoff) | (kz^2 > cutoff)
    end

    return @par(new)(u,rhs,aux,nx,ny,nz,lx,ly,lz,ν,kx,ky,kz,p,ip,rm1x,rm1y,rm1z,rm2x,rm2x,rm2z,dealias,ρ,ps,α,dρdz,g, ρrhs, rrm1,rrm2)
  end

end

function BoussinesqParameters(u::VectorField,nx::Integer,ny::Integer,nz::Integer,lx::Real,ly::Real,lz::Real,ν::Real,ρ::PaddedArray, dρdz::Real,α::Real,g::Real,integrator::Symbol,deat::Symbol) 
  ncx = div(nx,2)+1
  lcs = ncx*ny*nz
  lcv = 3*lcs
  lrs = 2*lcs
  lrv = 2*lcv
  rx = 1:div(2ncx,3)
  kxr = SVector{length(rx),UInt32}(rx)
  ry = vcat(UInt32(1):(UInt32(div(ny,3))+1),UInt32(ny-div(ny,3)-1):UInt32(ny))
  kyr = SVector{length(ry),UInt32}(ry)
  rz = vcat(1:(div(nz,3)+1),(nz-div(nz,3)-1):nz)
  kzr = SVector{length(rz),UInt32}(rz)
  return BoussinesqParameters{ncx,ny,nz,lcs,lcv,nx,lrs,lrv,integrator,deat,kxr,kyr,kzr}(u,nx,ny,nz,lx,ly,lz,ν,ρ,α,dρdz,g)
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


  FFTW.set_num_threads(Threads.nthreads())
  
  haskey(d,:timeIntegrator) ? (integrator = Symbol(d[:timeIntegrator])) : (integrator = :Adams_Bashforth3rdO)
  integrator in (:Euller,:Adams_Bashforth3rdO) || error("Unkown time integration method in global file: $integrator")

  haskey(d,:dealias) ? (Dealiastype = Symbol(d[:dealias])) : (Dealiastype = :sphere)

  isfile("fftw_wisdom") && FFTW.import_wisdom("fftw_wisdom")

  if haskey(d,:model)
    model = Symbol(d[:model])
    if model == :PassiveScalar
      α = ν/parse(Float64,d[:Pr])
      dρdz = parse(Float64,d[:densityGradient])/parse(Float64,d[:referenceDensity])
      s = PassiveScalarParameters(u,nx,ny,nz,lx,ly,lz,ν,PaddedArray(zeros(nx,ny,nz)),α,dρdz,integrator,Dealiastype)
    elseif model == :Boussinesq 
      α = ν/parse(Float64,d[:Pr])
      dρdz = parse(Float64,d[:densityGradient])/parse(Float64,d[:referenceDensity])
      g = parse(Float64,d[:zAcceleration])
      s = BoussinesqParameters(u,nx,ny,nz,lx,ly,lz,ν,PaddedArray(zeros(nx,ny,nz)),α,dρdz,g,integrator,Dealiastype)
    else
      error("Unkown Model in global file: $model")
    end
  else
    s = Parameters(u,nx,ny,nz,lx,ly,lz,ν,integrator,Dealiastype)
  end

  FFTW.export_wisdom("fftw_wisdom")

  return s
end

parameters() = parameters(readglobal())

@par sizecomp(s::@par(AbstractParameters)) = (Kxr,Kyr,Kzr)