
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

abstract type AbstractParameters{Nx,Ny,Nz,Lcs,Lcv,Nrx,Lrs,Lrv} end

Integrator(x::AbstractParameters) = :Adams_Bashforth3rdO

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
  rm1::Array{Complex128,4}
  rm2::Array{Complex128,4}
end

struct Parameters{Nx,Ny,Nz,Lcs,Lcv,Nrx,Lrs,Lrv} <: AbstractParameters{Nx,Ny,Nz,Lcs,Lcv,Nrx,Lrs,Lrv}
  @GenericParameters
  
  function Parameters{Nx,Ny,Nz,Lcs,Lcv,Nrx,Lrs,Lrv}(u::VectorField{PaddedArray{Float64,4,false}},nx::Int64,ny::Int64,nz::Int64,lx::Float64,ly::Float64,lz::Float64,ν::Float64) where {Nx,Ny,Nz,Lcs,Lcv,Nrx,Lrs,Lrv}
    
    rhs = similar(u)
    aux = similar(u)

    kx = SArray{Tuple{Nx,1,1}}(reshape(rfftfreq(nx,lx),(Nx,1,1)))
    ky = SArray{Tuple{1,Ny,1}}(reshape(fftfreq(ny,ly),(1,Ny,1)))
    kz = SArray{Tuple{1,1,Nz}}(reshape(fftfreq(nz,lz),(1,1,Nz)))
    
    aux = VectorField(PaddedArray(nx,ny,nz,3))
    p = plan_rfft!(aux,1:3,flags=FFTW.MEASURE)
    p.pinv = plan_irfft!(aux,1:3,flags=FFTW.MEASURE)
    ip = Base.DFT.ScaledPlan(FFTW.rFFTWPlan{Complex{Float64},FFTW.BACKWARD,false,4}(complex(aux), real(aux), 1:3, FFTW.MEASURE&FFTW.DESTROY_INPUT,FFTW.NO_TIMELIMIT),Base.DFT.normalization(Float64, size(real(aux)), 1:3))
    rm1 = Array{Complex128}((Nx,Ny,Nz,4))
    rm2 = Array{Complex128}((Nx,Ny,Nz,4))
    return new{Nx,Ny,Nz,Lcs,Lcv,Nrx,Lrs,Lrv}(u,rhs,aux,nx,ny,nz,lx,ly,lz,ν,kx,ky,kz,p,ip,rm1,rm2)
  end

end

function Parameters(u::VectorField,nx::Integer,ny::Integer,nz::Integer,lx::Real,ly::Real,lz::Real,ν::Real) 
  ncx = div(nx,2)+1
  lcs = ncx*ny*nz
  lcv = 3*lcs
  nrx = 2*ncx
  lrs = 2*lcs
  lrv = 2*lcv
  return Parameters{ncx,ny,nz,lcs,lcv,nrx,lrs,lrv}(u,nx,ny,nz,lx,ly,lz,ν)
end

function Parameters()
  par = readglobal()
  return Parameters(par)
end

function Parameters(par::Dict)
  nx = parse(Int,par["nx"])
  ny = parse(Int,par["ny"])
  nz = parse(Int,par["nz"])
  lx = parse(Float64,par["xDomainSize"])
  ly = parse(Float64,par["yDomainSize"])
  lz = parse(Float64,par["zDomainSize"])
  ν = parse(Float64,par["kinematicViscosity"])
  u = VectorField("u1.0","u2.0","u3.0",nx,ny,nz)
  return Parameters(u,nx,ny,nz,lx,ly,lz,ν)
end

abstract type ScalarParameters{Nx,Ny,Nz,Lcs,Lcv,Nrx,Lrs,Lrv} <: AbstractParameters{Nx,Ny,Nz,Lcs,Lcv,Nrx,Lrs,Lrv} end

struct PassiveScalarParameters{Nx,Ny,Nz,Lcs,Lcv,Nrx,Lrs,Lrv} <: ScalarParameters{Nx,Ny,Nz,Lcs,Lcv,Nrx,Lrs,Lrv}
  @GenericParameters
  ρ::PaddedArray{Float64,3,false}
  ps::Base.DFT.FFTW.rFFTWPlan{Float64,-1,true,3}
  α::Float64 #Difusitivity = ν/Pr  
  dρdz::Float64  
  ρrhs::PaddedArray{Float64,3,false}
  rrm1::Array{Complex128,3}
  rrm2::Array{Complex128,3}

  function PassiveScalarParameters{Nx,Ny,Nz,Lcs,Lcv,Nrx,Lrs,Lrv}(u::VectorField, nx::Integer, ny::Integer, nz::Integer, lx::Real, ly::Real, lz::Real, ν::Real, ρ::PaddedArray, α::Real,dρdz::Real) where {Nx,Ny,Nz,Lcs,Lcv,Nrx,Lrs,Lrv}
    
    rhs = similar(u)
    aux = similar(u)

    kx = SArray{Tuple{Nx,1,1}}(reshape(rfftfreq(nx,lx),(Nx,1,1)))
    ky = SArray{Tuple{1,Ny,1}}(reshape(fftfreq(ny,ly),(1,Ny,1)))
    kz = SArray{Tuple{1,1,Nz}}(reshape(fftfreq(nz,lz),(1,1,Nz)))
    
    aux = VectorField(PaddedArray(nx,ny,nz,3))
    p = plan_rfft!(aux,1:3,flags=FFTW.MEASURE)
    p.pinv = plan_irfft!(aux,1:3,flags=FFTW.MEASURE)
    ip = Base.DFT.ScaledPlan(FFTW.rFFTWPlan{Complex{Float64},FFTW.BACKWARD,false,4}(complex(aux), real(aux), 1:3, FFTW.MEASURE&FFTW.DESTROY_INPUT,FFTW.NO_TIMELIMIT),Base.DFT.normalization(Float64, size(real(aux)), 1:3))
    rm1 = Array{Complex128}((Nx,Ny,Nz,4))
    rm2 = Array{Complex128}((Nx,Ny,Nz,4))
    ρrhs = similar(ρ)
    ps = plan_rfft!(ρrhs,flags=FFTW.MEASURE)
    ps.pinv = plan_irfft!(ρrhs,flags=FFTW.MEASURE)
    rrm1 = Array{Complex128}((Nx,Ny,Nz))
    rrm2 = Array{Complex128}((Nx,Ny,Nz))

    return new{Nx,Ny,Nz,Lcs,Lcv,Nrx,Lrs,Lrv}(u,rhs,aux,nx,ny,nz,lx,ly,lz,ν,kx,ky,kz,p,ip,rm1,rm2,ρ,ps,α,dρdz, ρrhs, rrm1,rrm2)
  end

end

function PassiveScalarParameters(u::VectorField,nx::Integer,ny::Integer,nz::Integer,lx::Real,ly::Real,lz::Real,ν::Real,ρ::PaddedArray, α::Real,dρdz::Real) 
  ncx = div(nx,2)+1
  lcs = ncx*ny*nz
  lcv = 3*lcs
  nrx = 2*ncx
  lrs = 2*lcs
  lrv = 2*lcv
  return PassiveScalarParameters{ncx,ny,nz,lcs,lcv,nrx,lrs,lrv}(u,nx,ny,nz,lx,ly,lz,ν,ρ,α,dρdz)
end

struct BoussinesqParameters{Nx,Ny,Nz,Lcs,Lcv,Nrx,Lrs,Lrv} <: ScalarParameters{Nx,Ny,Nz,Lcs,Lcv,Nrx,Lrs,Lrv}
  @GenericParameters
  ρ::PaddedArray{Float64,3,false}
  ps::Base.DFT.FFTW.rFFTWPlan{Float64,-1,true,3}
  α::Float64 #Difusitivity = ν/Pr  
  dρdz::Float64 #This is actually dρsdz/ρ₀
  g::Float64
  ρrhs::PaddedArray{Float64,3,false}
  rrm1::Array{Complex128,3}
  rrm2::Array{Complex128,3}

  function BoussinesqParameters{Nx,Ny,Nz,Lcs,Lcv,Nrx,Lrs,Lrv}(u::VectorField, nx::Integer, ny::Integer, nz::Integer, lx::Real, ly::Real, lz::Real, ν::Real, ρ::PaddedArray, α::Real, dρdz::Real, g::Real) where {Nx,Ny,Nz,Lcs,Lcv,Nrx,Lrs,Lrv}
    
    rhs = similar(u)
    aux = similar(u)

    kx = SArray{Tuple{Nx,1,1}}(reshape(rfftfreq(nx,lx),(Nx,1,1)))
    ky = SArray{Tuple{1,Ny,1}}(reshape(fftfreq(ny,ly),(1,Ny,1)))
    kz = SArray{Tuple{1,1,Nz}}(reshape(fftfreq(nz,lz),(1,1,Nz)))
    
    aux = VectorField(PaddedArray(nx,ny,nz,3))
    p = plan_rfft!(aux,1:3,flags=FFTW.MEASURE)
    p.pinv = plan_irfft!(aux,1:3,flags=FFTW.MEASURE)
    ip = Base.DFT.ScaledPlan(FFTW.rFFTWPlan{Complex{Float64},FFTW.BACKWARD,false,4}(complex(aux), real(aux), 1:3, FFTW.MEASURE&FFTW.DESTROY_INPUT,FFTW.NO_TIMELIMIT),Base.DFT.normalization(Float64, size(real(aux)), 1:3))
    rm1 = Array{Complex128}((Nx,Ny,Nz,4))
    rm2 = Array{Complex128}((Nx,Ny,Nz,4))
    ρrhs = similar(ρ)
    ps = plan_rfft!(ρrhs,flags=FFTW.MEASURE)
    ps.pinv = plan_irfft!(ρrhs,flags=FFTW.MEASURE)
    rrm1 = Array{Complex128}((Nx,Ny,Nz))
    rrm2 = Array{Complex128}((Nx,Ny,Nz))

    return new{Nx,Ny,Nz,Lcs,Lcv,Nrx,Lrs,Lrv}(u,rhs,aux,nx,ny,nz,lx,ly,lz,ν,kx,ky,kz,p,ip,rm1,rm2,ρ,ps,α,dρdz,g, ρrhs, rrm1,rrm2)
  end

end

function BoussinesqParameters(u::VectorField,nx::Integer,ny::Integer,nz::Integer,lx::Real,ly::Real,lz::Real,ν::Real,ρ::PaddedArray, dρdz::Real,α::Real,g::Real) 
  ncx = div(nx,2)+1
  lcs = ncx*ny*nz
  lcv = 3*lcs
  nrx = 2*ncx
  lrs = 2*lcs
  lrv = 2*lcv
  return BoussinesqParameters{ncx,ny,nz,lcs,lcv,nrx,lrs,lrv}(u,nx,ny,nz,lx,ly,lz,ν,ρ,α,dρdz,g)
end

function parameters(d::Dict)

  nx = parse(Int,d["nx"])
  ny = parse(Int,d["ny"])
  nz = parse(Int,d["nz"])
  lx = parse(Float64,d["xDomainSize"])
  ly = parse(Float64,d["yDomainSize"])
  lz = parse(Float64,d["zDomainSize"])
  ν = parse(Float64,d["kinematicViscosity"])
  u = VectorField("u1.0","u2.0","u3.0",nx,ny,nz)

  if haskey(d,"model")
    model = Symbol(d["model"])
    if model == :PassiveScalar
      α = ν/parse(Float64,d["Pr"])
      dρdz = parse(Float64,d["densityGradient"])/parse(Float64,d["referenceDensity"])
      s = PassiveScalarParameters(u,nx,ny,nz,lx,ly,lz,ν,PaddedArray(zeros(nx,ny,nz)),α,dρdz)
    elseif model == :Boussinesq 
      α = ν/parse(Float64,d["Pr"])
      dρdz = parse(Float64,d["densityGradient"])/parse(Float64,d["referenceDensity"])
      g = parse(Float64,d["zAcceleration"])
      s = BoussinesqParameters(u,nx,ny,nz,lx,ly,lz,ν,PaddedArray(zeros(nx,ny,nz)),α,dρdz,g)
    else
      error("Unkown Model in global file")
    end
  else
    s = Parameters(d)
  end
  return s
end

parameters() = parameters(readglobal())