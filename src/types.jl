
struct VectorField{T, A<:AbstractPaddedArray{T, 4, false}} <: AbstractPaddedArray{T,4,false}
  data::A
  cx::Array{Complex{T},3}
  cy::Array{Complex{T},3}
  cz::Array{Complex{T},3}
  rx::Array{T,3}
  ry::Array{T,3}
  rz::Array{T,3}

  function VectorField{T,A}(data::A) where {T,A<:AbstractPaddedArray{T, 4,false}}
    cdims = size(data)
    cnx, cny, cnz, _ = cdims
    cx = unsafe_wrap(Array{Complex{T},3},pointer(complex(data)),(cnx,cny,cnz))
    cy = unsafe_wrap(Array{Complex{T},3},pointer(complex(data),sub2ind(cdims,1,1,1,2)),(cnx,cny,cnz))
    cz = unsafe_wrap(Array{Complex{T},3},pointer(complex(data),sub2ind(cdims,1,1,1,3)),(cnx,cny,cnz))

    rnx,rny,rnz,_ = size(rawreal(data))
    rdims = (rnx,rny,rnz)
    rx = reinterpret(T,cx,rdims)
    ry = reinterpret(T,cy,rdims)
    rz = reinterpret(T,cz,rdims)
    return new{T,A}(data,cx,cy,cz,rx,ry,rz)
  end
end

VectorField(data::AbstractPaddedArray{T,4,false}) where {T} = VectorField{T,typeof(data)}(data)

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
Base.similar(V::VectorField{T,A}) where {T,A} = VectorField{T,A}(similar(V.data))
Base.copy(V::VectorField{T,A}) where {T,A} = VectorField{T,A}(copy(V.data))

InplaceRealFFTW.rfft!(V::VectorField{T,A}) where {T,A} = rfft!(V,1:3) 
InplaceRealFFTW.irfft!(V::VectorField{T,A}) where {T,A} = irfft!(V,1:3) 

#------------------------------------------------------------------------------------------------------

abstract type AbstractParameters{Nx,Ny,Nz} end

struct Parameters{Nx,Ny,Nz} <: AbstractParameters{Nx,Ny,Nz}
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
  
  function Parameters{Nx,Ny,Nz}(nx::Int64,ny::Int64,nz::Int64,lx::Float64,ly::Float64,lz::Float64,ν::Float64) where {Nx,Ny,Nz}

    kx = SArray{Tuple{Nx,1,1}}(reshape(rfftfreq(nx,lx),(Nx,1,1)))
    ky = SArray{Tuple{1,Ny,1}}(reshape(fftfreq(ny,ly),(1,Ny,1)))
    kz = SArray{Tuple{1,1,Nz}}(reshape(fftfreq(nz,lz),(1,1,Nz)))
    
    aux = VectorField(PaddedArray(nx,ny,nz,3))
    p = plan_rfft!(aux,1:3,flags=FFTW.MEASURE)
    p.pinv = plan_irfft!(aux,1:3,flags=FFTW.MEASURE)
    ip = Base.DFT.ScaledPlan(FFTW.rFFTWPlan{Complex{Float64},FFTW.BACKWARD,false,4}(complex(aux), real(aux), 1:3, FFTW.MEASURE&FFTW.DESTROY_INPUT,FFTW.NO_TIMELIMIT),Base.DFT.normalization(Float64, size(real(aux)), 1:3))
    return new{Nx,Ny,Nz}(nx,ny,nz,lx,ly,lz,ν,kx,ky,kz,p,ip)
  end

end

Parameters(nx::Integer,ny::Integer,nz::Integer,lx::Real,ly::Real,lz::Real,ν::Real) = Parameters{div(nx,2)+1,ny,nz}(nx,ny,nz,lx,ly,lz,ν)