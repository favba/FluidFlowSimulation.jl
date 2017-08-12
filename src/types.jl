
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

@inline Base.real(V::VectorField) = real(V.data)
@inline Base.complex(V::VectorField) = complex(V.data) 
@inline InplaceRealFFTW.rawreal(V::VectorField) = rawreal(V.data)
Base.similar(V::VectorField) = VectorField(similar(V.data))

InplaceRealFFTW.rfft!(V::VectorField{T,A}) where {T,A} = rfft!(V,1:3) 
InplaceRealFFTW.irfft!(V::VectorField{T,A}) where {T,A} = irfft!(V,1:3) 

#------------------------------------------------------------------------------------------------------

abstract type AbstractParameters end

struct Parameters <: AbstractParameters
  nx::Int64
  ny::Int64
  nz::Int64
  lx::Float64
  ly::Float64
  lz::Float64
  ν::Float64
  kx::Vector{Float64}
  ky::Vector{Float64}
  kz::Vector{Float64}
  p::Base.DFT.FFTW.rFFTWPlan{Float64,-1,true,4}
  ip::Base.DFT.ScaledPlan{Complex{Float64},Base.DFT.FFTW.rFFTWPlan{Complex{Float64},1,false,4},Float64}
  
  function Parameters(nx::Int64,ny::Int64,nz::Int64,lx::Float64,ly::Float64,lz::Float64,ν::Float64)
    kx = rfftfreq(nx,lx)
    ky = fftfreq(ny,ly)    
    kz = fftfreq(nz,lz)    
    
    aux = VectorField(PaddedArray(nx,ny,nz,3))
    p = plan_rfft!(aux,1:3,flags=FFTW.MEASURE)
    p.pinv = plan_irfft!(aux,1:3,flags=FFTW.MEASURE)
    ip = Base.DFT.ScaledPlan(FFTW.rFFTWPlan{Complex{Float64},FFTW.BACKWARD,false,4}(complex(aux), real(aux), 1:3, FFTW.MEASURE&FFTW.DESTROY_INPUT,FFTW.NO_TIMELIMIT),Base.DFT.normalization(Float64, size(real(aux)), 1:3))
    return new(nx,ny,nz,lx,ly,lz,ν,kx,ky,kz,p,ip)
  end

end
