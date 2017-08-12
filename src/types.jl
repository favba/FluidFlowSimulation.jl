
struct VectorField{T, N, A<:AbstractPaddedArray{T, N, false}} <: AbstractPaddedArray{T,N,false}
  data::A
  cx::SubArray{Complex{T},3,Array{Complex{T},4},Tuple{Base.Slice{Base.OneTo{Int64}},Base.Slice{Base.OneTo{Int64}},Base.Slice{Base.OneTo{Int64}},Int64},true}
  cy::SubArray{Complex{T},3,Array{Complex{T},4},Tuple{Base.Slice{Base.OneTo{Int64}},Base.Slice{Base.OneTo{Int64}},Base.Slice{Base.OneTo{Int64}},Int64},true}
  cz::SubArray{Complex{T},3,Array{Complex{T},4},Tuple{Base.Slice{Base.OneTo{Int64}},Base.Slice{Base.OneTo{Int64}},Base.Slice{Base.OneTo{Int64}},Int64},true}
  rx::SubArray{T,3,Array{T,4},Tuple{Base.Slice{Base.OneTo{Int64}},Base.Slice{Base.OneTo{Int64}},Base.Slice{Base.OneTo{Int64}},Int64},true}
  ry::SubArray{T,3,Array{T,4},Tuple{Base.Slice{Base.OneTo{Int64}},Base.Slice{Base.OneTo{Int64}},Base.Slice{Base.OneTo{Int64}},Int64},true}
  rz::SubArray{T,3,Array{T,4},Tuple{Base.Slice{Base.OneTo{Int64}},Base.Slice{Base.OneTo{Int64}},Base.Slice{Base.OneTo{Int64}},Int64},true}

  function VectorField{T,N,A}(data::A) where {T,N,A<:AbstractPaddedArray{T, N,false}}
    cx = view(complex(data),(Colon() for i=1:3)...,1)
    cy = view(complex(data),(Colon() for i=1:3)...,2)
    cz = view(complex(data),(Colon() for i=1:3)...,3)
    rx = view(rawreal(data),(Colon() for i=1:3)...,1)
    ry = view(rawreal(data),(Colon() for i=1:3)...,2)
    rz = view(rawreal(data),(Colon() for i=1:3)...,3)
    return new{T,N,A}(data,cx,cy,cz,rx,ry,rz)
  end
end

VectorField(data::AbstractPaddedArray{T,N,false}) where {T,N} = VectorField{T,N,typeof(data)}(data)

@inline Base.real(V::VectorField) = real(V.data)
@inline Base.complex(V::VectorField) = complex(V.data) 
@inline InplaceRealFFTW.rawreal(V::VectorField) = rawreal(V.data)
Base.similar(V::VectorField) = VectorField(similar(V.data))

InplaceRealFFTW.rfft!(V::VectorField{T,N,A}) where {T,N,A} = rfft!(V,1:3) 
InplaceRealFFTW.irfft!(V::VectorField{T,N,A}) where {T,N,A} = irfft!(V,1:3) 

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
