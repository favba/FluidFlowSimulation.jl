
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