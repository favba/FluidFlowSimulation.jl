
struct VectorField{T, N1, A<:AbstractPaddedArray{T, N1}, N2, C<:AbstractArray{Complex{T}, N2}, R<:AbstractArray{T, N2}} <: AbstractPaddedArray{T,N1}
  data::A
  cx::C
  cy::C
  cz::C
  rx::R
  ry::R
  rz::R

  function VectorField(data::PaddedArray{T,N}) where {T,N}
    N2 = N-1
    cx = view(complex(data),(Colon() for i=1:N2)...,1)
    cy = view(complex(data),(Colon() for i=1:N2)...,2)
    cz = view(complex(data),(Colon() for i=1:N2)...,3)
    rx = view(rawreal(data),(Colon() for i=1:N2)...,1)
    ry = view(rawreal(data),(Colon() for i=1:N2)...,2)
    rz = view(rawreal(data),(Colon() for i=1:N2)...,3)
    C = typeof(cx)
    R = typeof(rx)
    return new{T,N,PaddedArray{T,N},N2,C,R}(data,cx,cy,cz,rx,ry,rz)
  end
end

@inline real(V::VectorField) = real(V.data)
@inline complex(V::VectorField) = complex(V.data) 
@inline InplaceRealFFTW.rawreal(V::VectorField) = rawreal(V.data)
Base.similar(V::VectorField) = VectorField(similar(V.data))

InplaceRealFFTW.rfft!(V::VectorField{T,N1,A,N2,C,R}) where {T,N1,A,N2,C,R} = rfft!(V,1:N2) 
InplaceRealFFTW.irfft!(V::VectorField{T,N1,A,N2,C,R}) where {T,N1,A,N2,C,R} = irfft!(V,1:N2) 