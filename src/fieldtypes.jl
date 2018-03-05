struct StaticView{Size,L,AT,T,N,Ind} <: DenseArray{T,N}
  data::AT
end

function StaticView(a::DenseArray{T,N},n) where {T,N}
  sa = size(a)
  newsize = sa[1:(N-1)]
  newlength = prod(newsize)
  return StaticView{Tuple{newsize...},newlength,typeof(a),T,N-1,n}(a)
end

Base.@pure Base.length(a::Union{StaticView{S,L,A,T,N,I},Type{StaticView{S,L,A,T,N,I}}}) where {S,L,A,T,N,I} = Int(L)
Base.@pure Base.size(a::Union{StaticView{S,L,A,T,N,I},Type{StaticView{S,L,A,T,N,I}}}) where {S,L,A,T,N,I} = tuple(S.parameters...)
Base.IndexStyle(::Type{<:StaticView}) = IndexLinear()

@inline @generated function Base.getindex(a::StaticView{S,L,A,T,N,I},i2::Integer) where {S,L,A,T,N,I} 
  s = (I-1)*length(a)
  quote
    i = i2 + $s
    d = a.data
    @boundscheck checkbounds(d,i)
    r = @inbounds d[i]
    return r
  end
end

@inline @generated function Base.getindex(a::StaticView{S,L,A,T,N,I},In::Vararg{Integer,N}) where {S,L,A,T,N,I} 
  s = I
  p = Expr(:tuple)
  for i=1:N
    push!(p.args,:(In[$i]))
  end
  push!(p.args,:($I))
  quote
    d = a.data
    @boundscheck checkbounds(d,$p...)
    r = @inbounds getindex(d,$p...)
    return r
  end
end

@inline @generated function Base.setindex!(a::StaticView{S,L,A,T,N,I}, x,i2::Integer) where {S,L,A,T,N,I} 
  s = (I-1)*length(a)
  quote
    i = i2 + $s
    d = a.data
    @boundscheck checkbounds(d,i)
    @inbounds setindex!(d,x,i)
  end
end

@inline @generated function Base.setindex!(a::StaticView{S,L,A,T,N,I}, x, In::Vararg{Integer,N}) where {S,L,A,T,N,I} 
  s = I
  p = Expr(:tuple)
  for i=1:N
    push!(p.args,:(In[$i]))
  end
  push!(p.args,:($I))
  quote
    d = a.data
    @boundscheck checkbounds(d,$p...)
    @inbounds setindex!(d, x, $p...)
  end
end

@generated function Base.unsafe_convert(::Type{Ptr{T}},a::StaticView{S,L,A,T,N,I}) where {S,L,A,T,N,I}
  s = (I-1)*length(a) + 1
  quote
    return pointer(a.data,$s)
  end
end

Base.complex(a::AbstractPaddedArray) = InplaceRealFFT.unsafe_complex_view(a)
struct VectorField{A<:AbstractPaddedArray{Float64, 4}} <: AbstractPaddedArray{Float64,4}
  data::A
  cx::Array{Complex128,3}
  cy::Array{Complex128,3}
  cz::Array{Complex128,3}
  rx::Array{Float64,3}
  ry::Array{Float64,3}
  rz::Array{Float64,3}

  function VectorField{A}(data::A) where {A<:AbstractPaddedArray{Float64, 4}}
    cdims = size(data)
    cnx, cny, cnz, _ = cdims
    cx = unsafe_wrap(Array{Complex128,3},pointer(complex(data)),(cnx,cny,cnz))
    cy = unsafe_wrap(Array{Complex128,3},pointer(complex(data),sub2ind(cdims,1,1,1,2)),(cnx,cny,cnz))
    cz = unsafe_wrap(Array{Complex128,3},pointer(complex(data),sub2ind(cdims,1,1,1,3)),(cnx,cny,cnz))

    rnx,rny,rnz,_ = size(parent(real(data)))
    rdims = (rnx,rny,rnz)
    #rx = reinterpret(Float64,cx,rdims)
    rx = unsafe_wrap(Array{Float64,3},reinterpret(Ptr{Float64},pointer(cx)),rdims)
    #ry = reinterpret(Float64,cy,rdims)
    ry = unsafe_wrap(Array{Float64,3},reinterpret(Ptr{Float64},pointer(cy)),rdims)
    #rz = reinterpret(Float64,cz,rdims)
    rz = unsafe_wrap(Array{Float64,3},reinterpret(Ptr{Float64},pointer(cz)),rdims)

    return new{A}(data,cx,cy,cz,rx,ry,rz)
  end
end

VectorField(data::AbstractPaddedArray{Float64,4}) = VectorField{typeof(data)}(data)

function VectorField(ux::AbstractString,uy::AbstractString,uz::AbstractString,nx::Integer,ny::Integer,nz::Integer)
  field = VectorField(PaddedArray(nx,ny,nz,3))
  read!(ux,field.rx)
  read!(uy,field.ry)
  read!(uz,field.rz)
  return field
end

@inline Base.real(V::VectorField) = real(V.data)
@inline InplaceRealFFT.unsafe_complex_view(V::VectorField) = InplaceRealFFT.unsafe_complex_view(V.data) 
Base.similar(V::VectorField{A}) where {A} = VectorField{A}(similar(V.data))
Base.copy(V::VectorField{A}) where {A} = VectorField{A}(copy(V.data))

InplaceRealFFT.rfft!(V::VectorField{A}) where {A} = rfft!(V,1:3) 
InplaceRealFFT.irfft!(V::VectorField{A}) where {A} = irfft!(V,1:3) 


struct SymmetricTracelessTensor <: AbstractPaddedArray{Float64,4}
  data::PaddedArray{Float64,4,false}
  cxx::Array{Complex128,3}
  cxy::Array{Complex128,3}
  cxz::Array{Complex128,3}
  cyy::Array{Complex128,3}
  cyz::Array{Complex128,3}
  rxx::Array{Float64,3}
  rxy::Array{Float64,3}
  rxz::Array{Float64,3}
  ryy::Array{Float64,3}
  ryz::Array{Float64,3}


  function SymmetricTracelessTensor(data::A) where {A<:AbstractPaddedArray{Float64, 4}}
    cdims = size(data)
    cnx, cny, cnz, nfields = cdims

    @assert nfields == 5

    cxx = unsafe_wrap(Array{Complex128,3},pointer(complex(data)),(cnx,cny,cnz))
    cxy = unsafe_wrap(Array{Complex128,3},pointer(complex(data),sub2ind(cdims,1,1,1,2)),(cnx,cny,cnz))
    cxz = unsafe_wrap(Array{Complex128,3},pointer(complex(data),sub2ind(cdims,1,1,1,3)),(cnx,cny,cnz))
    cyy = unsafe_wrap(Array{Complex128,3},pointer(complex(data),sub2ind(cdims,1,1,1,4)),(cnx,cny,cnz))
    cyz = unsafe_wrap(Array{Complex128,3},pointer(complex(data),sub2ind(cdims,1,1,1,5)),(cnx,cny,cnz))

    rnx,rny,rnz,_ = size(parent(real(data)))
    rdims = (rnx,rny,rnz)

    rxx = unsafe_wrap(Array{Float64,3},reinterpret(Ptr{Float64},pointer(cxx)),rdims)
    rxy = unsafe_wrap(Array{Float64,3},reinterpret(Ptr{Float64},pointer(cxy)),rdims)
    rxz = unsafe_wrap(Array{Float64,3},reinterpret(Ptr{Float64},pointer(cxz)),rdims)
    ryy = unsafe_wrap(Array{Float64,3},reinterpret(Ptr{Float64},pointer(cyy)),rdims)
    ryz = unsafe_wrap(Array{Float64,3},reinterpret(Ptr{Float64},pointer(cyz)),rdims)

    return new(data,cxx,cxy,cxz,cyy,cyz,rxx,rxy,rxz,ryy,ryz)
  end
end

SymmetricTracelessTensor(nx::Integer,ny::Integer,nz::Integer) = SymmetricTracelessTensor(PaddedArray(zeros(nx,ny,nz,5)))
SymmetricTracelessTensor(dim::NTuple{3,Integer}) = SymmetricTracelessTensor(dim...)

@inline Base.real(T::SymmetricTracelessTensor) = real(T.data)
@inline InplaceRealFFT.unsafe_complex_view(T::SymmetricTracelessTensor) = InplaceRealFFT.unsafe_complex_view(T.data) 
