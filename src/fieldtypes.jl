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

Base.complex(a::AbstractPaddedArray) = InplaceRealFFT.complex_view(a)
struct VectorField{A<:AbstractPaddedArray{Float64, 4},SR,SC,LR,LC} <: AbstractPaddedArray{Float64,4}
  data::A
  cx::StaticView{SC,LC,A,Complex{Float64},3,1}
  cy::StaticView{SC,LC,A,Complex{Float64},3,2}
  cz::StaticView{SC,LC,A,Complex{Float64},3,3}
  rx::StaticView{SR,LR,Array{Float64,4},Float64,3,1}
  ry::StaticView{SR,LR,Array{Float64,4},Float64,3,2}
  rz::StaticView{SR,LR,Array{Float64,4},Float64,3,3}

  function VectorField{A}(data::A) where {A<:AbstractPaddedArray{Float64, 4}}

    cx = StaticView(data,1)
    cy = StaticView(data,2)
    cz = StaticView(data,3)

    rx = StaticView(parent(real(data)),1)
    ry = StaticView(parent(real(data)),2)
    rz = StaticView(parent(real(data)),3)

    return new{A,Tuple{size(rx)...},Tuple{size(cx)...},length(rx),length(cx)}(data,cx,cy,cz,rx,ry,rz)
  end
end

VectorField(data::AbstractPaddedArray{Float64,4}) = VectorField{typeof(data)}(data)

function VectorField(ux::AbstractString,uy::AbstractString,uz::AbstractString,nx::Integer,ny::Integer,nz::Integer)
  field = VectorField(PaddedArray(nx,ny,nz,3))
  nb = sizeof(Float64)*2*(div(nx,2)+1)*ny*nz
  open(ux) do f; Base.unsafe_read(f,pointer(field.rx),nb); end
  open(uy) do f; Base.unsafe_read(f,pointer(field.ry),nb); end
  open(uz) do f; Base.unsafe_read(f,pointer(field.rz),nb); end
  return field
end

@inline Base.real(V::VectorField) = real(V.data)
@inline InplaceRealFFT.complex_view(V::VectorField) = InplaceRealFFT.complex_view(V.data) 
Base.similar(V::VectorField{A}) where {A} = VectorField{A}(similar(V.data))
Base.copy(V::VectorField{A}) where {A} = VectorField{A}(copy(V.data))

InplaceRealFFT.rfft!(V::VectorField{A}) where {A} = rfft!(V,1:3) 
InplaceRealFFT.irfft!(V::VectorField{A}) where {A} = irfft!(V,1:3) 


struct SymmetricTracelessTensor{SR,SC,LR,LC} <: AbstractPaddedArray{Float64,4}
  data::PaddedArray{Float64,4,3,false}
  cxx::StaticView{SC,LC,PaddedArray{Float64,4,3,false},Complex{Float64},3,1}
  cxy::StaticView{SC,LC,PaddedArray{Float64,4,3,false},Complex{Float64},3,2}
  cxz::StaticView{SC,LC,PaddedArray{Float64,4,3,false},Complex{Float64},3,3}
  cyy::StaticView{SC,LC,PaddedArray{Float64,4,3,false},Complex{Float64},3,4}
  cyz::StaticView{SC,LC,PaddedArray{Float64,4,3,false},Complex{Float64},3,5}
  rxx::StaticView{SR,LR,Array{Float64,4},Float64,3,1}
  rxy::StaticView{SR,LR,Array{Float64,4},Float64,3,2}
  rxz::StaticView{SR,LR,Array{Float64,4},Float64,3,3}
  ryy::StaticView{SR,LR,Array{Float64,4},Float64,3,4}
  ryz::StaticView{SR,LR,Array{Float64,4},Float64,3,5}

  function SymmetricTracelessTensor(data::A) where {A<:AbstractPaddedArray{Float64, 4}}
    cdims = size(data)
    cnx, cny, cnz, nfields = cdims

    @assert nfields == 5

    cxx = StaticView(data,1)
    cxy = StaticView(data,2)
    cxz = StaticView(data,3)
    cyy = StaticView(data,4)
    cyz = StaticView(data,5)

    rxx = StaticView(parent(real(data)),1)
    rxy = StaticView(parent(real(data)),2)
    rxz = StaticView(parent(real(data)),3)
    ryy = StaticView(parent(real(data)),4)
    ryz = StaticView(parent(real(data)),5)

    return new{Tuple{size(rxx)...},Tuple{size(cxx)...},length(rxx),length(cxx)}(data,cxx,cxy,cxz,cyy,cyz,rxx,rxy,rxz,ryy,ryz)
  end
end

SymmetricTracelessTensor(nx::Integer,ny::Integer,nz::Integer) = SymmetricTracelessTensor(PaddedArray(zeros(nx,ny,nz,5)))
SymmetricTracelessTensor(dim::NTuple{3,Integer}) = SymmetricTracelessTensor(dim...)

@inline Base.real(T::SymmetricTracelessTensor) = real(T.data)
@inline InplaceRealFFT.complex_view(T::SymmetricTracelessTensor) = InplaceRealFFT.complex_view(T.data) 
