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
