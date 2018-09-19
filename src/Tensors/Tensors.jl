module Tensors
import LinearAlgebra

export Vec, SymTrTen

abstract type AbstractVec{T} <: DenseArray{T,1} end
abstract type AbstractTen{T} <: AbstractArray{T,2} end

#----------------------------------------------- Vectors --------------------------------------------------

@inline Base.@propagate_inbounds Base.getindex(a::AbstractVec,I::Integer) = getfield(a,I)
Base.IndexStyle(a::Type{<:AbstractVec}) = Base.IndexLinear()

@inline xpos(a::AbstractVec) = a.x
@inline ypos(a::AbstractVec) = a.y
@inline zpos(a::AbstractVec) = a.z

#pos of a node returns a tuple with the positions
pos(a::AbstractVec) = (xpos(a),ypos(a),zpos(a))

LinearAlgebra.norm(a::AbstractVec) = sqrt(muladd(xpos(a), xpos(a), muladd(ypos(a), ypos(a), zpos(a)^2)))
distance(a::AbstractVec,b::AbstractVec) = sqrt((xpos(b)-xpos(a))^2 + (ypos(b)-ypos(a))^2 + (zpos(b)-zpos(a))^2)

Base.:+(a::AbstractVec,b::AbstractVec) = Vec(xpos(a)+xpos(b), ypos(a)+ypos(b), zpos(a)+zpos(b))
Base.:-(a::AbstractVec,b::AbstractVec) = Vec(xpos(a)-xpos(b), ypos(a)-ypos(b), zpos(a)-zpos(b))
Base.:*(a::Number,b::AbstractVec) = Vec(a*xpos(b), a*ypos(b), a*zpos(b))
Base.:*(b::AbstractVec,a::Number) = Vec(a*xpos(b), a*ypos(b), a*zpos(b))
Base.:/(b::AbstractVec,a::Number) = Vec(xpos(b)/a, ypos(b)/a, zpos(b)/a)

LinearAlgebra.dot(a::AbstractVec,b::AbstractVec) = muladd(xpos(a), xpos(b), muladd(ypos(a), ypos(b), zpos(a)*zpos(b)))

LinearAlgebra.cross(a::AbstractVec,b::AbstractVec) = Vec(ypos(a)*zpos(b) - zpos(a)*ypos(b), zpos(a)*xpos(b) - xpos(a)*zpos(b), xpos(a)*ypos(b) - ypos(a)*xpos(b))

struct Vec{T<:Number} <: AbstractVec{T}
    x::T
    y::T
    z::T
end

Base.size(a::Vec) = (3,)
Base.length(a::Vec) = 3
Base.zero(a::Type{Vec{T}}) where {T} = Vec{T}(zero(T),zero(T),zero(T))
Vec(x::T,y::T,z::T) where T = Vec{T}(x,y,z)
Vec(x,y,z) = Vec(promote(x,y,z)...)

#----------------------------------------------- Vectors --------------------------------------------------

#--------------------------------------- Symmetric Traceless Tensors ------------------------------------------

struct SymTrTen{T<:Number} <: AbstractTen{T}
    xx::T
    xy::T
    xz::T
    yy::T 
    yz::T 
end

Base.IndexStyle(a::Type{<:SymTrTen}) = Base.IndexLinear()

@inline Base.@propagate_inbounds function Base.getindex(a::SymTrTen,I::Integer) 
    I <= 3 && return getfield(a,I)
    I == 4 && return a.xy
    I == 5 && return a.yy
    I == 6 && return a.yz
    I == 7 && return a.xz
    I == 8 && return a.yz
    I == 9 && return -(a.xx + a.yy)
end

Base.size(a::SymTrTen) = 
    (3,3)

Base.length(a::SymTrTen) = 
    9

Base.zero(a::Type{SymTrTen{T}}) where {T} = 
    SymTrTen{T}(zero(T),zero(T),zero(T),zero(T),zero(T))

SymTrTen(xx::T,xy::T,xz::T,yy::T,yz::T) where {T} = 
    SymTrTen{T}(xx,xy,xz,yy,yz)

SymTrTen(q,w,e,r,t) = 
    SymTrTen(promote(q,w,e,r,t)...)

Base.zero(a::Type{SymTrTen{T}}) where {T} = SymTrTen{T}(zero(T),zero(T),zero(T),zero(T),zero(T))

@inline Base.:+(a::SymTrTen{T},b::SymTrTen{T2}) where {T,T2} = 
    SymTrTen{promote_type(T,T2)}(a.xx+b.xx, a.xy+b.xy, a.xz+b.xz, a.yy+b.yy, a.yz + b.yz)

@inline Base.:-(a::SymTrTen{T},b::SymTrTen{T2}) where {T,T2} = 
    SymTrTen{promote_type(T,T2)}(a.xx-b.xx, a.xy-b.xy, a.xz-b.xz, a.yy-b.yy, a.yz - b.yz)

@inline Base.:*(a::T,b::SymTrTen{T2}) where {T<:Number,T2<:Number} = 
    SymTrTen{promote_type(T,T2)}(a*b.xx, a*b.xy, a*b.xz, a*b.yy, a*b.yz)

@inline Base.:*(b::SymTrTen{T},a::T2) where {T<:Number,T2<:Number} = 
    a*b

@inline Base.:/(b::SymTrTen{T},a::T2) where {T<:Number,T2<:Number} = 
    SymTrTen{promote_type(T,T2)}(b.xx/a, b.xy/a, b.xz/a, b.yy/a, b.yz/a)

@inline Base.:(:)(a::SymTrTen,b::SymTrTen) = 
    muladd(a.xx, b.xx, muladd(a.xy, b.xy, muladd(a.xz, b.xz, muladd(a.yy, b.yy, a.yz*b.yz))))

@inline LinearAlgebra.norm(a::SymTrTen) = 
    sqrt(2(a:a))

@inline LinearAlgebra.dot(a::SymTrTen{T},b::Vec{T2}) where {T,T2} = 
    Vec{promote_type(T,T2)}(muladd(a.xx, b.x, muladd(a.xy, b.y, a.xz*b.z)), 
        muladd(a.xy, b.x, muladd(a.yy, b.y, a.yz*b.z)),
        muladd(a.xz, b.x, muladd(a.yz, b.y, -(a.xx+a.yy)*b.z)))

@inline LinearAlgebra.dot(b::Vec,a::SymTrTen) = 
    a⋅b

#--------------------------------------- Symmetric Traceless Tensors ------------------------------------------

end