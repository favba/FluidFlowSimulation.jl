struct CatVec{T,A<:AbstractVector{T},B<:AbstractVector{T},L1,L2} <: AbstractVector{T}
    i::A
    f::B
end

CatVec(i::AbstractVector,f::AbstractVector) = CatVec{eltype(i),typeof(i),typeof(f),length(i),length(f)}(i,f)

Base.@pure Base.length(a::CatVec{T,A,B,L1,L2}) where {T,A,B,L1,L2} = L1+L2
Base.@pure Base.size(a::CatVec) = (length(a),)

Base.@propagate_inbounds @inline function Base.getindex(x::CatVec{T,A,B,L1,L2},i::Int) where {T,A,B,L1,L2}
    @boundscheck if i < 1 || i > length(x)
        throw(BoundsError(x,i))
    end
    return i<=L1 ? x.i[i] : x.f[i-L1]
end

Base.iterate(l::CatVec,state=firstindex(l)) = state <= length(l) ? (@inbounds l[state], state + 1) : nothing