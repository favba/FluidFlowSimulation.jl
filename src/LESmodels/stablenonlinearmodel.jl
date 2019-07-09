struct StableNonLinearLESModel{T} <: AbstractLESModel
    c::T
    Δ²::T
    tau::SymTrTenField{T,3,2,false}
    reduction::Vector{T}
end

function StableNonLinearLESModel(c::T,Δ::Real,dim::NTuple{3,Integer}) where {T<:Real}
    data = SymTrTenField{T}(dim,(LX,LY,LZ))
    #fill!(data,0)
    reduction = zeros(THR ? Threads.nthreads() : 1)
    return StableNonLinearLESModel{T}(c,Δ^2, data,reduction)
end

is_stable_nl_model(a::Type{T}) where {T<:StableNonLinearLESModel} = true
@inline @par is_stable_nl_model(s::Type{T}) where {T<:@par(AbstractSimulation)} = is_stable_nl_model(LESModelType)
@inline is_stable_nl_model(s::T) where {T<:AbstractSimulation} = is_stable_nl_model(T)
is_stable_nl_model(a) = false

msg(a::StableNonLinearLESModel) = "\nLES model: Stable Non-linear Model\nConstant: $(a.c)\nFilter Width: $(sqrt(a.Δ²))\n"

@inline function stable_nl_eddy_viscosity(S::SymTen,W::AntiSymTen,c::Real,Δ2::Real)
    a = S:S
    nu = ifelse(a == 0.,zero(eltype(a)),
         @fastmath max(0.0,c*Δ2*((square(W):S) - (square(S):S))/(2a)))
    mnu = ifelse(a == 0.,zero(eltype(a)),
         @fastmath -min(0.0,c*Δ2*((square(W):S) - (square(S):S))/(2a)))
    return nu,mnu
end