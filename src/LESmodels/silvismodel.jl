struct SilvisModel{T}<:AbstractLESModel
    c::T
    cp::T
    Δ²::T
    tau::SymTrTenField{T,3,2,false}
    reduction::Vector{T}
end

function SilvisModel(c::T,cp::T,Δ::Real,dim::NTuple{3,Integer}) where {T<:Real}
    data = SymTrTenField{T}(dim,(LX,LY,LZ))
    #fill!(data,0)
    reduction = zeros(THR ? Threads.nthreads() : 1)
    return SilvisModel{T}(c,cp,Δ^2, data,reduction)
end

is_Silvis(a::Union{<:SilvisModel,Type{<:SilvisModel}}) = true
@inline @par is_Silvis(s::Type{T}) where {T<:@par(AbstractSimulation)} = is_Silvis(LESModelType)
@inline is_Silvis(s::T) where {T<:AbstractSimulation} = is_Silvis(T)
is_Silvis(a) = false

msg(a::SilvisModel) = "\nLES model: Silvis\nS Constant: $(a.c)\nP Constant: $(a.cp)\nFilter Width: $(sqrt(a.Δ²))\n"

fvs(S,w) = norm(S⋅w)/(fsqrt(S:S)*norm(w))

@inline function Silvis_eddy_viscosity(S::SymTen,f::Real,c::Real,Δ2::Real)
    return c*c*Δ2*f*f*f*fsqrt(S:S)/2
end

@inline function Silvis_P_coeff(cp::Real,f::Real,Δ2::Real)
    return cp*Δ2*f*f*f*f/4
end