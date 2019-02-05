abstract type AbstractLESModel end

    is_Smagorinsky(a) = false
    is_SandP(a) = false
    lesscalarmodel(a) = NoLESScalar

struct NoLESModel <: AbstractLESModel end

    statsheader(a::NoLESModel) = ""

    stats(a::NoLESModel,s::AbstractSimulation) = ()

    msg(a::NoLESModel) = "\nLES model: No LES model\n"


# Smagorinsky Model Start ======================================================

abstract type EddyViscosityModel <: AbstractLESModel end

struct Smagorinsky{T} <: EddyViscosityModel
    c::T
    Δ²::T
    tau::SymTrTenField{T,3,2,false}
    reduction::Vector{T}
    pr::ScalarField{T,3,2,false}
end

function Smagorinsky(c::T,Δ::Real,dim::NTuple{3,Integer}) where {T<:Real}
    data = SymTrTenField{T}(dim,(LX,LY,LZ))
    pr = ScalarField{T}(dim,(LX,LY,LZ))
    #fill!(data,0)
    reduction = zeros(THR ? Threads.nthreads() : 1)
    return Smagorinsky{T}(c,Δ^2, data,reduction,pr)
end

is_Smagorinsky(a::Union{<:Smagorinsky,Type{<:Smagorinsky}}) = true
@inline @par is_Smagorinsky(s::Type{T}) where {T<:@par(AbstractSimulation)} = is_Smagorinsky(LESModelType)
@inline is_Smagorinsky(s::T) where {T<:AbstractSimulation} = is_Smagorinsky(T)
is_Smagorinsky(a) = false

statsheader(a::Smagorinsky) = "pr"

stats(a::Smagorinsky,s::AbstractSimulation) = (tmean(a.pr.rr,s),)

msg(a::Smagorinsky) = "\nLES model: Smagorinsky\nConstant: $(a.c)\nFilter Width: $(sqrt(a.Δ²))\n"

# Smagorinsky Model End ======================================================

struct DynamicSmagorinsky{T<:Real} <: EddyViscosityModel
    Δ²::T
    Δ̂²::T
    c::ScalarField{T,3,2,false}
    cmin::T
    L::SymTenField{T,3,2,false}
    M::SymTrTenField{T,3,2,false}
    tau::SymTrTenField{T,3,2,false}
    S::SymTrTenField{T,3,2,false}
    û::VectorField{T,3,2,false}
    reduction::Vector{T}
    pr::ScalarField{T,3,2,false}
end

function DynamicSmagorinsky(Δ::T, d2::T, dim::NTuple{3,Integer}, cmin::Real=0.0) where {T<:Real}
    c = ScalarField{T}(dim,(LX,LY,LZ))
    L = SymTenField(dim,(LX,LY,LZ))
    tau = SymTrTenField(dim,(LX,LY,LZ))
    M = similar(tau)
    S = similar(tau)
    u = VectorField{T}(dim,(LX,LY,LZ))
    #fill!(data,0)
    reduction = zeros(THR ? Threads.nthreads() : 1)
    return DynamicSmagorinsky{T}(Δ^2, d2^2, c, cmin, L, M, tau, S, u, reduction, similar(c))
end

DynamicSmagorinsky(Δ::T, dim::NTuple{3,Integer},b::Bool=false) where {T<:Real} = DynamicSmagorinsky(Δ, 2Δ, dim,b)

is_dynamic_les(a::Union{<:DynamicSmagorinsky,<:Type{<:DynamicSmagorinsky}}) = true
@inline @par is_dynamic_les(s::Type{T}) where {T<:@par(AbstractSimulation)} = is_dynamic_les(LESModelType)
@inline is_dynamic_les(s::T) where {T<:AbstractSimulation} = is_dynamic_les(T)
is_dynamic_les(a) = false

statsheader(a::DynamicSmagorinsky) = "pr"

stats(a::DynamicSmagorinsky,s::AbstractSimulation) = (tmean(a.pr.rr,s),)

msg(a::DynamicSmagorinsky) = "\nLES model: Dynamic Smagorinsky\nFilter Width: $(sqrt(a.Δ²))\nTest Filter Width: $(sqrt(a.Δ̂²))\nMinimum coefficient permitted: $(a.cmin)\n"

# Vreman Model Start ======================================================

struct VremanLESModel{T} <: EddyViscosityModel
    c::T
    Δ²::T
    tau::SymTrTenField{T,3,2,false}
    reduction::Vector{T}
    pr::ScalarField{T,3,2,false}
end

function VremanLESModel(c::T,Δ::Real,dim::NTuple{3,Integer}) where {T<:Real}
    data = SymTrTenField{T}(dim,(LX,LY,LZ))
    pr = ScalarField{T}(dim,(LX,LY,LZ))
    #fill!(data,0)
    reduction = zeros(THR ? Threads.nthreads() : 1)
    return VremanLESModel{T}(c,Δ^2, data,reduction,pr)
end

is_Vreman(a::Union{<:VremanLESModel,Type{<:VremanLESModel}}) = true
@inline @par is_Vreman(s::Type{T}) where {T<:@par(AbstractSimulation)} = is_Vreman(LESModelType)
@inline is_Vreman(s::T) where {T<:AbstractSimulation} = is_Vreman(T)
is_Vreman(a) = false

statsheader(a::VremanLESModel) = "pr"

stats(a::VremanLESModel,s::VremanLESModel) = (tmean(a.pr.rr,s),)

msg(a::VremanLESModel) = "\nLES model: Vreman\nConstant: $(a.c)\nFilter Width: $(sqrt(a.Δ²))\n"

include("VremanLESmodel.jl")

# Smagorinsky+P Model Start ======================================================

struct SandP{T<:Real,Smodel<:EddyViscosityModel} <: AbstractLESModel
    cb::T # default: 0.1
    Smodel::Smodel
end

@inline function Base.getproperty(a::SandP,s::Symbol)
    if s === :Smodel || s === :cb
        return getfield(a,s)
    else
        return getfield(getfield(a,:Smodel),s)
    end
end

is_SandP(a::Union{<:SandP,Type{<:SandP}}) = true
@inline @par is_SandP(s::Type{T}) where {T<:@par(AbstractSimulation)} = is_SandP(LESModelType)
@inline is_SandP(s::T) where {T<:AbstractSimulation} = is_SandP(T)

@inline is_Smagorinsky(a::Union{<:SandP{t,S},<:Type{SandP{t,S}}}) where {t,S} = is_Smagorinsky(S)
@inline is_Vreman(a::Union{<:SandP{t,S},<:Type{SandP{t,S}}}) where {t,S} = is_Vreman(S)
@inline is_dynamic_les(a::Union{<:SandP{t,S},<:Type{SandP{t,S}}}) where {t,S} = is_dynamic_les(S)

statsheader(a::SandP) = "pr"

stats(a::SandP,s::AbstractSimulation) = (tmean(a.pr.rr,s),)

msg(a::SandP) = "\nLES model: SandP\nP tensor constant: $(a.cb)\nEddy Viscosity model: [$(msg(a.Smodel))]\n"

# Scalar models =====================================================================

abstract type AbstractLESScalar end

struct NoLESScalar <: AbstractLESScalar end

struct EddyDiffusion <: AbstractLESScalar end

struct VorticityDiffusion{T<:Real} <: AbstractLESScalar
    c::T
end

VorticityDiffusion() = VorticityDiffusion{Float64}(1.3)

is_vorticity_model(::Any) = false
is_vorticity_model(::Type{T}) where {T<:VorticityDiffusion} = true