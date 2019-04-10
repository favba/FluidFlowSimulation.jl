abstract type AbstractLESModel end

    is_Smagorinsky(a) = false
    is_SandP(a) = false
    lesscalarmodel(a) = NoLESScalar

struct NoLESModel <: AbstractLESModel end

    statsheader(a::NoLESModel) = ""

    stats(a::NoLESModel,s::AbstractSimulation) = ()

    msg(a::NoLESModel) = "\nLES model: No LES model\n"


# Smagorinsky Model Start ======================================================

stats(a::AbstractLESModel,s::AbstractSimulation) = (les_stats(s.reduction,a.tau,s.u))

abstract type EddyViscosityModel <: AbstractLESModel end

struct Smagorinsky{T} <: EddyViscosityModel
    c::T
    Δ²::T
    tau::SymTrTenField{T,3,2,false}
    reduction::Vector{T}
end

function Smagorinsky(c::T,Δ::Real,dim::NTuple{3,Integer}) where {T<:Real}
    data = SymTrTenField{T}(dim,(LX,LY,LZ))
    #fill!(data,0)
    reduction = zeros(THR ? Threads.nthreads() : 1)
    return Smagorinsky{T}(c,Δ^2, data,reduction)
end

is_Smagorinsky(a::Union{<:Smagorinsky,Type{<:Smagorinsky}}) = true
@inline @par is_Smagorinsky(s::Type{T}) where {T<:@par(AbstractSimulation)} = is_Smagorinsky(LESModelType)
@inline is_Smagorinsky(s::T) where {T<:AbstractSimulation} = is_Smagorinsky(T)
is_Smagorinsky(a) = false

statsheader(a::Smagorinsky) = "pr"

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
    return DynamicSmagorinsky{T}(Δ^2, d2^2, c, cmin, L, M, tau, S, u, reduction)
end

DynamicSmagorinsky(Δ::T, dim::NTuple{3,Integer},b::Bool=false) where {T<:Real} = DynamicSmagorinsky(Δ, 2Δ, dim,b)

is_dynamic_les(a::Union{<:DynamicSmagorinsky,<:Type{<:DynamicSmagorinsky}}) = true
@inline @par is_dynamic_les(s::Type{T}) where {T<:@par(AbstractSimulation)} = is_dynamic_les(LESModelType)
@inline is_dynamic_les(s::T) where {T<:AbstractSimulation} = is_dynamic_les(T)
is_dynamic_les(a) = false

statsheader(a::DynamicSmagorinsky) = "pr"

msg(a::DynamicSmagorinsky) = "\nLES model: Dynamic Smagorinsky\nFilter Width: $(sqrt(a.Δ²))\nTest Filter Width: $(sqrt(a.Δ̂²))\nMinimum coefficient permitted: $(a.cmin)\n"

# Vreman Model Start ======================================================

struct VremanLESModel{T} <: EddyViscosityModel
    c::T
    Δ²::T
    tau::SymTrTenField{T,3,2,false}
    reduction::Vector{T}
end

function VremanLESModel(c::T,Δ::Real,dim::NTuple{3,Integer}) where {T<:Real}
    data = SymTrTenField{T}(dim,(LX,LY,LZ))
    #fill!(data,0)
    reduction = zeros(THR ? Threads.nthreads() : 1)
    return VremanLESModel{T}(c,Δ^2, data,reduction)
end

is_Vreman(a::Union{<:VremanLESModel,Type{<:VremanLESModel}}) = true
@inline @par is_Vreman(s::Type{T}) where {T<:@par(AbstractSimulation)} = is_Vreman(LESModelType)
@inline is_Vreman(s::T) where {T<:AbstractSimulation} = is_Vreman(T)
is_Vreman(a) = false

statsheader(a::VremanLESModel) = "pr"

msg(a::VremanLESModel) = "\nLES model: Vreman\nConstant: $(a.c)\nFilter Width: $(sqrt(a.Δ²))\n"

include("VremanLESModel.jl")

# Vreman Model Start ======================================================

struct ProductionViscosityLESModel{T} <: EddyViscosityModel
    c::T
    Δ²::T
    tau::SymTrTenField{T,3,2,false}
    reduction::Vector{T}
end

function ProductionViscosityLESModel(c::T,Δ::Real,dim::NTuple{3,Integer}) where {T<:Real}
    data = SymTrTenField{T}(dim,(LX,LY,LZ))
    #fill!(data,0)
    reduction = zeros(THR ? Threads.nthreads() : 1)
    return ProductionViscosityLESModel{T}(c,Δ^2, data,reduction)
end

is_production_model(a::Type{T}) where {T<:ProductionViscosityLESModel} = true
@inline @par is_production_model(s::Type{T}) where {T<:@par(AbstractSimulation)} = is_production_model(LESModelType)
@inline is_production_model(s::T) where {T<:AbstractSimulation} = is_production_model(T)
is_production_model(a) = false

statsheader(a::ProductionViscosityLESModel) = "pr"

msg(a::ProductionViscosityLESModel) = "\nLES model: Production Viscosity\nConstant: $(a.c)\nFilter Width: $(sqrt(a.Δ²))\n"

include("ProductionViscosityLESModel.jl")

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
@inline is_FakeSmagorinsky(a::Union{<:SandP{t,S},<:Type{SandP{t,S}}}) where {t,S} = is_FakeSmagorinsky(S)
@inline is_Vreman(a::Union{<:SandP{t,S},<:Type{SandP{t,S}}}) where {t,S} = is_Vreman(S)
@inline is_dynamic_les(a::Union{<:SandP{t,S},<:Type{SandP{t,S}}}) where {t,S} = is_dynamic_les(S)
@inline is_production_model(a::Union{<:SandP{t,S},<:Type{SandP{t,S}}}) where {t,S} = is_production_model(S)

statsheader(a::SandP) = "pr"

msg(a::SandP) = "\nLES model: SandP\nP tensor constant: $(a.cb)\nEddy Viscosity model: {$(msg(a.Smodel))}\n"


# Fake Smagorinsky Model Start ======================================================

struct FakeSmagorinsky{T} <: EddyViscosityModel
    Δ²::T
    tau::SymTrTenField{T,3,2,false}
    reduction::Vector{T}
end

function FakeSmagorinsky(Δ::T,dim::NTuple{3,Integer}) where {T<:Real}
    data = SymTrTenField{T}(dim,(LX,LY,LZ))
    #fill!(data,0)
    reduction = zeros(THR ? Threads.nthreads() : 1)
    return FakeSmagorinsky{T}(Δ^2, data,reduction)
end

is_FakeSmagorinsky(a::Union{<:FakeSmagorinsky,Type{<:FakeSmagorinsky}}) = true
@inline @par is_FakeSmagorinsky(s::Type{T}) where {T<:@par(AbstractSimulation)} = is_FakeSmagorinsky(LESModelType)
@inline is_FakeSmagorinsky(s::T) where {T<:AbstractSimulation} = is_FakeSmagorinsky(T)
is_FakeSmagorinsky(a) = false

statsheader(a::FakeSmagorinsky) = ""

msg(a::FakeSmagorinsky) = "\nLES model: FakeSmagorinsky (nut = 0)\nFilter Width: $(sqrt(a.Δ²))\n"

# Fake Smagorinsky Model End ======================================================

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


function les_types(d,nx,ny,nz,lx,ly,lz)

    lestype = NoLESModel()
    if haskey(d,:lesModel)
        if d[:lesModel] == "Smagorinsky"
            c = haskey(d,:smagorinskyConstant) ? Float64(eval(Meta.parse(d[:smagorinskyConstant]))) : 0.17 
            Δ = haskey(d,:filterWidth) ? Float64(eval(Meta.parse(d[:filterWidth]))) : 2*(lx*2π/nx)  
            lestype = Smagorinsky(c,Δ,(nx,ny,nz))
        elseif d[:lesModel] == "Vreman"
            c = haskey(d,:vremanConstant) ? Float64(eval(Meta.parse(d[:vremanConstant]))) : 2*(0.17)^2
            Δ = haskey(d,:filterWidth) ? Float64(eval(Meta.parse(d[:filterWidth]))) : 2*(lx*2π/nx)  
            lestype = VremanLESModel(c,Δ,(nx,ny,nz))
        elseif d[:lesModel] == "productionViscosity"
            c = haskey(d,:productionConstant) ? Float64(eval(Meta.parse(d[:productionConstant]))) : 0.4
            Δ = haskey(d,:filterWidth) ? Float64(eval(Meta.parse(d[:filterWidth]))) : 2*(lx*2π/nx)  
            lestype = ProductionViscosityLESModel(c,Δ,(nx,ny,nz))
        elseif d[:lesModel] == "dynamicSmagorinsky"
            Δ = haskey(d,:filterWidth) ? Float64(eval(Meta.parse(d[:filterWidth]))) : 2*(lx*2π/nx)
            tΔ = haskey(d,:TestFilterWidth) ? Float64(eval(Meta.parse(d[:TestFilterWidth]))) : 2*Δ
            backscatter = haskey(d,:cmin) ? parse(Float64,d[:cmin]) : 0.0
            lestype = DynamicSmagorinsky(Δ,tΔ,(nx,ny,nz), backscatter)
        elseif d[:lesModel] == "fakeSmagorinsky"
            Δ = haskey(d,:filterWidth) ? Float64(eval(Meta.parse(d[:filterWidth]))) : 2*(lx*2π/nx)  
            lestype = FakeSmagorinsky(Δ,(nx,ny,nz))
        elseif d[:lesModel] == "SandP"
            cb = haskey(d,:pConstant) ? Float64(eval(Meta.parse(d[:pConstant]))) : 0.1
            Slestype = if d[:sLesModel] == "Smagorinsky"
                c = haskey(d,:smagorinskyConstant) ? Float64(eval(Meta.parse(d[:smagorinskyConstant]))) : 0.17 
                Δ = haskey(d,:filterWidth) ? Float64(eval(Meta.parse(d[:filterWidth]))) : 2*(lx*2π/nx)  
                Smagorinsky(c,Δ,(nx,ny,nz))
            elseif d[:sLesModel] == "Vreman"
                c = haskey(d,:vremanConstant) ? Float64(eval(Meta.parse(d[:vremanConstant]))) : 2*(0.17)^2
                Δ = haskey(d,:filterWidth) ? Float64(eval(Meta.parse(d[:filterWidth]))) : 2*(lx*2π/nx)  
                lestype = VremanLESModel(c,Δ,(nx,ny,nz))
            elseif d[:sLesModel] == "productionViscosity"
                c = haskey(d,:productionConstant) ? Float64(eval(Meta.parse(d[:productionConstant]))) : 0.4
                Δ = haskey(d,:filterWidth) ? Float64(eval(Meta.parse(d[:filterWidth]))) : 2*(lx*2π/nx)  
                lestype = ProductionViscosityLESModel(c,Δ,(nx,ny,nz))
            elseif d[:sLesModel] == "dynamicSmagorinsky"
                Δ = haskey(d,:filterWidth) ? Float64(eval(Meta.parse(d[:filterWidth]))) : 2*(lx*2π/nx)
                tΔ = haskey(d,:TestFilterWidth) ? Float64(eval(Meta.parse(d[:TestFilterWidth]))) : 2*Δ
                backscatter = haskey(d,:cmin) ? parse(Float64,d[:cmin]) : 0.0
                DynamicSmagorinsky(Δ,tΔ,(nx,ny,nz), backscatter)
            elseif d[:sLesModel] == "fakeSmagorinsky"
                Δ = haskey(d,:filterWidth) ? Float64(eval(Meta.parse(d[:filterWidth]))) : 2*(lx*2π/nx)  
                FakeSmagorinsky(Δ,(nx,ny,nz))
            end
            lestype = SandP(cb,Slestype)
        end
    end

    return lestype
end