abstract type AbstractLESModel end

    is_Smagorinsky(a) = false
    is_SandP(a) = false
    lesscalarmodel(a) = NoLESScalar

struct NoLESModel <: AbstractLESModel end

    statsheader(a::NoLESModel) = ""

    stats(a::NoLESModel,s::AbstractSimulation) = ()

    msg(a::NoLESModel) = "\nLES model: No LES model\n"


# Smagorinsky Model Start ======================================================

stats(a::AbstractLESModel,s::AbstractSimulation) = (les_stats(s.reductionh,s.reductionv,a.tau,s.u))
statsheader(a::AbstractLESModel) = "prh,prv,praniso,pr"

abstract type EddyViscosityModel <: AbstractLESModel end
abstract type DynamicEddyViscosityModel <: EddyViscosityModel end

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

msg(a::Smagorinsky) = "\nLES model: Smagorinsky\nConstant: $(a.c)\nFilter Width: $(sqrt(a.Δ²))\n"

# Smagorinsky Model End ======================================================

struct DynamicSmagorinsky{T<:Real} <: DynamicEddyViscosityModel
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

# Smagorinsky+P Model Start ======================================================

struct DynSandP{T<:Real,Smodel<:DynamicEddyViscosityModel} <: AbstractLESModel
    P::SymTrTenField{T,3,2,false}
    ŵ::VectorField{T,3,2,false}
    cp::ScalarField{T,3,2,false}
    Smodel::Smodel
end

function DynSandP(s::S) where S<:DynamicEddyViscosityModel
    P = SymTrTenField((NRX,NY,NZ),(LX,LY,LZ))
    w = VectorField((NRX,NY,NZ),(LX,LY,LZ))
    cp = ScalarField((NRX,NY,NZ),(LX,LY,LZ))
    return DynSandP{Float64,S}(P,w,cp,s)
end

@inline function Base.getproperty(a::DynSandP,s::Symbol)
    if s === :Smodel || s === :P || s === :ŵ || s === :cp
        return getfield(a,s)
    else
        return getfield(getfield(a,:Smodel),s)
    end
end

is_dynP_les(a::Union{<:DynSandP,<:Type{<:DynSandP}}) = true
is_dynP_les(a) = false
@inline @par is_dynP_les(s::Type{T}) where {T<:@par(AbstractSimulation)} = is_dynP_les(LESModelType)
@inline is_dynP_les(s::T) where {T<:AbstractSimulation} = is_dynP_les(T)

@inline is_Smagorinsky(a::Union{<:DynSandP{t,S},<:Type{DynSandP{t,S}}}) where {t,S} = is_Smagorinsky(S)
@inline is_FakeSmagorinsky(a::Union{<:DynSandP{t,S},<:Type{DynSandP{t,S}}}) where {t,S} = is_FakeSmagorinsky(S)
@inline is_Vreman(a::Union{<:DynSandP{t,S},<:Type{DynSandP{t,S}}}) where {t,S} = is_Vreman(S)
@inline is_dynamic_les(a::Union{<:DynSandP{t,S},<:Type{DynSandP{t,S}}}) where {t,S} = is_dynamic_les(S)
@inline is_production_model(a::Union{<:DynSandP{t,S},<:Type{DynSandP{t,S}}}) where {t,S} = is_production_model(S)

msg(a::DynSandP) = "\nLES model: Dynamic SandP\nEddy Viscosity model: {$(msg(a.Smodel))}\n"

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
            Δ = haskey(d,:filterWidth) ? Float64(eval(Meta.parse(d[:filterWidth]))) : (lx*2π/nx)/Globals.cutoffr
            lestype = Smagorinsky(c,Δ,(nx,ny,nz))
        elseif d[:lesModel] == "Vreman"
            c = haskey(d,:vremanConstant) ? Float64(eval(Meta.parse(d[:vremanConstant]))) : 2*(0.17)^2
            Δ = haskey(d,:filterWidth) ? Float64(eval(Meta.parse(d[:filterWidth]))) : (lx*2π/nx)/Globals.cutoffr
            lestype = VremanLESModel(c,Δ,(nx,ny,nz))
        elseif d[:lesModel] == "productionViscosity"
            c = haskey(d,:productionConstant) ? Float64(eval(Meta.parse(d[:productionConstant]))) : 0.4
            Δ = haskey(d,:filterWidth) ? Float64(eval(Meta.parse(d[:filterWidth]))) : (lx*2π/nx)/Globals.cutoffr
            lestype = ProductionViscosityLESModel(c,Δ,(nx,ny,nz))
        elseif d[:lesModel] == "dynamicSmagorinsky"
            Δ = haskey(d,:filterWidth) ? Float64(eval(Meta.parse(d[:filterWidth]))) : (lx*2π/nx)/Globals.cutoffr
            tΔ = haskey(d,:TestFilterWidth) ? Float64(eval(Meta.parse(d[:TestFilterWidth]))) : 2*Δ
            backscatter = haskey(d,:cmin) ? parse(Float64,d[:cmin]) : 0.0
            lestype = DynamicSmagorinsky(Δ,tΔ,(nx,ny,nz), backscatter)
        elseif d[:lesModel] == "fakeSmagorinsky"
            Δ = haskey(d,:filterWidth) ? Float64(eval(Meta.parse(d[:filterWidth]))) : (lx*2π/nx)/Globals.cutoffr
            lestype = FakeSmagorinsky(Δ,(nx,ny,nz))
        elseif d[:lesModel] == "SandP"
            cb = haskey(d,:pConstant) ? Float64(eval(Meta.parse(d[:pConstant]))) : 0.1
            Slestype = if d[:sLesModel] == "Smagorinsky"
                c = haskey(d,:smagorinskyConstant) ? Float64(eval(Meta.parse(d[:smagorinskyConstant]))) : 0.17 
                Δ = haskey(d,:filterWidth) ? Float64(eval(Meta.parse(d[:filterWidth]))) : 2*(lx*2π/nx)/Globals.cutoffr
                Smagorinsky(c,Δ,(nx,ny,nz))
            elseif d[:sLesModel] == "Vreman"
                c = haskey(d,:vremanConstant) ? Float64(eval(Meta.parse(d[:vremanConstant]))) : 2*(0.17)^2
                Δ = haskey(d,:filterWidth) ? Float64(eval(Meta.parse(d[:filterWidth]))) : (lx*2π/nx)/Globals.cutoffr
                lestype = VremanLESModel(c,Δ,(nx,ny,nz))
            elseif d[:sLesModel] == "productionViscosity"
                c = haskey(d,:productionConstant) ? Float64(eval(Meta.parse(d[:productionConstant]))) : 0.1
                Δ = haskey(d,:filterWidth) ? Float64(eval(Meta.parse(d[:filterWidth]))) : (lx*2π/nx)/Globals.cutoffr
                lestype = ProductionViscosityLESModel(c,Δ,(nx,ny,nz))
            elseif d[:sLesModel] == "dynamicSmagorinsky"
                Δ = haskey(d,:filterWidth) ? Float64(eval(Meta.parse(d[:filterWidth]))) : (lx*2π/nx)/Globals.cutoffr
                tΔ = haskey(d,:TestFilterWidth) ? Float64(eval(Meta.parse(d[:TestFilterWidth]))) : 2*Δ
                backscatter = haskey(d,:cmin) ? parse(Float64,d[:cmin]) : 0.0
                DynamicSmagorinsky(Δ,tΔ,(nx,ny,nz), backscatter)
            elseif d[:sLesModel] == "fakeSmagorinsky"
                Δ = haskey(d,:filterWidth) ? Float64(eval(Meta.parse(d[:filterWidth]))) : (lx*2π/nx)/Globals.cutoffr
                FakeSmagorinsky(Δ,(nx,ny,nz))
            end
            lestype = SandP(cb,Slestype)
        elseif d[:lesModel] == "dynamicSandP"
            Slestype = if d[:sLesModel] == "dynamicSmagorinsky"
                Δ = haskey(d,:filterWidth) ? Float64(eval(Meta.parse(d[:filterWidth]))) : (lx*2π/nx)/Globals.cutoffr
                tΔ = haskey(d,:TestFilterWidth) ? Float64(eval(Meta.parse(d[:TestFilterWidth]))) : 2*Δ
                backscatter = haskey(d,:cmin) ? parse(Float64,d[:cmin]) : 0.0
                DynamicSmagorinsky(Δ,tΔ,(nx,ny,nz), backscatter)
            end
            lestype = DynSandP(Slestype)
        end
    end

    return lestype
end