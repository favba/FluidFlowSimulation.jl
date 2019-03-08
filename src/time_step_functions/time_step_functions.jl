abstract type AbstractTimeStep end
abstract type AbstractScalarTimeStep{Adp,N} <: AbstractTimeStep end
abstract type AbstractScalarTimeStepWithIF{Adp,N} <: AbstractScalarTimeStep{Adp,N} end

@inline is_implicit(::Type{<:AbstractScalarTimeStep}) = false
@inline is_implicit(::Type{<:AbstractScalarTimeStepWithIF}) = true
@inline @par is_implicit(::Type{<:@par(AbstractSimulation)}) = is_implicit(VelocityTimeStepType)

has_variable_timestep(t::Type{<:AbstractScalarTimeStep{adp,N}}) where {adp,N} =
    adp

@inline has_variable_timestep(t::AbstractScalarTimeStep{adp,N}) where {adp,N} =
    has_variable_timestep(typeof(t))
  
set_dt!(t::Type{<:AbstractScalarTimeStep{false,N}},dt::Real) where {N} =
    nothing

@inline set_dt!(t::AbstractScalarTimeStep{false,N},dt::Real) where {N} =
    set_dt!(typeof(t))

struct VectorTimeStep{cfl,Tx<:AbstractScalarTimeStep,Ty<:AbstractScalarTimeStep,Tz<:AbstractScalarTimeStep} <: AbstractTimeStep
    x::Tx
    y::Ty 
    z::Tz
end

function VectorTimeStep{cfl}(tx,ty,tz) where {cfl}
    return VectorTimeStep{cfl,typeof(tx),typeof(ty),typeof(tz)}(tx,ty,tz) 
end

@inline is_implicit(::Type{<:VectorTimeStep{cfl,Tx,Ty,Tz}}) where {cfl,Tx,Ty,Tz} = is_implicit(Tx)

get_cfl(t::Type{VectorTimeStep{cfl,Tx,Ty,Tz}}) where{cfl,Tx,Ty,Tz} =
    cfl

@inline get_cfl(t::VectorTimeStep) =
    get_cfl(typeof(t))

@par get_cfl(s::Union{@par(AbstractSimulation),Type{@par(AbstractSimulation)}}) =
    get_cfl(VelocityTimeStepType)

function initialize!(t::VectorTimeStep,rhs::VectorField,s::AbstractSimulation)
    vis = ν
    initialize!(t.x,rhs.rr.x,vis,s)
    initialize!(t.y,rhs.rr.y,vis,s)
    initialize!(t.z,rhs.rr.z,vis,s)
end

function set_dt!(t::VectorTimeStep,dt)
    set_dt!(t.x,dt)
    set_dt!(t.y,dt)
    set_dt!(t.z,dt)
end

@inline get_dt(t::A) where {A<:VectorTimeStep} =
    get_dt(t.x)

@inline get_dt(s::A) where {A<:AbstractSimulation} =
    get_dt(s.timestep)

has_variable_timestep(t::Type{VectorTimeStep{cfl,Tx,Ty,Tz}}) where {cfl,Tx,Ty,Tz} =
    has_variable_timestep(Tx)

@inline has_variable_timestep(t::VectorTimeStep) =
    has_variable_timestep(typeof(t))

@inline @par has_variable_timestep(s::Type{A}) where {A<:@par(AbstractSimulation)} =
    has_variable_timestep(VelocityTimeStepType)

@inline has_variable_timestep(s::A) where {A<:AbstractSimulation} =
    has_variable_timestep(A)

function (f::VectorTimeStep)(u::VectorField,rhs::VectorField,s::AbstractSimulation)
    if hasforcing(s)
        f.x(u.c.x,rhs.c.x,s.forcing.forcex,s)
        f.y(u.c.y,rhs.c.y,s.forcing.forcey,s)
        if typeof(s.forcing) <: RfForcing
            f.z(u.c.z,rhs.c.z,s)
        else
            f.z(u.c.z,rhs.c.z,s.forcing.forcez,s)
        end
    else
        f.x(u.c.x,rhs.c.x,s)
        f.y(u.c.y,rhs.c.y,s)
        f.z(u.c.z,rhs.c.z,s)
    end
    pressure_projection!(u.c.x,u.c.y,u.c.z,s)
    return nothing
end

include("Euller.jl")  
include("Adams_Bashforth.jl")  
include("ETD.jl")  

@par function set_dt!(s::@par(AbstractSimulation))
    umax = maximum(s.reduction)
    dx = 2π*LY/NY
    cfl = get_cfl(s)
    νt = ν
    newdt =cfl * dx/umax
    # is_implicit(s) || (newdt = min(newdt, (2νt/umax^2)/2))
    if hasdensity(s) 
        ρmax = maximum(s.densitystratification.reduction)
        g = norm(gravity(s))
        k = diffusivity(s)
        newdt = min(newdt, 
        cfl * sqrt(dx/(ρmax*g))/2)#,
        #cfl * (dx^2)/(k)/2)#, 
        #cfl * (((ρmax*g)^(-2/3))*(2k)^(1/3))/2,
        #cfl * (2k/umax^2)/2)
        set_dt!(s.densitystratification.timestep,newdt)
    end
    set_dt!(s.timestep,newdt)
    haspassivescalar(s) && set_dt!(s.passivescalar.timestep,newdt)
    return nothing
end
