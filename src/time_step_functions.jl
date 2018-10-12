abstract type AbstractTimeStep end
abstract type AbstractScalarTimeStep{Adp,initdt,N} <: AbstractTimeStep end
abstract type AbstractScalarTimeStepWithIF{Adp,initdt,N} <: AbstractScalarTimeStep{Adp,initdt,N} end

@inline is_implicit(::Type{<:AbstractScalarTimeStep}) = false
@inline is_implicit(::Type{<:AbstractScalarTimeStepWithIF}) = true

has_variable_timestep(t::Type{<:AbstractScalarTimeStep{adp,idt,N}}) where {adp,idt,N} =
    adp

@inline has_variable_timestep(t::AbstractScalarTimeStep{adp,idt,N}) where {adp,idt,N} =
    has_variable_timestep(typeof(t))
  
get_dt(t::Type{<:AbstractScalarTimeStep{false,idt,N}}) where {idt,N} =
    idt

@inline get_dt(t::AbstractScalarTimeStep{false,idt,N}) where {idt,N} =
    get_dt(typeof(t))

set_dt!(t::Type{<:AbstractScalarTimeStep{false,idt,N}},dt::Real) where {idt,N} =
    nothing

@inline set_dt!(t::AbstractScalarTimeStep{false,idt,N},dt::Real) where {idt,N} =
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
    vis = nu(s)
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
        f.x(u.cx,rhs.cx,s.forcing.forcex,s)
        f.y(u.cy,rhs.cy,s.forcing.forcey,s)
        if typeof(s.forcing) <: RfForcing
            f.z(u.cz,rhs.cz,s)
        else
            f.z(u.cz,rhs.cz,s.forcing.forcez,s)
        end
    else
        f.x(u.cx,rhs.cx,s)
        f.y(u.cy,rhs.cy,s)
        f.z(u.cz,rhs.cz,s)
    end
    pressure_projection!(u.cx,u.cy,u.cz,s)
    return nothing
end

include("time_step_functions/Euller.jl")  
include("time_step_functions/Adams_Bashforth.jl")  
include("time_step_functions/ETD.jl")  

@par function set_dt!(s::@par(AbstractSimulation))
    umax = maximum(s.reduction)
    dx = 2π*Ly/Ny
    cfl = get_cfl(s)
    #νt = nu(s)
    newdt =cfl * dx/umax
    #newdt = min(newdt, cfl * (2νt/umax^2)/2)
    if hasles(s)
        nut_max = maximum(s.lesmodel.reduction)
        newdt = min(newdt,cfl*((dx^2)/nut_max)/2)
    end
    if hasdensity(s) 
        ρmax = maximum(s.densitystratification.reduction)
        g = abs(gravity(s))
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
