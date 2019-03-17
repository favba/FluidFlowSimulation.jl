include("ETD1stand2ndO.jl")

struct ETD3rdO_Coefficients{Adptive,HyperviscosityType}
    c::Array{Float64,3}
    At::Array{Float64,3}
    Bt::Array{Float64,3}
    Ct::Array{Float64,3}
    hyperviscosity::HyperviscosityType
    citeration::Base.RefValue{Int}
    ETD3rdO_Coefficients{A,H}(c,At,Bt,Ct,h) where {A,H} = new{A,H}(c,At,Bt,Ct,h,Ref(2))
end

const ETD3rdO_coefficients_dict = Dict()

struct ETD3rdO{Adaptative, HyperViscosityType} <: AbstractScalarTimeStepWithIF{Adaptative,2}
    fm1::ScalarField{Float64,3,2,false} #Store latest step
    fm2::ScalarField{Float64,3,2,false} #Store 2 steps before
    coefficients::Base.RefValue{ETD3rdO_Coefficients{Adaptative,HyperViscosityType}}
    dt::Base.RefValue{Float64}
    dt2::Base.RefValue{Float64}
    dt3::Base.RefValue{Float64}
    iteration::Base.RefValue{Int}
end

@inline function Base.getproperty(t::ETD3rdO,s::Symbol)
    if s  === :c || s === :At || s === :Bt || s === :Ct || s === :hyperviscosity || s === :citeration
        return getfield(getindex(getfield(t,:coefficients)),s)
    else
        return getfield(t,s)
    end
end

function ETD3rdO(adp::Bool,indt::Real,hviscosity,ν::Real)
    dt = Ref(indt)
    dt2 = Ref(indt)
    dt3 = Ref(indt)
    p = ν => hviscosity 

    if haskey(ETD3rdO_coefficients_dict, p)
        coef = ETD3rdO_coefficients_dict[p]
    else
        c = zeros(NX,NY,NZ)
        At = zeros(size(c))
        Bt = zeros(size(c))
        Ct = zeros(size(c))
        init_c!(c,ν,hviscosity)
        coef = Ref(ETD3rdO_Coefficients{adp,typeof(hviscosity)}(c,At,Bt,Ct,hviscosity))
        ETD3rdO_coefficients_dict[p] = coef
    end

    fm1 = ScalarField{Float64}((NRX,NY,NZ),(LX,LY,LZ))
    fm2 = ScalarField{Float64}((NRX,NY,NZ),(LX,LY,LZ))
    return ETD3rdO{adp,typeof(hviscosity)}(fm1,fm2,coef,dt,dt2,dt3,Ref(0))
end

init_c!(c::Array{Float64,3},ν::Real,::NoHyperViscosity) = init_c!(c,-ν)
init_c!(c::Array{Float64,3},ν::Real,hv::HyperViscosity) = init_c_hv!(c,-ν,hv)
init_c!(c::Array{Float64,3},ν::Real,hv::SpectralBarrier) = init_c_spectral_barrier!(c,-ν,hv)

function init_c!(c::AbstractArray,mν::Real)
    @mthreads for k in ZRANGE
        for j in YRANGE
            @inbounds @msimd for i in XRANGE
                c[i,j,k] = muladd(KX[i], KX[i], muladd(KY[j], KY[j], KZ[k]*KZ[k]))*mν
            end
        end
    end
end

function init_c_spectral_barrier!(c::AbstractArray,mν::Real,hv::SpectralBarrier)
    @mthreads for k in ZRANGE
        f = hv.func
        for j in YRANGE
            @inbounds @msimd for i in XRANGE
                modk = muladd(KX[i], KX[i], muladd(KY[j], KY[j], KZ[k]*KZ[k])) 
                c[i,j,k] = muladd(modk, mν, f(sqrt(modk))) 
            end
        end
    end
end

function init_c_hv!(c::AbstractArray,mν::Real,hv::HyperViscosity{n,M}) where {n,M}
    mνh::Float64 = -n
    @mthreads for k in ZRANGE
        for j in YRANGE
            @inbounds @msimd for i in XRANGE
                modk = muladd(KX[i], KX[i], muladd(KY[j], KY[j], KZ[k]*KZ[k])) 
                c[i,j,k] = muladd(modk, mν, modk^M * mνh) 
            end
        end
    end
end

get_dt(t::ETD3rdO) = getindex(t.dt)

function initialize!(t::ETD3rdO,rhs::AbstractArray,vis,s::AbstractSimulation)
    mycopy!(data(t.fm1),rhs) 
    mycopy!(data(t.fm2), data(t.fm1)) 
    setindex!(t.dt,get_dt(s))
    setindex!(t.dt2,t.dt[])
    setindex!(t.dt3,t.dt[])
    set_dt!(t,t.dt[])
    return nothing
end

function set_dt!(t::ETD3rdO{true,Hyper},dt::Real) where {Hyper} 
        setindex!(t.dt3,t.dt2[])
        setindex!(t.dt2,t.dt[])
        setindex!(t.dt,dt)
end

function set_coefficients!(t::ETD3rdO{true},rhs)
    i = t.iteration[]
    if i > t.citeration[]
        t.citeration[] += 1
        set_ABCt!(t)
    end
    return nothing
end

function set_coefficients!(t::ETD3rdO{false},rhs)
    i = t.iteration[]
    return nothing
end

function fix_fm2!(fm2::AbstractArray,fm1::AbstractArray,rhs::AbstractArray,dt2::Real,dt3::Real)
    @mthreads for j in 1:NT
        fix_fm2!(fm2,fm1,rhs,dt2,dt3,j)
    end
end

function fix_fm2!(fm2::AbstractArray,fm1::AbstractArray,rhs::AbstractArray,dt2::Real,dt3::Real,j::Integer)
    @inbounds @msimd for i in COMPLEX_RANGES[j]
        fm2[i] = -(dt3+dt2)*(rhs[i] - fm1[i])/dt2 + rhs[i]
    end
end

function set_ABCt!(t::ETD3rdO)
    @mthreads for j in 1:NT
        set_ABCt!(t,j)
    end
    return nothing
end

function set_ABCt!(t::ETD3rdO,j::Integer)
    l = t.c
    At = t.At
    Bt = t.Bt
    Ct = t.Ct
    dt = t.dt[]
    dt2 = t.dt2[]
    dt3 = t.dt3[]
    a0,a1,a2,a3 = get_ats(t)
    b0,b1,b2 = get_bts(t)
    c0,c1,c2 = get_cts(t)

    e1 = dt^2 + dt*(2dt2+dt3)
    e2 = dt2*(dt2+dt3)
    e3 = 2dt
    e4 = 2dt2+dt3

    e5 = dt*(dt+dt2+dt3)
    e6 = dt2+dt3
    e7 = dt2*dt3
    e8 = dt*(dt+dt2)
    e9 = dt3*(dt2+dt3)
 
    @inbounds @msimd for i in COMPLEX_RANGES[j]
        l1 = l[i]
        l2 = l1*l1
        l3 = l2*l1
        ldt = l1*dt
        test = -l1*dt<=1e-4
        test2 = l1 == -Inf

        At[i] = ifelse(test2, 0.0, ifelse(test,
            muladd(a1, l1, muladd(a2, l2, muladd(a3, l3, a0))),
            (2expm1(ldt) - l2*(e1 - e2*expm1(ldt)) - l1*(e3 - e4*expm1(ldt))) / (l3*e2)))
        Bt[i] = ifelse(test2, 0.0, ifelse(test,
            muladd(b1, l1, muladd(b2, l2, b0)),
            (e5*l2 - 2expm1(ldt) + l1*(e3 - e6*expm1(ldt)) )/(l3*e7)))
        Ct[i] = ifelse(test2, 0.0, ifelse(test,muladd(c1, l1, muladd(c2, l2, c0)),
            (2expm1(ldt) - e8*l2 - l1*(e3 - dt2*expm1(ldt))) / (l3*e9)))
    end

    return nothing
end

@inline function get_ats(t::ETD3rdO)
    dt = t.dt[]
    dt2 = t.dt2[]
    dt3 = t.dt3[]
    a0 = (dt*(2*dt*dt + 6dt2*(dt2 + dt3) + 3dt*(2dt2 + dt3)))/(6*dt2*(dt2+dt3))
    a1 = (dt*dt*(dt^2 + 6dt2*(dt2 + dt3) + 2dt*(2dt2 + dt3)))/(12dt2*(dt2 + dt3))
    a2 = (dt*dt*dt*(4dt2*(dt2 + dt3) + dt*(2dt2 + dt3)))/(24*dt2*(dt2+dt3))
    a3 = dt*dt*dt*dt/24 
    return a0,a1,a2,a3
end

@inline function get_bts(t::ETD3rdO)
    dt = t.dt[]
    dt2 = t.dt2[]
    dt3 = t.dt3[]
    b0 = -((dt^2 * (2dt + 3*(dt2 + dt3)))/(6*(dt2*dt3)))
    b1 = -(((dt*dt*dt*(dt + 2*(dt2 + dt3))))/(12*(dt2*dt3)))
    b2 = -((dt*dt*dt*dt*(dt2 + dt3))/(24*(dt2*dt3)))
    return b0,b1,b2
end

@inline function get_cts(t::ETD3rdO)
    dt = t.dt[]
    dt2 = t.dt2[]
    dt3 = t.dt3[]
    c0 = (dt*dt*(2dt + 3dt2))/(6dt3*(dt2 + dt3))
    c1 = (dt*dt*dt*(dt + 2dt2))/(12dt3*(dt2 + dt3))
    c2 = (dt*dt*dt*dt*dt2)/(24*dt3*(dt2 + dt3))
    return c0,c1,c2
end

@par function (f::ETD3rdO)(ρ::AbstractArray{<:Complex,3},ρrhs::AbstractArray{<:Complex,3}, s::@par(AbstractSimulation))
    i = f.iteration[] += 1
    if i == 1
        ETD1stO!(ρ,ρrhs,f.c,f.dt[])
    elseif i == 2
        ETD2ndO!(ρ,ρrhs,f.c,f.fm1,f.dt[],f.dt2[])
    else
        set_coefficients!(f,ρrhs)
        @mthreads for kk = ZRANGE
            _tETD3rdO!(kk, ρ,ρrhs,f.fm1,f.fm2,f,s)
        end
    end
    mycopy!(f.fm2,f.fm1)
    mycopy!(f.fm1,ρrhs)
    return nothing
end

@par function _tETD3rdO!(k::Integer, u::AbstractArray{T,3}, rhs::AbstractArray, rm1::AbstractArray, rm2::AbstractArray, f, s::@par(AbstractSimulation)) where {T}
@inbounds begin
    At = f.At
    Bt = f.Bt
    Ct = f.Ct
    c = f.c
    dt = get_dt(f)
    for j in YRANGE
        @msimd for i in XRANGE
            u[i,j,k] = muladd(At[i,j,k], rhs[i,j,k], muladd(Bt[i,j,k], rm1[i,j,k], muladd(Ct[i,j,k], rm2[i,j,k], exp(c[i,j,k]*dt)*u[i,j,k])))
        end
    end
end
end

# with forcing
#@par function (f::ETD3rdO)(ρ::AbstractArray{<:Complex,3},ρrhs::AbstractArray{<:Complex,3}, forcing::AbstractArray{<:Complex,3}, s::@par(AbstractSimulation))
#    set_coefficients!(f,ρrhs)
#    @mthreads for kk = ZRANGE
#        _tETD3rdO!(kk, ρ,ρrhs, forcing, f.fm1,f.fm2,f,s)
#    end
#    mycopy!(f.fm2,f.fm1)
#    mycopy!(f.fm1,ρrhs)
#    return nothing
#end

#@inline @par function _tETD3rdO!(k::Integer, u::AbstractArray{Complex{Float64},3}, rhs::AbstractArray, forcing, rm1::AbstractArray, rm2::AbstractArray, f,s::@par(AbstractSimulation)) 
#@inbounds begin
#    At = f.At
#    Bt = f.Bt
#    Ct = f.Ct
#    c = f.c
#    dt = get_dt(f)
#    if (6 < k < NZ-4)
#        for j in YRANGE
#            @msimd for i in XRANGE
#                u[i,j,k] = muladd(At[i,j,k], rhs[i,j,k], muladd(Bt[i,j,k], rm1[i,j,k], muladd(Ct[i,j,k], rm2[i,j,k], exp(c[i,j,k]*dt)*u[i,j,k])))
#            end
#        end
#    else
#        for j in YRANGE
#            @msimd for i in XRANGE
#                u[i,j,k] = muladd(At[i,j,k], rhs[i,j,k], muladd(Bt[i,j,k], rm1[i,j,k], muladd(Ct[i,j,k], rm2[i,j,k],muladd(exp(c[i,j,k]*dt), u[i,j,k], forcing[i,j,k]))))
#            end
#        end
#    end
#end
#end
