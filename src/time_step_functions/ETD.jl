struct ETD3rdO{Adaptative,initdt,Hyper} <: AbstractScalarTimeStepWithIF{Adaptative,initdt,2}
    fm1::ScalarField{Float64,3,2,false} #Store latest step
    fm2::ScalarField{Float64,3,2,false} #Store 2 steps before
    c::Array{Float64,3}
    At::Array{Float64,3}
    Bt::Array{Float64,3}
    Ct::Array{Float64,3}
    dt::Base.RefValue{Float64}
    dt2::Base.RefValue{Float64}
    dt3::Base.RefValue{Float64}
end

function ETD3rdO{adp,indt,Hyper}() where {adp,indt,Hyper}
    dt = Ref(indt)
    dt2 = Ref(indt)
    dt3 = Ref(indt)
    c = zeros(NX,NY,NZ)
    At = zero(c)
    Bt = zero(c)
    Ct = zero(c)
    fm1 = ScalarField{Float64}((NRX,NY,NZ),(LX,LY,LZ))
    fm2 = ScalarField{Float64}((NRX,NY,NZ),(LX,LY,LZ))
    return ETD3rdO{adp,indt,Hyper}(fm1,fm2,c,At,Bt,Ct,dt,dt2,dt3)
end

get_dt(t::ETD3rdO{true,idt,H}) where {idt,H} = 
    getindex(t.dt)

function initialize!(t::ETD3rdO,rhs::AbstractArray,vis,s::AbstractSimulation)
    mycopy!(data(t.fm1),rhs) 
    mycopy!(data(t.fm2), data(t.fm1)) 
    setindex!(t.dt,get_dt(s))
    setindex!(t.dt2,t.dt[])
    setindex!(t.dt3,t.dt[])
    init_c!(t,t.c,-vis,s)
    set_ABCt!(t)
    return nothing
end

@par function init_c!(t::ETD3rdO{adp,indt,false},c::AbstractArray,mν,s::@par(AbstractSimulation)) where{adp,indt}
    @mthreads for k in ZRANGE
        for j in YRANGE
            @inbounds @msimd for i in XRANGE
                c[i,j,k] = muladd(KX[i], KX[i], muladd(KY[j], KY[j], KZ[k]*KZ[k]))*mν
            end
        end
    end
end

@par function init_c!(t::ETD3rdO{adp,indt,true},c::AbstractArray,aux,s::@par(AbstractSimulation)) where {adp,indt}
    mν::Float64 = -ν
    mνh::Float64 = -nuh(s)
    M::Int = get_hyperviscosity_exponent(s)
    @mthreads for k in ZRANGE
        for j in YRANGE
            @inbounds @msimd for i in XRANGE
                modk = muladd(KX[i], KX[i], muladd(KY[j], KY[j], KZ[k]*KZ[k])) 
                c[i,j,k] = muladd(modk, mν, modk^M * mνh) 
            end
        end
    end
end

function set_dt!(t::ETD3rdO{true,idt,Hyper},dt::Real) where {idt,Hyper} 
    setindex!(t.dt3,t.dt2[])
    setindex!(t.dt2,t.dt[])
    setindex!(t.dt,dt)
    set_ABCt!(t)
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
    b0,b1,b2,b3 = get_bts(t)
    c0,c1,c2,c3 = get_cts(t)

    e1 = dt^2 + dt*(2dt2+dt3)
    e2 = dt2*(dt2+dt3)
    e3 = 2dt
    e4 = 2dt2+dt3

    e5 = dt*(dt+dt2+dt3)
    e6 = dt2+dt3
    e7 = dt2*dt3
    e8 = dt*(dt+dt2)
    e9 = dt3*(dt2+dt3)
 
    @inbounds for i in COMPLEX_RANGES[j]
        l1 = l[i]
        l2 = l1*l1
        l3 = l2*l1
        ldt = l1*dt
        test = -l1*dt<=0.01

        At[i] = ifelse(test,
            muladd(a1, l1, muladd(a2, l2, muladd(a3, l3, a0))),
            (2expm1(ldt) - l2*(e1 - e2*expm1(ldt)) - l1*(e3 - e4*expm1(ldt))) / (l3*e2))
        Bt[i] = ifelse(test,
            muladd(b1, l1, muladd(b2, l2, muladd(b3, l3, b0))),
            (e5*l2 - 2expm1(ldt) + l1*(e3 - e6*expm1(ldt)) )/(l3*e7))
        Ct[i] = ifelse(test,muladd(c1, l1, muladd(c2, l2, muladd(c3, l3, c0))),
            (2expm1(ldt) - e8*l2 - l1*(e3 - dt2*expm1(ldt))) / (l3*e9))
    end

    return nothing
end

@inline function get_ats(t::ETD3rdO)
    dt = t.dt[]
    dt2 = t.dt2[]
    dt3 = t.dt3[]
    a0 = (dt*(2*dt*dt + 6dt2*(dt2 + dt3) + 3dt*(2dt2 + dt3)))/(6*dt2*(dt2+dt3))
    a1 = (dt*dt*(dt^2 + 6dt2*(dt2 + dt3) + 2dt*(2dt2 + dt3)))/(12dt2*(dt2 + dt3))
    a2 = (dt*dt*dt*(2*dt^2 + 20dt2*(dt2 + dt3) + 5dt*(2dt2 + dt3)))/(120dt2*(dt2 + dt3))
    a3 = (dt*dt*dt*dt*(dt^2 + 15dt2*(dt2 + dt3) + 3dt*(2dt2 + dt3)))/(360dt2*(dt2 + dt3))
    return a0,a1,a2,a3
end

@inline function get_bts(t::ETD3rdO)
    dt = t.dt[]
    dt2 = t.dt2[]
    dt3 = t.dt3[]
    b0 = -((dt^2 * (2dt + 3*(dt2 + dt3)))/(6*(dt2*dt3)))
    b1 = -(((dt*dt*dt*(dt + 2*(dt2 + dt3))))/(12*(dt2*dt3)))
    b2 = -((dt*dt*dt*dt*(2dt + 5*(dt2 + dt3)))/(120*(dt2*dt3)))
    b3 = -(((dt*dt*dt*dt*(dt + 3*(dt2 + dt3))))/(360*(dt2*dt3)))
    return b0,b1,b2,b3
end

@inline function get_cts(t::ETD3rdO)
    dt = t.dt[]
    dt2 = t.dt2[]
    dt3 = t.dt3[]
    c0 = (dt*dt*(2dt + 3dt2))/(6dt3*(dt2 + dt3))
    c1 = (dt*dt*dt*(dt + 2dt2))/(12dt3*(dt2 + dt3))
    c2 = (dt*dt*dt*dt*(2dt + 5dt2))/(120dt3*(dt2 + dt3))
    c3 = (dt^5 * (dt + 3dt2))/(360dt3*(dt2 + dt3))
    return c0,c1,c2,c3
end

@par function (f::ETD3rdO)(ρ::AbstractArray{<:Complex,3},ρrhs::AbstractArray{<:Complex,3}, s::@par(AbstractSimulation))
    @mthreads for kk = ZRANGE
        _tETD3rdO!(kk, ρ,ρrhs,f.fm1,f.fm2,f,s)
    end
    return nothing
end

@inline @par function _tETD3rdO!(k::Integer, u::AbstractArray{T,3}, rhs::AbstractArray, rm1::AbstractArray, rm2::AbstractArray, f, s::@par(AbstractSimulation)) where {T}
@inbounds begin
    At = f.At
    Bt = f.Bt
    Ct = f.Ct
    c = f.c
    dt = get_dt(f)
    for j in YRANGE
        @msimd for i in XRANGE
            u[i,j,k] = muladd(At[i,j,k], rhs[i,j,k], muladd(Bt[i,j,k], rm1[i,j,k], muladd(Ct[i,j,k], rm2[i,j,k], exp(c[i,j,k]*dt)*u[i,j,k])))
            rm2[i,j,k] = rm1[i,j,k]
            rm1[i,j,k] = rhs[i,j,k]
        end
    end
end
end

# with forcing
@par function (f::ETD3rdO)(ρ::AbstractArray{<:Complex,3},ρrhs::AbstractArray{<:Complex,3}, forcing::AbstractArray{<:Complex,3}, s::@par(AbstractSimulation))
    @mthreads for kk = ZRANGE
        _tETD3rdO!(kk, ρ,ρrhs, forcing, f.fm1,f.fm2,f,s)
    end
    return nothing
end

@inline @par function _tETD3rdO!(k::Integer, u::AbstractArray{Complex{Float64},3}, rhs::AbstractArray, forcing, rm1::AbstractArray, rm2::AbstractArray, f,s::@par(AbstractSimulation)) 
@inbounds begin
    At = f.At
    Bt = f.Bt
    Ct = f.Ct
    c = f.c
    dt = get_dt(f)
    if (6 < k < NZ-k+1)
        for j in YRANGE
            @msimd for i in XRANGE
                u[i,j,k] = muladd(At[i,j,k], rhs[i,j,k], muladd(Bt[i,j,k], rm1[i,j,k], muladd(Ct[i,j,k], rm2[i,j,k], exp(c[i,j,k]*dt)*u[i,j,k])))
                rm2[i,j,k] = rm1[i,j,k]
                rm1[i,j,k] = rhs[i,j,k]
            end
        end
    else
        for j in YRANGE
            @msimd for i in XRANGE
                u[i,j,k] = muladd(At[i,j,k], rhs[i,j,k], muladd(Bt[i,j,k], rm1[i,j,k], muladd(Ct[i,j,k], rm2[i,j,k],muladd(exp(c[i,j,k]*dt), u[i,j,k], forcing[i,j,k]))))
                rm2[i,j,k] = rm1[i,j,k]
                rm1[i,j,k] = rhs[i,j,k]
            end
        end
    end
end
end
