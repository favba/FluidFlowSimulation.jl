struct ETD3rdO{Adaptative,initdt,Hyper} <: AbstractScalarTimeStepWithIF{Adaptative,initdt,2}
    fm1::PaddedArray{Float64,3,false} #Store latest step
    fm2::PaddedArray{Float64,3,false} #Store 2 steps before
    c::Array{Float64,3}
    At::Array{Float64,3}
    Bt::Array{Float64,3}
    Ct::Array{Float64,3}
    dt::Base.RefValue{Float64}
    dt2::Base.RefValue{Float64}
    dt3::Base.RefValue{Float64}
end

function ETD3rdO{adp,indt,Hyper}(Kxr,Kyr,Kzr,nx,ny,nz) where {adp,indt,Hyper}
    dt = Ref(indt)
    dt2 = Ref(indt)
    dt3 = Ref(indt)
    c = zeros(nx,ny,nz)
    At = zero(c)
    Bt = zero(c)
    Ct = zero(c)
    fm1 = PaddedArray(2Kxr[1][1],length(Kyr[1])+length(Kyr[2]),length(Kzr))
    fm2 = PaddedArray(2Kxr[1][1],length(Kyr[1])+length(Kyr[2]),length(Kzr))
    return ETD3rdO{adp,indt,Hyper}(fm1,fm2,c,At,Bt,Ct,dt,dt2,dt3)
end

get_dt(t::ETD3rdO{true,idt,H}) where {idt,H} = 
    getindex(t.dt)

function initialize!(t::ETD3rdO,rhs::AbstractArray,vis,s::AbstractSimulation)
    mycopy!(data(t.fm1),rhs,s) 
    @inbounds copy!(t.fm2, t.fm1) 
    setindex!(t.dt,get_dt(s))
    setindex!(t.dt2,t.dt[])
    setindex!(t.dt3,t.dt[])
    init_c!(t,t.c,-vis,s)
    return nothing
end

@par function init_c!(t::ETD3rdO{adp,indt,false},c::AbstractArray,mν,s::@par(AbstractSimulation)) where{adp,indt}
    @mthreads for k in Kzr
        for y in Kyr, j in y
            @fastmath @inbounds @msimd for i in 1:(Kxr[k][j])
                c[i,j,k] = muladd(kx[i], kx[i], muladd(ky[j], ky[j], kz[k]*kz[k]))*mν
            end
        end
    end
end

@par function init_c!(t::ETD3rdO{adp,indt,true},c::AbstractArray,aux,s::@par(AbstractSimulation)) where {adp,indt}
    mν::Float64 = -nu(s)
    mνh::Float64 = -nuh(s)
    M::Int = get_hyperviscosity_exponent(s)
    @mthreads for k in Kzr
        for y in Kyr, j in y
            @fastmath @inbounds @msimd for i in 1:(Kxr[k][j])
                modk = muladd(kx[i], kx[i], muladd(ky[j], ky[j], kz[k]*kz[k])) 
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

@par function set_ABCt!(t::ETD3rdO)
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

    if l[end]*t.dt[] <= 1.0
        ind1 = linearindices(At)
        ind2 = 1:0
    else
        i = findfirst(x -> x*dt >= 1., l) - 1
        ind1 = 1:i
        fim = length(l)
        ind2 = (fim-i+1):fim
    end

    for i in ind1
        @inbounds begin
        l1 = l[i]
        l2 = l1*l[i]
        l3 = l2*l[i]
        At[i] = muladd(a1, l1, muladd(a2, l2, muladd(a3, l3, a0)))
        Bt[i] = muladd(b1, l1, muladd(b2, l2, muladd(b3, l3, b0)))
        Ct[i] = muladd(c1, l1, muladd(c2, l2, muladd(c3, l3, c0)))
        end
    end

    e1 = dt^2 + dt*(2dt2+dt3)
    e2 = dt2*(dt2+dt3)
    e3 = 2dt
    e4 = 2dt2+dt3

    e5 = dt*(dt+dt2+dt3)
    e6 = dt2+dt3
    e7 = dt2*dt3
    e8 = dt*(dt+dt2)
    e9 = dt3*(dt2+dt3)
    for i in ind2
        @inbounds begin
        l1 = l[i]
        ldt = l1*dt
        l2 = l1*l1
        l3 = l2*l1

        At[i] = (2expm1(ldt) - l2*(e1 - e2*expm1(ldt)) - l1*(e3 - e4*expm1(ldt))) / (l3*e2)
        Bt[i] = (e5*l2 - 2expm1(ldt) + l1*(e3 - e6*expm1(ldt)) )/(l3*e7)
        Ct[i] = (2expm1(ldt) - e8*l2 - l1*(e3 + dt2*expm1(ldt))) / (l3*e9)
        end
    end
    return nothing
end

@inline function get_ats(t::ETD3rdO{true,indt,H}) where {indt,H}
    dt = t.dt[]
    dt2 = t.dt2[]
    dt3 = t.dt3[]
    a0 = (dt*(2*dt*dt + 6dt2*(dt2 + dt3) + 3dt*(2dt2 + dt3)))/(6*dt2*(dt2+dt3))
    a1 = (dt*dt*(dt^2 + 6dt2*(dt2 + dt3) + 2dt*(2dt2 + dt3)))/(12dt2*(dt2 + dt3))
    a2 = (dt*dt*dt*(2*dt^2 + 20dt2*(dt2 + dt3) + 5dt*(2dt2 + dt3)))/(120dt2*(dt2 + dt3))
    a3 = (dt*dt*dt*dt*(dt^2 + 15dt2*(dt2 + dt3) + 3dt*(2dt2 + dt3)))/(360dt2*(dt2 + dt3))
    return a0,a1,a2,a3
end

@inline function get_bts(t::ETD3rdO{true,indt,H}) where {indt,H}
    dt = t.dt[]
    dt2 = t.dt2[]
    dt3 = t.dt3[]
    b0 = -((dt^2 * (2dt + 3*(dt2 + dt3)))/(6*(dt2*dt3)))
    b1 = -(((dt*dt*dt*(dt + 2*(dt2 + dt3))))/(12*(dt2*dt3)))
    b2 = -((dt*dt*dt*dt*(2dt + 5*(dt2 + dt3)))/(120*(dt2*dt3)))
    b3 = -(((dt*dt*dt*dt*(dt + 3*(dt2 + dt3))))/(360*(dt2*dt3)))
    return b0,b1,b2,b3
end

@inline function get_cts(t::ETD3rdO{true,indt,H}) where {indt,H}
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
    @mthreads for kk = 1:length(Kzr)
        _tETD3rdO!(kk, ρ,ρrhs,f.fm1,f.fm2,f,s)
    end
    return nothing
end

@inline @par function _tETD3rdO!(kk::Integer, u::AbstractArray{T,3}, rhs::AbstractArray, rm1::AbstractArray, rm2::AbstractArray, f, s::@par(AbstractSimulation)) where {T}
@inbounds begin
    At = T === Float64 ? data(f.At) : f.At
    Bt = T === Float64 ? data(f.Bt) : f.Bt
    Ct = T === Float64 ? data(f.Ct) : f.Ct
    c = T === Float64 ? data(f.c) : f.c
    dt = get_dt(f)
    k = Kzr[kk]
    jj::Int = 1
    for y in Kyr, j in y
        @fastmath @msimd for i in 1:((T === Float64 ? 2 : 1)*Kxr[k][j])
            u[i,j,k] = muladd(At[i,j,k], rhs[i,j,k], muladd(Bt[i,j,k], rm1[i,jj,kk], muladd(Ct[i,j,k], rm2[i,jj,kk], exp(c[i,j,k]*dt)*u[i,j,k])))
            rm2[i,jj,kk] = rm1[i,jj,kk]
            rm1[i,jj,kk] = rhs[i,j,k]
        end
        jj+=1
    end
end
end

# with forcing
@par function (f::ETD3rdO)(ρ::AbstractArray{<:Complex,3},ρrhs::AbstractArray{<:Complex,3}, forcing::AbstractArray{<:Complex,3}, s::@par(AbstractSimulation))
    @mthreads for kk = 1:length(Kzr)
        _tETD3rdO!(kk, ρ,ρrhs, forcing, f.fm1,f.fm2,f,s)
    end
    return nothing
end

@inline @par function _tETD3rdO!(kk::Integer, u::AbstractArray{Complex{Float64},3}, rhs::AbstractArray, forcing, rm1::AbstractArray, rm2::AbstractArray, f,s::@par(AbstractSimulation)) 
@inbounds begin
    At = f.At
    Bt = f.Bt
    Ct = f.Ct
    c = f.c
    dt = get_dt(f)
    @inbounds k = Kzr[kk]
    jj::Int = 1
    if (6 < k < Nz-k+1)
        for y in Kyr, j in y
            @fastmath @msimd for i in 1:(Kxr[k][j])
                u[i,j,k] = muladd(At[i,j,k], rhs[i,j,k], muladd(Bt[i,j,k], rm1[i,jj,kk], muladd(Ct[i,j,k], rm2[i,jj,kk], exp(c[i,j,k]*dt)*u[i,j,k])))
                rm2[i,jj,kk] = rm1[i,jj,kk]
                rm1[i,jj,kk] = rhs[i,j,k]
            end
        jj+=1
        end
    else
        for y in Kyr, j in y
            @fastmath @msimd for i in 1:(Kxr[k][j])
                u[i,j,k] = muladd(At[i,j,k], rhs[i,j,k], muladd(Bt[i,j,k], rm1[i,jj,kk], muladd(Ct[i,j,k], rm2[i,jj,kk],muladd(exp(c[i,j,k]*dt), u[i,j,k], forcing[i,j,k]))))
                rm2[i,jj,kk] = rm1[i,jj,kk]
                rm1[i,jj,kk] = rhs[i,j,k]
            end
        jj+=1
        end
    end
end
end
