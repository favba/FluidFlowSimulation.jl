struct Adams_Bashforth3rdO{Adaptative} <: AbstractScalarTimeStep{Adaptative,2}
    fm1::ScalarField{Float64,3,2,false} #Store latest step
    fm2::ScalarField{Float64,3,2,false} #Store 2 steps before
    At::Base.RefValue{Float64} # 23*dt/12 for constant time stepping
    Bt::Base.RefValue{Float64} # -16*dt/12 for constant time stepping
    Ct::Base.RefValue{Float64} # 5*dt/12 for constant time stepping
    dt::Base.RefValue{Float64}
    dt2::Base.RefValue{Float64}
    dt3::Base.RefValue{Float64}
    iteration::Base.RefValue{Int}
end

function Adams_Bashforth3rdO(adp,indt)
    at = Ref(23*indt/12)
    bt = Ref(-16*indt/12)
    ct = Ref(5*indt/12)
    dt = Ref(indt)
    dt2 = Ref(indt)
    dt3 = Ref(indt)
    fm1 = ScalarField{Float64}((NRX,NY,NZ),(LX,LY,LZ))
    fm2 = ScalarField{Float64}((NRX,NY,NZ),(LX,LY,LZ))
    return Adams_Bashforth3rdO{adp}(fm1,fm2,at,bt,ct,dt,dt2,dt3,Ref(0))
end

get_dt(t::Adams_Bashforth3rdO) = 
    getindex(t.dt)

function initialize!(t::Adams_Bashforth3rdO,rhs::AbstractArray,vis,s::AbstractSimulation)
    mycopy!(data(t.fm1), rhs) 
    mycopy!(data(t.fm2), data(t.fm1)) 
    setindex!(t.dt,get_dt(s))
    setindex!(t.dt2,t.dt[])
    setindex!(t.dt3,t.dt[])
    return nothing
end

function set_dt!(t::Adams_Bashforth3rdO{true},dt::Real)
    if t.iteration[] == 0
        setindex!(t.dt3,dt/2)
        setindex!(t.dt2,dt/2)
        setindex!(t.dt,dt/2)
    else
        setindex!(t.dt3,t.dt2[])
        setindex!(t.dt2,t.dt[])
        setindex!(t.dt,dt)
    end
end

function get_At(t::Adams_Bashforth3rdO{true})
    dt = t.dt[]
    dt2 = t.dt2[]
    dt3 = t.dt3[]

    At = 1. + (dt*(2*dt + 6*dt2 + 3*dt3))/(6*dt2*(dt2 + dt3))
    At *= dt
    return At
end

get_At(t::Type{Adams_Bashforth3rdO{false}}) =
    23*get_dt(t)/12

@inline get_At(t::A) where {A<:Adams_Bashforth3rdO{false}} = 
    get_At(A)

function get_Bt(t::Adams_Bashforth3rdO{true})
    dt = t.dt[]
    dt2 = t.dt2[]
    dt3 = t.dt3[]

    Bt = -dt*dt*(2*dt + 3*dt2 + 3*dt3)/(6*dt2*dt3*(dt+dt2))
    Bt *= (dt+dt2)
    return Bt
end

get_Bt(t::Type{Adams_Bashforth3rdO{false}}) =
    -16*get_dt(t)/12

@inline get_Bt(t::A) where {A<:Adams_Bashforth3rdO{false}} =
    get_Bt(A)

function get_Ct(t::Adams_Bashforth3rdO{true})
    dt = t.dt[]
    dt2 = t.dt2[]
    dt3 = t.dt3[]
    
    Ct = dt*dt*(2*dt + 3*dt2)/(6*dt3*(dt + dt2 + dt3)*(dt2 + dt3))
    Ct*=(dt+dt2+dt3)
    return Ct
end
    
get_Ct(t::Type{Adams_Bashforth3rdO{false}}) =
    5*get_det(t)/12

@inline get_Ct(t::A) where {A<:Adams_Bashforth3rdO{false}} =
    get_Ct(A)

function set_coefficients!(t::Adams_Bashforth3rdO,rhs)
    i = t.iteration[] += 1
    i == 2 && fix_fm2!(t.fm2,t.fm1,rhs,t.dt2[],t.dt3[])
end

@par function (f::Adams_Bashforth3rdO)(ρ::AbstractArray{<:Complex,3},ρrhs::AbstractArray{<:Complex,3}, s::@par(AbstractSimulation))
    set_coefficients!(f,ρrhs)
    @mthreads for kk in ZRANGE
        _tAdams_Bashforth3rdO!(kk, ρ,ρrhs,f.fm1,f.fm2,f,s)
    end
    return nothing
end

@inline @par function _tAdams_Bashforth3rdO!(k::Integer, u::AbstractArray{Complex{Float64},3}, rhs::AbstractArray, rm1::AbstractArray, rm2::AbstractArray, f, s::@par(AbstractSimulation)) 
@inbounds begin
    At = get_At(f)
    Bt = get_Bt(f)
    Ct = get_Ct(f)
    for j in YRANGE
        @msimd for i in XRANGE
            #u[i] += dt12*(23*rhs[i] - 16rm1[i] + 5rm2[i])
            #u[i,j,k] = muladd(muladd(23, rhs[i,j,k], muladd(-16, rm1[i,jj,kk], 5rm2[i,jj,kk])), dt12, u[i,j,k])
            u[i,j,k] = muladd(At, rhs[i,j,k], muladd(Bt, rm1[i,j,k], muladd(Ct, rm2[i,j,k], u[i,j,k])))
            rm2[i,j,k] = rm1[i,j,k]
            rm1[i,j,k] = rhs[i,j,k]
        end
    end
end
end

# with forcing
@par function (f::Adams_Bashforth3rdO)(ρ::AbstractArray{<:Complex,3},ρrhs::AbstractArray{<:Complex,3}, forcing::AbstractArray{<:Complex,3}, s::@par(AbstractSimulation))
    set_coefficients!(f,ρrhs)
    @mthreads for kk in ZRANGE
        _tAdams_Bashforth3rdO!(kk, ρ,ρrhs, forcing, f.fm1,f.fm2,f,s)
    end
    return nothing
end

@inline @par function _tAdams_Bashforth3rdO!(k::Integer, u::AbstractArray{Complex{Float64},3}, rhs::AbstractArray, forcing, rm1::AbstractArray, rm2::AbstractArray, f,s::@par(AbstractSimulation)) 
@inbounds begin
    At = get_At(f)
    Bt = get_Bt(f)
    Ct = get_Ct(f)
    if (6 < k < NZ-4)
        for j in YRANGE
            @msimd for i in XRANGE
                #u[i] += dt12*(23*rhs[i] - 16rm1[i] + 5rm2[i])
                u[i,j,k] = muladd(At, rhs[i,j,k], muladd(Bt, rm1[i,j,k], muladd(Ct, rm2[i,j,k], u[i,j,k])))
                rm2[i,j,k] = rm1[i,j,k]
                rm1[i,j,k] = rhs[i,j,k]
            end
        end
    else
        for j in YRANGE
            @msimd for i in XRANGE
                #u[i] += dt12*(23*rhs[i] - 16rm1[i] + 5rm2[i])
                u[i,j,k] = muladd(At, rhs[i,j,k], muladd(Bt, rm1[i,j,k], muladd(Ct, rm2[i,j,k], u[i,j,k]))) + forcing[i,j,k]
                rm2[i,j,k] = rm1[i,j,k]
                rm1[i,j,k] = rhs[i,j,k]
            end
        end
    end
end
end
