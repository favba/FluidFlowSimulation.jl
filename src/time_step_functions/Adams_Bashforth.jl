struct Adams_Bashforth3rdO{Adaptative,initdt} <: AbstractScalarTimeStep{Adaptative,initdt,2}
    fm1::PaddedArray{Float64,3,2,false} #Store latest step
    fm2::PaddedArray{Float64,3,2,false} #Store 2 steps before
    At::Base.RefValue{Float64} # 23*dt/12 for constant time stepping
    Bt::Base.RefValue{Float64} # -16*dt/12 for constant time stepping
    Ct::Base.RefValue{Float64} # 5*dt/12 for constant time stepping
    dt::Base.RefValue{Float64}
    dt2::Base.RefValue{Float64}
    dt3::Base.RefValue{Float64}
end

function Adams_Bashforth3rdO{adp,indt}() where {adp,indt}
    at = Ref(23*indt/12)
    bt = Ref(-16*indt/12)
    ct = Ref(5*indt/12)
    dt = Ref(indt)
    dt2 = Ref(indt)
    dt3 = Ref(indt)
    fm1 = PaddedArray{Float64}(Nrx,Ny,Nz)
    fm2 = PaddedArray{Float64}(Nrx,Ny,Nz)
    return Adams_Bashforth3rdO{adp,indt}(fm1,fm2,at,bt,ct,dt,dt2,dt3)
end

get_dt(t::Adams_Bashforth3rdO{true,idt}) where {idt} = 
    getindex(t.dt)

function initialize!(t::Adams_Bashforth3rdO,rhs::AbstractArray,vis,s::AbstractSimulation)
    mycopy!(data(t.fm1), rhs,s) 
    mycopy!(data(t.fm2), data(t.fm1),s) 
    setindex!(t.dt,get_dt(s))
    setindex!(t.dt2,t.dt[])
    setindex!(t.dt3,t.dt[])
    return nothing
end

function set_dt!(t::Adams_Bashforth3rdO{true,idt},dt::Real) where {idt} 
    setindex!(t.dt3,t.dt2[])
    setindex!(t.dt2,t.dt[])
    setindex!(t.dt,dt)
end

function get_At(t::Adams_Bashforth3rdO{true,idt}) where {idt} 
    dt = t.dt[]
    dt2 = t.dt2[]
    dt3 = t.dt3[]

    At = 1. + (dt*(2*dt + 6*dt2 + 3*dt3))/(6*dt2*(dt2 + dt3))
    At *= dt
    return At
end

get_At(t::Type{Adams_Bashforth3rdO{false,idt}}) where {idt} =
    23*idt/12

@inline get_At(t::A) where {indt,A<:Adams_Bashforth3rdO{false,indt}} = 
    get_At(A)

function get_Bt(t::Adams_Bashforth3rdO{true,idt}) where {idt} 
    dt = t.dt[]
    dt2 = t.dt2[]
    dt3 = t.dt3[]

    Bt = -dt*dt*(2*dt + 3*dt2 + 3*dt3)/(6*dt2*dt3*(dt+dt2))
    Bt *= (dt+dt2)
    return Bt
end

get_Bt(t::Type{Adams_Bashforth3rdO{false,idt}}) where {idt} =
    -16*idt/12

@inline get_Bt(t::A) where {indt,A<:Adams_Bashforth3rdO{false,indt}} =
    get_Bt(A)

function get_Ct(t::Adams_Bashforth3rdO{true,idt}) where {idt} 
    dt = t.dt[]
    dt2 = t.dt2[]
    dt3 = t.dt3[]
    
    Ct= dt*dt*(2*dt + 3*dt2)/(6*dt3*(dt + dt2 + dt3)*(dt2 + dt3))
    Ct*=(dt+dt2+dt3)
    return Ct
end
    
get_Ct(t::Type{Adams_Bashforth3rdO{false,idt}}) where {idt} =
    5*idt/12

@inline get_Ct(t::A) where {indt,A<:Adams_Bashforth3rdO{false,indt}} =
    get_Ct(A)

@par function (f::Adams_Bashforth3rdO)(ρ::AbstractArray{<:Complex,3},ρrhs::AbstractArray{<:Complex,3}, s::@par(AbstractSimulation))
    @mthreads for kk = 1:Nz
        _tAdams_Bashforth3rdO!(kk, ρ,ρrhs,f.fm1,f.fm2,f,s)
    end
    return nothing
end

@inline @par function _tAdams_Bashforth3rdO!(k::Integer, u::AbstractArray{Complex{Float64},3}, rhs::AbstractArray, rm1::AbstractArray, rm2::AbstractArray, f, s::@par(AbstractSimulation)) 
@inbounds begin
    At = get_At(f)
    Bt = get_Bt(f)
    Ct = get_Ct(f)
    for j in 1:Ny
        @msimd for i in 1:Nx
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
    @mthreads for kk = 1:Nz
        _tAdams_Bashforth3rdO!(kk, ρ,ρrhs, forcing, f.fm1,f.fm2,f,s)
    end
    return nothing
end

@inline @par function _tAdams_Bashforth3rdO!(k::Integer, u::AbstractArray{Complex{Float64},3}, rhs::AbstractArray, forcing, rm1::AbstractArray, rm2::AbstractArray, f,s::@par(AbstractSimulation)) 
@inbounds begin
    At = get_At(f)
    Bt = get_Bt(f)
    Ct = get_Ct(f)
    if (6 < k < Nz-k+1)
        for j in 1:Ny
            @msimd for i in 1:Nx
                #u[i] += dt12*(23*rhs[i] - 16rm1[i] + 5rm2[i])
                u[i,j,k] = muladd(At, rhs[i,j,k], muladd(Bt, rm1[i,j,k], muladd(Ct, rm2[i,j,k], u[i,j,k])))
                rm2[i,j,k] = rm1[i,j,k]
                rm1[i,j,k] = rhs[i,j,k]
            end
        end
    else
        for j in 1:Ny
            @msimd for i in 1:Nx
                #u[i] += dt12*(23*rhs[i] - 16rm1[i] + 5rm2[i])
                u[i,j,k] = muladd(At, rhs[i,j,k], muladd(Bt, rm1[i,j,k], muladd(Ct, rm2[i,j,k], u[i,j,k]))) + forcing[i,j,k]
                rm2[i,j,k] = rm1[i,j,k]
                rm1[i,j,k] = rhs[i,j,k]
            end
        end
    end
end
end
