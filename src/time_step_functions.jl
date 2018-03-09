# Structs Start

  abstract type AbstractTimeStep end
  abstract type AbstractScalarTimeStep{Adp,initdt,N} <: AbstractTimeStep end

  has_variable_timestep(t::Type{<:AbstractScalarTimeStep{adp,idt,N}}) where {adp,idt,N} = adp
  @inline has_variable_timestep(t::AbstractScalarTimeStep{adp,idt,N}) where {adp,idt,N} = has_variable_timestep(typeof(t))
  
  get_dt(t::Type{<:AbstractScalarTimeStep{false,idt,N}}) where {idt,N} = idt
  @inline get_dt(t::AbstractScalarTimeStep{false,idt,N}) where {idt,N} = get_dt(typeof(t))

  set_dt!(t::Type{<:AbstractScalarTimeStep{false,idt,N}},dt::Real) where {idt,N} = nothing
  @inline set_dt!(t::AbstractScalarTimeStep{false,idt,N},dt::Real) where {idt,N} = set_dt!(typeof(t))

  struct VectorTimeStep{cfl,Tx<:AbstractScalarTimeStep,Ty<:AbstractScalarTimeStep,Tz<:AbstractScalarTimeStep} <: AbstractTimeStep
    x::Tx
    y::Ty 
    z::Tz
  end

  function VectorTimeStep{cfl}(tx,ty,tz) where {cfl}
    return VectorTimeStep{cfl,typeof(tx),typeof(ty),typeof(tz)}(tx,ty,tz) 
  end
 
  get_cfl(t::Type{VectorTimeStep{cfl,Tx,Ty,Tz}}) where{cfl,Tx,Ty,Tz} = cfl
  @inline get_cfl(t::VectorTimeStep) = get_cfl(typeof(t))
  @par get_cfl(s::Union{@par(AbstractSimulation),Type{@par(AbstractSimulation)}}) = get_cfl(VelocityTimeStepType)

  function initialize!(t::VectorTimeStep,rhs::VectorField,s::AbstractSimulation)
    initialize!(t.x,rhs.rx,s)
    initialize!(t.y,rhs.ry,s)
    initialize!(t.z,rhs.rz,s)
  end

  function set_dt!(t::VectorTimeStep,dt)
    set_dt!(t.x,dt)
    set_dt!(t.y,dt)
    set_dt!(t.z,dt)
  end

  @inline get_dt(t::A) where {A<:VectorTimeStep} = get_dt(t.x)
  @inline get_dt(s::A) where {A<:AbstractSimulation} = get_dt(s.timestep)

  has_variable_timestep(t::Type{VectorTimeStep{cfl,Tx,Ty,Tz}}) where {cfl,Tx,Ty,Tz} = has_variable_timestep(Tx)
  @inline has_variable_timestep(t::VectorTimeStep) = has_variable_timestep(typeof(t))
  @inline @par has_variable_timestep(s::Type{A}) where {A<:@par(AbstractSimulation)} = has_variable_timestep(VelocityTimeStepType)
  @inline has_variable_timestep(s::A) where {A<:AbstractSimulation} = has_variable_timestep(A)

  function (f::VectorTimeStep)(u::VectorField,rhs::VectorField,s::AbstractSimulation)
    if hasforcing(s)
      f.x(u.rx,rhs.rx,parent(real(s.forcing.forcex)),s)
      f.y(u.ry,rhs.ry,parent(real(s.forcing.forcey)),s)
      if typeof(s.forcing) <: RfForcing
        f.z(u.rz,rhs.rz,s)
      else
        f.z(u.rz,rhs.rz,parent(real(s.forcing.forcez)),s)
      end
    else
      f.x(u.rx,rhs.rx,s)
      f.y(u.ry,rhs.ry,s)
      f.z(u.rz,rhs.rz,s)
    end
  end

  struct Euller{Adptative,initdt} <: AbstractScalarTimeStep{Adptative,initdt,0}
    dt::Base.RefValue{Float64} 
  end

  function initialize!(t::Euller,rhs,s)
    set_dt!(t,get_dt(s))
    return nothing
  end
  
  get_dt(t::Euller{true,idt}) where {idt} = getindex(t.dt)

  set_dt!(t::Euller{true,idt},dt::Real) where {idt} = setindex!(t.dt,dt)

  #initialize!(t::Euller,rhs,s::AbstractSimulation) = nothing

  struct Adams_Bashforth3rdO{Adaptative,initdt} <: AbstractScalarTimeStep{Adaptative,initdt,2}
    fm1::Array{Float64,3} #Store latest step
    fm2::Array{Float64,3} #Store 2 steps before
    At::Base.RefValue{Float64} # 23*dt/12 for constant time stepping
    Bt::Base.RefValue{Float64} # -16*dt/12 for constant time stepping
    Ct::Base.RefValue{Float64} # 5*dt/12 for constant time stepping
    dt::Base.RefValue{Float64}
    dt2::Base.RefValue{Float64}
    dt3::Base.RefValue{Float64}
  end

  function Adams_Bashforth3rdO{adp,indt}(Kxr,Kyr,Kzr) where {adp,indt}
    at = Ref(23*indt/12)
    bt = Ref(-16*indt/12)
    ct = Ref(5*indt/12)
    dt = Ref(indt)
    dt2 = Ref(indt)
    dt3 = Ref(indt)
    fm1 = zeros(2Kxr[1][1],length(Kyr[1])+length(Kyr[2]),length(Kzr))
    fm2 = zeros(2Kxr[1][1],length(Kyr[1])+length(Kyr[2]),length(Kzr))
    return Adams_Bashforth3rdO{adp,indt}(fm1,fm2,at,bt,ct,dt,dt2,dt3)
  end

  get_dt(t::Adams_Bashforth3rdO{true,idt}) where {idt} = getindex(t.dt)

  function initialize!(t::Adams_Bashforth3rdO,rhs::AbstractArray,s::AbstractSimulation)
    mycopy!(t.fm1,rhs,s) 
    @inbounds copy!(t.fm2, t.fm1) 
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

  get_At(t::Type{Adams_Bashforth3rdO{false,idt}}) where {idt} = 23*idt/12
  @inline get_At(t::A) where {indt,A<:Adams_Bashforth3rdO{false,indt}} = get_At(A)

  function get_Bt(t::Adams_Bashforth3rdO{true,idt}) where {idt} 
    dt = t.dt[]
    dt2 = t.dt2[]
    dt3 = t.dt3[]

    Bt= -dt*dt*(2*dt + 3*dt2 + 3*dt3)/(6*dt2*dt3*(dt+dt2))
    Bt*=-(dt+dt2)
    return Bt
  end

  get_Bt(t::Type{Adams_Bashforth3rdO{false,idt}}) where {idt} = -16*idt/12
  @inline get_Bt(t::A) where {indt,A<:Adams_Bashforth3rdO{false,indt}} = get_Bt(A)

  function get_Ct(t::Adams_Bashforth3rdO{true,idt}) where {idt} 
    dt = t.dt[]
    dt2 = t.dt2[]
    dt3 = t.dt3[]

    Ct= dt*dt*(2*dt + 3*dt2)/(6*dt3*(dt + dt2 + dt3)*(dt2 + dt3))
    Ct*=(dt+dt2+dt3)
    return Ct
  end

  get_Ct(t::Type{Adams_Bashforth3rdO{false,idt}}) where {idt} = 5*idt/12
  @inline get_Ct(t::A) where {indt,A<:Adams_Bashforth3rdO{false,indt}} = get_Ct(A)

#Struct End


# Implementations

  @par function set_dt!(s::@par(AbstractSimulation))
    umax = maximum(s.reduction)
    dx = 2π*Ly/Ny
    cfl = get_cfl(s)
    νt = nu(s)
    if hasles(s)
      tnu = maximum(s.lesmodel.reduction)
      νt += tnu
    end
    newdt = min(cfl * dx/umax, cfl * (2νt/umax^2)/10)
    if hasdensity(s) 
      ρmax = maximum(s.densitystratification.reduction)
      g = abs(gravity(s))
      k = diffusivity(s)
      if hasles(s)
        k += tnu
      end
      newdt = min(newdt, 
        cfl * sqrt(dx/(ρmax*g))/10,
        cfl * (dx^2)/(k)/10, 
        cfl * (((ρmax*g)^(-2/3))*(2k)^(1/3))/10,
        cfl * (2k/umax^2)/10)
      set_dt!(s.densitystratification.timestep,newdt)
    end
    set_dt!(s.timestep,newdt)
    haspassivescalar(s) && set_dt!(s.passivescalar.timestep,newdt)
    return nothing
  end

# Euller Start

  @par function (f::Euller)(ρ::AbstractArray{<:Real,3},rhs::AbstractArray{<:Real,3},dt::Real,s::@par(AbstractSimulation))
    @mthreads for k in Kzr
      for y in Kyr, j in y
        @inbounds @fastmath @msimd for i in 1:(2Kxr[k][j])
          #u += dt*rhs
          ρ[i,j,k] = muladd(dt,rhs[i,j,k],ρ[i,j,k])
        end
      end
    end
    return nothing
  end

# Euller End

# Adams_Bashforth3rdO start
  @par function (f::Adams_Bashforth3rdO)(ρ::AbstractArray{<:Real,3},ρrhs::AbstractArray{<:Real,3}, s::@par(AbstractSimulation))
    @mthreads for kk = 1:length(Kzr)
    _tAdams_Bashforth3rdO!(kk, ρ,ρrhs,f.fm1,f.fm2,f,s)
    end
    return nothing
  end
  
  @inline @par function _tAdams_Bashforth3rdO!(kk::Integer, u::AbstractArray{Float64,3}, rhs::AbstractArray, rm1::AbstractArray, rm2::AbstractArray, f, s::@par(AbstractSimulation)) 
    @inbounds begin
      At = get_At(f)
      Bt = get_Bt(f)
      Ct = get_Ct(f)
      k = Kzr[kk]
      jj::Int = 1
      for y in Kyr, j in y
        @fastmath @msimd for i in 1:(2Kxr[k][j])
          #u[i] += dt12*(23*rhs[i] - 16rm1[i] + 5rm2[i])
          #u[i,j,k] = muladd(muladd(23, rhs[i,j,k], muladd(-16, rm1[i,jj,kk], 5rm2[i,jj,kk])), dt12, u[i,j,k])
          u[i,j,k] = muladd(At, rhs[i,j,k], muladd(Bt, rm1[i,jj,kk], muladd(Ct, rm2[i,jj,kk], u[i,j,k])))
          rm2[i,jj,kk] = rm1[i,jj,kk]
          rm1[i,jj,kk] = rhs[i,j,k]
        end
      jj+=1
      end
    end
  end

  # with forcing
  @par function (f::Adams_Bashforth3rdO)(ρ::AbstractArray{<:Real,3},ρrhs::AbstractArray{<:Real,3}, forcing::AbstractArray{<:Real,3}, s::@par(AbstractSimulation))
    @mthreads for kk = 1:length(Kzr)
      _tAdams_Bashforth3rdO!(kk, ρ,ρrhs, forcing, f.fm1,f.fm2,f,s)
    end
    return nothing
  end
  
  @inline @par function _tAdams_Bashforth3rdO!(kk::Integer, u::AbstractArray{Float64,3}, rhs::AbstractArray, forcing, rm1::AbstractArray, rm2::AbstractArray, f,s::@par(AbstractSimulation)) 
    @inbounds begin
      At = get_At(f)
      Bt = get_Bt(f)
      Ct = get_Ct(f)
      @inbounds k = Kzr[kk]
      jj::Int = 1
      if (6 < k < Nz-k+1)
        for y in Kyr, j in y
          @fastmath @msimd for i in 1:(2Kxr[k][j])
            #u[i] += dt12*(23*rhs[i] - 16rm1[i] + 5rm2[i])
            u[i,j,k] = muladd(At, rhs[i,j,k], muladd(Bt, rm1[i,jj,kk], muladd(Ct, rm2[i,jj,kk], u[i,j,k])))
            rm2[i,jj,kk] = rm1[i,jj,kk]
            rm1[i,jj,kk] = rhs[i,j,k]
          end
        jj+=1
        end
      else
        for y in Kyr, j in y
          @fastmath @msimd for i in 1:(2Kxr[k][j])
            #u[i] += dt12*(23*rhs[i] - 16rm1[i] + 5rm2[i])
            u[i,j,k] = muladd(At, rhs[i,j,k], muladd(Bt, rm1[i,jj,kk], muladd(Ct, rm2[i,jj,kk], u[i,j,k]))) + forcing[i,j,k]
            rm2[i,jj,kk] = rm1[i,jj,kk]
            rm1[i,jj,kk] = rhs[i,j,k]
          end
        jj+=1
        end
      end
    end
  end
#
# Adams_Bashforth3rdO end