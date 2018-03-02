# Structs Start

  abstract type AbstractTimeStep end
  abstract type AbstractScalarTimeStep{N} <: AbstractTimeStep end

  struct VectorTimeStep{Tx<:AbstractScalarTimeStep,Ty<:AbstractScalarTimeStep,Tz<:AbstractScalarTimeStep} <: AbstractTimeStep
    x::Tx
    y::Ty 
    z::Tz
  end

  function initialize!(t::VectorTimeStep,rhs::VectorField,s::AbstractSimulation)
    initialize!(t.x,rhs.rx,s)
    initialize!(t.y,rhs.ry,s)
    initialize!(t.z,rhs.rz,s)
  end


  function (f::VectorTimeStep)(u::VectorField,rhs::VectorField,dt::Real,s::AbstractSimulation)
    if hasforcing(s)
      f.x(u.rx,rhs.rx,parent(real(s.forcing.forcex)), dt,s)
      f.y(u.ry,rhs.ry,parent(real(s.forcing.forcey)), dt,s)
      if typeof(s.forcing) <: RfForcing
        f.z(u.rz,rhs.rz,dt,s)
      else
        f.z(u.rz,rhs.rz,parent(real(s.forcing.forcez)), dt,s)
      end
    else
      f.x(u.rx,rhs.rx,dt,s)
      f.y(u.ry,rhs.ry,dt,s)
      f.z(u.rz,rhs.rz,dt,s)
    end
  end

  function (::Type{T})(Kxr,Kyr,Kzr) where {N,T<:AbstractScalarTimeStep{N}}
    if N === 0
      return T()
    else
      arrays = Array{Float64,3}[]
      for i = 1:N
        push!(arrays,zeros(2length(Kxr),length(Kyr[1])+length(Kyr[2]),length(Kzr)))
      end
    end
    return T(arrays...)
  end

  #@inline function (f::(T where {T<:AbstractScalarTimeStep{N} where N}))(ρ::PaddedArray,rhs::PaddedArray,dt::Real,s::AbstractSimulation)
  #  return f(parent(real(ρ)),parent(real(rhs)),dt,s)
  #end
  #

  struct Euller <: AbstractScalarTimeStep{0} end

  #initialize!(t::Euller,rhs,s::AbstractSimulation) = nothing

  struct Adams_Bashforth3rdO <: AbstractScalarTimeStep{2}
    fm1::Array{Float64,3} #Store latest step
    fm2::Array{Float64,3} #Store 2 steps before
  end

  @generated function initialize!(t::AbstractScalarTimeStep{N},rhs,s::AbstractSimulation) where N
    if N === 0 
      return :(nothing)
    elseif N === 1
      return :(mycopy!(t.fm1,rhs,s); return nothing)
    else
      blk= Expr(:block)
      push!(blk.args,:(mycopy!(t.fm1,rhs,s)))
      for i=2:N
        fmn = Symbol("fm",i) 
        fmnm1 = Symbol("fm",i-1) 
        push!(blk.args,:(@inbounds copy!(t.$(fmn), t.$(fmnm1))))
      end
      push!(blk.args,:(nothing))
      return blk
    end
  end

#Struct End


# Implementations

# Euller Start

  @par function (f::Euller)(ρ::AbstractArray{<:Real,3},rhs::AbstractArray{<:Real,3},dt::Real,s::@par(AbstractSimulation))
    @mthreads for k in Kzr
      for y in Kyr, j in y
        @inbounds @fastmath @msimd for i in 1:(2Kxr[j][k])
          #u += dt*rhs
          ρ[i,j,k] = muladd(dt,rhs[i,j,k],ρ[i,j,k])
        end
      end
    end
    return nothing
  end

# Euller End

# Adams_Bashforth3rd0 start
  function (f::Adams_Bashforth3rdO)(ρ::AbstractArray{<:Real,3},ρrhs::AbstractArray{<:Real,3}, dt::Real, s::AbstractSimulation)
    dt12 = dt/12
    _tAdams_Bashforth3rdO!(ρ,ρrhs,dt12,f.fm1,f.fm2,s)
    return nothing
  end
  
  @inline @inbounds @par function _tAdams_Bashforth3rdO!(u::AbstractArray{Float64,3}, rhs::AbstractArray, dt12::Real, rm1::AbstractArray, rm2::AbstractArray, s::@par(AbstractSimulation)) 
    @mthreads for kk in 1:length(Kzr)
      k = Kzr[kk]
      jj::Int = 1
      for y in Kyr, j in y
        @fastmath @msimd for i in 1:(2Kxr[j][k])
          #u[i] += dt12*(23*rhs[i] - 16rm1[i] + 5rm2[i])
          u[i,j,k] = muladd(muladd(23, rhs[i,j,k], muladd(-16, rm1[i,jj,kk], 5rm2[i,jj,kk])), dt12, u[i,j,k])
          rm2[i,jj,kk] = rm1[i,jj,kk]
          rm1[i,jj,kk] = rhs[i,j,k]
        end
      jj+=1
      end
    end
  end

  # with forcing
  @par function (f::Adams_Bashforth3rdO)(ρ::AbstractArray{<:Real,3},ρrhs::AbstractArray{<:Real,3}, forcing::AbstractArray{<:Real,3}, dt::Real, s::@par(AbstractSimulation))
    dt12 = dt/12
    for kk = 1:length(Kzr)
      _tAdams_Bashforth3rdO!(kk, ρ,ρrhs, forcing, dt12,f.fm1,f.fm2,s)
    end
    return nothing
  end
  
  @inline @inbounds @par function _tAdams_Bashforth3rdO!(kk::Integer, u::AbstractArray{Float64,3}, rhs::AbstractArray, forcing, dt12::Real, rm1::AbstractArray, rm2::AbstractArray, s::@par(AbstractSimulation)) 
    k = Kzr[kk]
    jj::Int = 1
    if (6 < k < Nz-k+1)
      for y in Kyr, j in y
        @fastmath @msimd for i in 1:(2Kxr[j][k])
          #u[i] += dt12*(23*rhs[i] - 16rm1[i] + 5rm2[i])
          u[i,j,k] = muladd(muladd(23, rhs[i,j,k], muladd(-16, rm1[i,jj,kk], 5rm2[i,jj,kk])), dt12, u[i,j,k])
          rm2[i,jj,kk] = rm1[i,jj,kk]
          rm1[i,jj,kk] = rhs[i,j,k]
        end
      jj+=1
      end
    else
      for y in Kyr, j in y
        @fastmath @msimd for i in 1:(2Kxr[j][k])
          #u[i] += dt12*(23*rhs[i] - 16rm1[i] + 5rm2[i])
          u[i,j,k] = muladd(muladd(23, rhs[i,j,k], muladd(-16, rm1[i,jj,kk], 5rm2[i,jj,kk])), dt12, u[i,j,k]) + forcing[i,j,k]
          rm2[i,jj,kk] = rm1[i,jj,kk]
          rm1[i,jj,kk] = rhs[i,j,k]
        end
      jj+=1
      end
    end
  end
#
# Adams_Bashforth3rd0 end