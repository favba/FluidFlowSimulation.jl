@par function advance_in_time!(s::A,init::Int64,Nsteps::Int64,dt::Float64) where {A<:@par(AbstractParameters)}

  s.p*s.u
  A <: ScalarParameters && s.ps*s.ρ

  if Integrator !== :Euller
    if init==0
      calculate_rhs!(s)
      mycopy!(s.rm2x,s.rhs.rx,s)
      copy!(s.rm1x,s.rm2x)
      mycopy!(s.rm2y,s.rhs.ry,s)
      copy!(s.rm1y,s.rm2y)
      mycopy!(s.rm2z,s.rhs.rz,s)
      copy!(s.rm1z,s.rm2z)
      if A <: ScalarParameters
        mycopy!(s.rrm1,rawreal(s.ρrhs),s)
        copy!(s.rrm2,s.rrm1)
      end
    end
  end

  for t=1:Nsteps
    init += 1
    calculate_rhs!(s)
    time_step!(s,dt)
  end

  s.p\s.u
  A <: ScalarParameters && s.ps\s.ρ
  return init
end

function calculate_rhs!(s::A) where {A<:AbstractParameters}
  compute_nonlinear!(s)
  add_viscosity!(s.rhs,s.u,s.ν,s)
  A<:BoussinesqParameters && addgravity!(s.rhs.cz,complex(s.ρ),-s.g,s)
  pressure_projection!(s.rhs.cx,s.rhs.cy,s.rhs.cz,s)
  A<:ScalarParameters && add_scalar_difusion!(complex(s.ρrhs),complex(s.ρ),s.α,s)
end

function compute_nonlinear!(s::A) where {A<:AbstractParameters}
  curl!(s.rhs,s.u,s)
  s.p\s.u

  A_mul_B!(real(s.aux),s.ip,complex(s.rhs))

  rcross!(s.rhs,s.u,s.aux,s)
  s.p*s.rhs
  dealias!(complex(s.rhs),s)
  if A<:ScalarParameters
    s.ps\s.ρ
    scalar_advection!(s.aux,s.ρ,s.u,s)
    s.p*s.aux
    dealias!(complex(s.aux),s)
    s.ps*s.ρ
    div!(complex(s.ρrhs),s.aux.cx,s.aux.cy,s.aux.cz,s.u.cz,-s.dρdz,s)
  end
  s.p*s.u
  return nothing
end

@inline @par function dealias!(rhs::AbstractArray{T,4},s::@par(AbstractParameters)) where {T<:Complex}
 @inbounds rhs[s.dealias] = zero(T)
end

function add_viscosity!(rhs::VectorField,u::VectorField,ν::Real,s::AbstractParameters)
  _add_viscosity!(rhs.cx,u.cx,-ν,s)
  _add_viscosity!(rhs.cy,u.cy,-ν,s)
  _add_viscosity!(rhs.cz,u.cz,-ν,s)
end

@par function _add_viscosity!(rhs::AbstractArray,u::AbstractArray,mν::Real,s::@par(AbstractParameters))
  Threads.@threads for k in Kzr
    for j in Kyr
      @simd for i in Kxr
        #@inbounds rhs[i,j,k] = (kx[i]*kx[i] + ky[j]*ky[j] + kz[k]*kz[k])*mŒΩ*u[i,j,k] + rhs[i,j,k]
        @inbounds rhs[i,j,k] = muladd(muladd(kx[i], kx[i], muladd(ky[j], ky[j], kz[k]*kz[k])), mν*u[i,j,k], rhs[i,j,k])
      end
    end
  end
end

function add_scalar_difusion!(rhs::AbstractArray,u::AbstractArray,ν::Real,s::AbstractParameters)
  _add_viscosity!(rhs,u,-ν,s)
end

@fastmath @par function pressure_projection!(rhsx,rhsy,rhsz,s::@par(AbstractParameters))
  @inbounds a = (rhsx[1],rhsy[1],rhsz[1])
  Threads.@threads for k in Kzr
    for j in Kyr
      for i in Kxr
        #@inbounds p1 = -(kx[i]*rhsx[i,j,k] + ky[j]*rhsy[i,j,k] + kz[k]*rhsz[i,j,k])/(kx[i]*kx[i] + ky[j]*ky[j] + kz[k]*kz[k])
        @inbounds p1 = -muladd(kx[i], rhsx[i,j,k], muladd(ky[j], rhsy[i,j,k], kz[k]*rhsz[i,j,k]))/muladd(kx[i], kx[i], muladd(ky[j], ky[j],  kz[k]*kz[k]))
        @inbounds rhsx[i,j,k] = muladd(kx[i],p1,rhsx[i,j,k])
        @inbounds rhsy[i,j,k] = muladd(ky[j],p1,rhsy[i,j,k])
        @inbounds rhsz[i,j,k] = muladd(kz[k],p1,rhsz[i,j,k])
      end
    end
  end
  @inbounds rhsx[1],rhsy[1],rhsz[1] = a
end

@par function addgravity!(rhs,ρ,g::Real,s::@par(BoussinesqParameters))
  Threads.@threads for k in Kzr
    for j in Kyr
      for i in Kxr
        @inbounds rhs[i,j,k] = muladd(ρ[i,j,k],g,rhs[i,j,k])
      end
    end
  end
end

@par function time_step!(s::A,dt::Real) where {A<:@par(AbstractParameters)}
  if Integrator == :Euller
    Euller!(rawreal(s.u),rawreal(s.rhs),dt,s)
    A <: ScalarParameters && Euller!(complex(s.ρ),complex(s.ρrhs),dt,s)
  elseif Integrator == :Adams_Bashforth3rdO
    Adams_Bashforth3rdO!(dt,s)
  end
end

@par function mycopy!(rm::AbstractArray{T,3},rhs::AbstractArray{T,3},s::@par(AbstractParameters)) where T<:Complex
  for (kk,k) in enumerate(Kzr)
    for (jj,j) in enumerate(Kyr)
      for i in Kxr
        @inbounds rm[i,jj,kk] = rhs[i,j,k]
      end
    end
  end
end

@par function mycopy!(rm::AbstractArray{T,3},rhs::AbstractArray{T,3},s::@par(AbstractParameters)) where T<:Real
  for (kk,k) in enumerate(Kzr)
    for (jj,j) in enumerate(Kyr)
      for i in 1:(2length(Kxr))
        @inbounds rm[i,jj,kk] = rhs[i,j,k]
      end
    end
  end
end