@par function advance_in_time!(s::A,init::Int64,Nsteps::Int64,dt::Real,time::Real) where {A<:@par(AbstractParameters)}

  for t=1:Nsteps
    init += 1
    calculate_rhs!(s)
    time_step!(s,dt)
    time += dt
  end

  return init, dt, time
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

  out_transform!(s.aux,s.rhs,s)

  rcross!(s.rhs,s.u,s.aux,s)
  s.p*s.rhs
  dealias!(s.rhs,s)
  if A<:ScalarParameters
    s.ps\s.ρ
    scalar_advection!(s.aux,s.ρ,s.u,s)
    s.p*s.aux
    dealias!(s.aux,s)
    s.ps*s.ρ
    div!(complex(s.ρrhs),s.aux.cx,s.aux.cy,s.aux.cz,s.u.cz,-s.dρdz,s)
  end
  s.p*s.u
  return nothing
end

out_transform!(out::VectorField,in::VectorField,s::AbstractParameters) = A_mul_B!(real(out),s.ip,complex(in))

@inline @par function dealias!(rhs::VectorField,s::@par(AbstractParameters))
  dealias!(rhs.cx,s.dealias,s)
  dealias!(rhs.cy,s.dealias,s)
  dealias!(rhs.cz,s.dealias,s)
end

dealias!(rhs::AbstractArray{<:Complex,3},s::AbstractParameters) = dealias!(rhs,s.dealias,s)

@inline @par function dealias!(rhs::AbstractArray{T,3},dealias,s::@par(AbstractParameters)) where {T<:Complex}
 @mthreads for i = 1:Lcs
  @inbounds begin
    dealias[i] && (rhs[i] = zero(T))
  end
 end
end

function add_viscosity!(rhs::VectorField,u::VectorField,ν::Real,s::AbstractParameters)
  _add_viscosity!(rhs.cx,u.cx,-ν,s)
  _add_viscosity!(rhs.cy,u.cy,-ν,s)
  _add_viscosity!(rhs.cz,u.cz,-ν,s)
end

@par function _add_viscosity!(rhs::AbstractArray,u::AbstractArray,mν::Real,s::@par(AbstractParameters))
  @mthreads for k in Kzr
    for j in Kyr
      @fastmath @inbounds @msimd for i in Kxr
        #rhs[i,j,k] = (kx[i]*kx[i] + ky[j]*ky[j] + kz[k]*kz[k])*mŒΩ*u[i,j,k] + rhs[i,j,k]
        rhs[i,j,k] = muladd(muladd(kx[i], kx[i], muladd(ky[j], ky[j], kz[k]*kz[k])), mν*u[i,j,k], rhs[i,j,k])
      end
    end
  end
end

function add_scalar_difusion!(rhs::AbstractArray,u::AbstractArray,ν::Real,s::AbstractParameters)
  _add_viscosity!(rhs,u,-ν,s)
end

 @par function pressure_projection!(rhsx,rhsy,rhsz,s::@par(AbstractParameters))
  @inbounds a = (rhsx[1],rhsy[1],rhsz[1])
  @mthreads for k in Kzr
    for j in Kyr
      @fastmath @inbounds @msimd for i in Kxr
        #p1 = -(kx[i]*rhsx[i,j,k] + ky[j]*rhsy[i,j,k] + kz[k]*rhsz[i,j,k])/(kx[i]*kx[i] + ky[j]*ky[j] + kz[k]*kz[k])
        p1 = -muladd(kx[i], rhsx[i,j,k], muladd(ky[j], rhsy[i,j,k], kz[k]*rhsz[i,j,k]))/muladd(kx[i], kx[i], muladd(ky[j], ky[j],  kz[k]*kz[k]))
        rhsx[i,j,k] = muladd(kx[i],p1,rhsx[i,j,k])
        rhsy[i,j,k] = muladd(ky[j],p1,rhsy[i,j,k])
        rhsz[i,j,k] = muladd(kz[k],p1,rhsz[i,j,k])
      end
    end
  end
  @inbounds rhsx[1],rhsy[1],rhsz[1] = a
end

@par function addgravity!(rhs,ρ,g::Real,s::@par(BoussinesqParameters))
  @mthreads for k in Kzr
    for j in Kyr
      @fastmath @inbounds @msimd for i in Kxr
        rhs[i,j,k] = muladd(ρ[i,j,k],g,rhs[i,j,k])
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
  @mthreads for kk in 1:length(Kzr)
    k = Kzr[kk]
    for (jj,j) in enumerate(Kyr)
      for i in Kxr
        @inbounds rm[i,jj,kk] = rhs[i,j,k]
      end
    end
  end
end

@par function mycopy!(rm::AbstractArray{T,3},rhs::AbstractArray{T,3},s::@par(AbstractParameters)) where T<:Real
  @mthreads for kk in 1:length(Kzr)
    k = Kzr[kk]
    for (jj,j) in enumerate(Kyr)
      for i in 1:(2length(Kxr))
        @inbounds rm[i,jj,kk] = rhs[i,j,k]
      end
    end
  end
end
