@par function advance_in_time!(s::A,init::Int64,Nsteps::Int64,dt::Float64) where {A<:@par(AbstractParameters)}

  s.p*s.u
  A <: ScalarParameters && s.ps*s.ρ

  if Integrator !== :Euller
    if init==0
      calculate_rhs!(s)
      copy!(s.rm2,rawreal(s.rhs))
      copy!(s.rm1,s.rm2)
      if A <: ScalarParameters
        copy!(s.rrm1,rawreal(s.ρrhs))
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
  add_viscosity!(s.rhs,s.u,s.ν,s.kx,s.ky,s.kz,s)
  A<:BoussinesqParameters && addgravity!(s.rhs.cz,complex(s.ρ),-s.g,s)
  pressure_projection!(s.rhs.cx,s.rhs.cy,s.rhs.cz,s.kx,s.ky,s.kz,s)
  A<:ScalarParameters && add_scalar_difusion!(complex(s.ρrhs),complex(s.ρ),s.α,s.kx,s.ky,s.kz,s)
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
    div!(complex(s.ρrhs),s.kx,s.ky,s.kz,s.aux.cx,s.aux.cy,s.aux.cz,s.u.cz,-s.dρdz,s)
  end
  s.p*s.u
  return nothing
end

@par function dealias!(rhs::AbstractArray{T,4},s::@par(AbstractParameters)) where {T<:Complex}
  @inbounds for l=1:3
  for k in (div(Nz,3)+2):(div(2Nz,3)+1)
    for j in 1:Ny
      for i in 1:Nx
        rhs[i,j,k,l] = zero(T)
      end
    end
  end
  for k in 1:Nz
    for j in (div(Ny,3)+2):(div(2Ny,3)+1)
      for i in 1:Nx
        rhs[i,j,k,l] = zero(T)
      end
    end
  end
  for k in 1:Nz
    for j in 1:Ny
      for i in (div(2Nx,3)+1):Nx
        rhs[i,j,k,l] = zero(T)
      end
    end
  end
  end
end

function add_viscosity!(rhs::VectorField,u::VectorField,ν::Real,kx::AbstractArray,ky::AbstractArray,kz::AbstractArray,s::AbstractParameters)
  _add_viscosity!(rhs.cx,u.cx,-ν,kx,ky,kz,s)
  _add_viscosity!(rhs.cy,u.cy,-ν,kx,ky,kz,s)
  _add_viscosity!(rhs.cz,u.cz,-ν,kx,ky,kz,s)
end

@par function _add_viscosity!(rhs::AbstractArray,u::AbstractArray,mν::Real,kx::AbstractArray,ky::AbstractArray,kz::AbstractArray,s::@par(AbstractParameters))
  if Tr
    _tadd_viscosity!(rhs,u,mν,kx,ky,kz,s)
  else
    for k = 1:Nz
      for j = 1:Ny
        for i = 1:Nx
          #@inbounds rhs[i,j,k] = (kx[i]*kx[i] + ky[j]*ky[j] + kz[k]*kz[k])*mν*u[i,j,k] + rhs[i,j,k]
          @inbounds rhs[i,j,k] = muladd(muladd(kx[i], kx[i], muladd(ky[j], ky[j], kz[k]*kz[k])), mν*u[i,j,k], rhs[i,j,k])
        end
      end
    end
  end
end

@par function _tadd_viscosity!(rhs::AbstractArray,u::AbstractArray,mν::Real,kx::AbstractArray,ky::AbstractArray,kz::AbstractArray,s::@par(AbstractParameters))
  Threads.@threads for k = 1:Nz
    for j = 1:Ny
      for i = 1:Nx
        #@inbounds rhs[i,j,k] = (kx[i]*kx[i] + ky[j]*ky[j] + kz[k]*kz[k])*mŒΩ*u[i,j,k] + rhs[i,j,k]
        @inbounds rhs[i,j,k] = muladd(muladd(kx[i], kx[i], muladd(ky[j], ky[j], kz[k]*kz[k])), mν*u[i,j,k], rhs[i,j,k])
      end
    end
  end
end

function add_scalar_difusion!(rhs::AbstractArray,u::AbstractArray,ν::Real,kx::AbstractArray,ky::AbstractArray,kz::AbstractArray,s::AbstractParameters)
  _add_viscosity!(rhs,u,-ν,kx,ky,kz,s)
end

@par function pressure_projection!(rhsx,rhsy,rhsz,kx,ky,kz,s::@par(AbstractParameters))
  if Tr
    _tpressure_projection!(rhsx,rhsy,rhsz,kz,ky,kz,s)
  else
    for k in 2:Nz
      for j in 2:Ny
        for i in 2:Nx
          #@inbounds p1 = -(kx[i]*rhsx[i,j,k] + ky[j]*rhsy[i,j,k] + kz[k]*rhsz[i,j,k])/(kx[i]*kx[i] + ky[j]*ky[j] + kz[k]*kz[k])
          @inbounds p1 = -muladd(kx[i], rhsx[i,j,k], muladd(ky[j], rhsy[i,j,k], kz[k]*rhsz[i,j,k]))/muladd(kx[i], kx[i], muladd(ky[j], ky[j],  kz[k]*kz[k]))
          @inbounds rhsx[i,j,k] = muladd(kx[i],p1,rhsx[i,j,k])
          @inbounds rhsy[i,j,k] = muladd(ky[j],p1,rhsy[i,j,k])
          @inbounds rhsz[i,j,k] = muladd(kz[k],p1,rhsz[i,j,k])
        end
      end
    end
  end
end

@par function _tpressure_projection!(rhsx,rhsy,rhsz,kx,ky,kz,s::@par(AbstractParameters))
  Threads.@threads for k in 2:Nz
    for j in 2:Ny
      for i in 2:Nx
        #@inbounds p1 = -(kx[i]*rhsx[i,j,k] + ky[j]*rhsy[i,j,k] + kz[k]*rhsz[i,j,k])/(kx[i]*kx[i] + ky[j]*ky[j] + kz[k]*kz[k])
        @inbounds p1 = -muladd(kx[i], rhsx[i,j,k], muladd(ky[j], rhsy[i,j,k], kz[k]*rhsz[i,j,k]))/muladd(kx[i], kx[i], muladd(ky[j], ky[j],  kz[k]*kz[k]))
        @inbounds rhsx[i,j,k] = muladd(kx[i],p1,rhsx[i,j,k])
        @inbounds rhsy[i,j,k] = muladd(ky[j],p1,rhsy[i,j,k])
        @inbounds rhsz[i,j,k] = muladd(kz[k],p1,rhsz[i,j,k])
      end
    end
  end
end

@par function addgravity!(rhs,ρ,g::Real,s::@par(BoussinesqParameters))
  if Tr
    _taddgravity!(rhs,ρ,g,s)
  else
    for i=1:Lcs
      @inbounds rhs[i] = muladd(ρ[i],g,rhs[i])
    end
  end
end

@par function _taddgravity!(rhs,ρ,g::Real,s::@par(BoussinesqParameters))
  Threads.@threads for i=1:Lcs
    @inbounds rhs[i] = muladd(ρ[i],g,rhs[i])
  end
end

@par function time_step!(s::A,dt::Real) where {A<:@par(AbstractParameters)}
  if Integrator == :Euller
    Euller!(rawreal(s.u),rawreal(s.rhs),dt,s)
    A <: ScalarParameters && Euller!(complex(s.ρ),complex(s.ρrhs),dt,s)
  elseif Integrator == :Adams_Bashforth3rdO
    Adams_Bashforth3rdO!(rawreal(s.u),rawreal(s.rhs),dt,s.rm1,s.rm2,s)
    A <: ScalarParameters && Adams_Bashforth3rdO!(rawreal(s.ρ),rawreal(s.ρrhs),dt,s.rrm1,s.rrm2,s)
  end
end
