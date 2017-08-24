function advance_in_time!(s::A,init::Int64,Nsteps::Int64,dt::Float64) where {A<:AbstractParameters}
  
  s.p*s.u
  A <: ScalarParameters && s.ps*s.ρ

  if Integrator(s) !== :Euller
    if init==0
      calculate_rhs!(s)
      copy!(s.rm2,complex(s.rhs))
      copy!(s.rm1,s.rm2)
      if A <: ScalarParameters
        copy!(s.rrm1,complex(s.ρrhs))
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
  calculate_pressure!(s.aux.cx,s.aux.cy,s.aux.cz,s.rhs.cx,s.rhs.cy,s.rhs.cz,s.kx,s.ky,s.kz,s)
  addpressure!(complex(s.rhs),complex(s.aux),s)
  A<:BoussinesqParameters && addgravity!(s.rhs.cz,complex(s.ρ),-s.g,s)
  add_viscosity!(complex(s.rhs),complex(s.u),s.ν,s.kx,s.ky,s.kz,s)
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
    scalar_advection!(rawreal(s.aux),rawreal(s.ρ),rawreal(s.u),s)
    s.p*s.aux
    dealias!(complex(s.aux),s)
    s.ps*s.ρ
  end
  s.p*s.u
  A<:ScalarParameters &&  div!(complex(s.ρrhs),s.kx,s.ky,s.kz,s.aux.cx,s.aux.cy,s.aux.cz,s.u.cz,s.dρdz,s)
  return nothing
end
  
function dealias!(rhs::AbstractArray{T,4},s::AbstractParameters{Nx,Ny,Nz,Lcs,Lcv,Nrx,Lrs,Lrv}) where {T,Nx,Ny,Nz,Lcs,Lcv,Nrx,Lrs,Lrv} 
  for l=1:3
  for k in (div(Nz,3)+2):(div(2Nz,3)+1)
    for j in (div(Ny,3)+2):(div(2Ny,3)+1)
      for i in (div(2Nx,3)+1):Nx
        @inbounds rhs[i,j,k,l] = zero(Complex128)
      end
    end
  end
  end
end
  
function add_viscosity!(rhs::AbstractArray,u::AbstractArray,ν::Real,kx::AbstractArray,ky::AbstractArray,kz::AbstractArray,s::AbstractParameters{Nx,Ny,Nz,Lcs,Lcv,Nrx,Lrs,Lrv}) where {Nx,Ny,Nz,Lcs,Lcv,Nrx,Lrs,Lrv}
  for l =1:3 
    for k = 1:Nz
      for j = 1:Ny
        for i = 1:Nx
          @inbounds rhs[i,j,k,l] -= (kx[i]*kx[i] + ky[j]*ky[j] + kz[k]*kz[k])*ν*u[i,j,k,l]
        end
      end
    end
  end
end

function add_scalar_difusion!(rhs::AbstractArray,u::AbstractArray,ν::Real,kx::AbstractArray,ky::AbstractArray,kz::AbstractArray,s::AbstractParameters{Nx,Ny,Nz,Lcs,Lcv,Nrx,Lrs,Lrv}) where {Nx,Ny,Nz,Lcs,Lcv,Nrx,Lrs,Lrv}
  for k = 1:Nz
    for j = 1:Ny
      for i = 1:Nx
        @inbounds rhs[i,j,k] -= (kx[i]*kx[i] + ky[j]*ky[j] + kz[k]*kz[k])*ν*u[i,j,k]
      end
    end
  end
end
 
function addpressure!(rhs::AbstractArray,aux::AbstractArray,s::AbstractParameters{Nx,Ny,Nz,Lcs,Lcv,Nrx,Lrs,Lrv}) where {Nx,Ny,Nz,Lcs,Lcv,Nrx,Lrs,Lrv}
  
  for i in 1:Lcv
    @inbounds rhs[i] = rhs[i] - aux[i]
  end
 
end
  
function calculate_pressure!(auxx,auxy,auxz,rhsx,rhsy,rhsz,kx,ky,kz,s::AbstractParameters{Nx,Ny,Nz,Lcs,Lcv,Nrx,Lrs,Lrv}) where {Nx,Ny,Nz,Lcs,Lcv,Nrx,Lrs,Lrv}
  for k in 2:Nz
    for j in 2:Ny
      for i in 2:Nx
        @inbounds p1 = (kx[i]*rhsx[i,j,k] + ky[j]*rhsy[i,j,k] + kz[k]*rhsz[i,j,k])
        @inbounds p1 = p1/(kx[i]*kx[i] + ky[j]*ky[j] + kz[k]*kz[k])
        @inbounds auxx[i,j,k] = kx[i]*p1
        @inbounds auxy[i,j,k] = ky[j]*p1
        @inbounds auxz[i,j,k] = kz[k]*p1
      end
    end
  end
end

function addgravity!(rhs,ρ,g::Real,s::BoussinesqParameters{Nx,Ny,Nz,Lcs,Lcv,Nrx,Lrs,Lrv}) where {Nx,Ny,Nz,Lcs,Lcv,Nrx,Lrs,Lrv}
  for i=1:Lcs
    @inbounds rhs[i] = muladd(ρ[i],g,rhs[i])
  end
end

function time_step!(s::A,dt::Real) where {A<:AbstractParameters}
  if Integrator(s) == :Euller
    Euller!(complex(s.u),complex(s.rhs),dt,s)
    A<:ScalarParameters && Euller!(complex(s.ρ),complex(s.ρrhs),dt,s)
  elseif Integrator(s) == :Adams_Bashforth3rdO
    Adams_Bashforth3rdO!(complex(s.u),complex(s.rhs),dt,s.rm1,s.rm2,s)
    A<:ScalarParameters && Adams_Bashforth3rdO!(complex(s.ρ),complex(s.ρrhs),dt,s.rrm1,s.rrm2,s)
  end
end