__precompile__()
module FluidFlowSimulation
export dns, VectorField, Parameters

using InplaceRealFFTW
using StaticArrays

include("macros.jl")
include("types.jl")
include("vectorfunctions.jl")
include("time_step_functions.jl")

function dns(u::VectorField,s::AbstractParameters,Nt::Int64,dt::Float64)

  rhs = similar(u)
  aux = similar(rhs)
  s.p*u

  if Integrator(s) !== :Euller
    calculate_rhs!(rhs,u,aux,s)
    copy!(s.rm2,rhs)
    copy!(s.rm1,s.rm2)
  end

  for t=1:Nt
    calculate_rhs!(rhs,u,aux,s)
    time_step!(u,rhs,dt,s)
  end

  return s.p\u
end

function calculate_rhs!(rhs::VectorField,u::VectorField,aux::VectorField,s::AbstractParameters)
  compute_nonlinear!(rhs,u,aux,s)  
  calculate_pressure!(aux.cx,aux.cy,aux.cz,rhs.cx,rhs.cy,rhs.cz,s.kx,s.ky,s.kz,s)
  addpressure!(rhs,aux,s)
  add_viscosity!(rhs,u,s.ν,s.kx,s.ky,s.kz,s)
end

function compute_nonlinear!(rhs::VectorField,u::VectorField,aux::VectorField,s::AbstractParameters)
    curl!(rhs,u,s)
    s.p\u

    A_mul_B!(real(aux),s.ip,complex(rhs))

    rcross!(rhs,u,aux,s)
    s.p*rhs
    dealias!(rhs,s)
    s.p*u
end

function dealias!(rhs::VectorField{T,A},s::AbstractParameters{Nx,Ny,Nz}) where {T,A,Nx,Ny,Nz} 
  for l=1:3
  for k in (div(Nz,3)+2):(div(2Nz,3)+1)
    for j in (div(Ny,3)+2):(div(2Ny,3)+1)
      for i in (div(2Nx,3)+1):Nx
      @inbounds  rhs[i,j,k,l] = zero(Complex{T})
      end
    end
  end
  end
end

function add_viscosity!(rhs::VectorField,u::VectorField,ν::Real,kx::AbstractArray,ky::AbstractArray,kz::AbstractArray,s::AbstractParameters{Nx,Ny,Nz}) where {Nx,Ny,Nz}
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

function addpressure!(rhs::VectorField,aux::VectorField,s::AbstractParameters{Nx,Ny,Nz}) where {Nx,Ny,Nz}

  for i in 1:3*Nx*Ny*Nz
    @inbounds rhs[i] = rhs[i] - aux[i]
  end

end

function calculate_pressure!(auxx,auxy,auxz,rhsx,rhsy,rhsz,kx,ky,kz,s::AbstractParameters{Nx,Ny,Nz}) where {Nx,Ny,Nz}
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

function time_step!(u::VectorField,rhs::VectorField,dt::Real,s::AbstractParameters{Nx,Ny,Nz}) where {Nx,Ny,Nz}
  if Integrator(s) == :Euller
    Euller!(u,rhs,dt,s)
  elseif Integrator(s) == :Adams_Bashforth3rdO
    Adams_Bashforth3rdO!(u,rhs,dt,s.rm1,s.rm2,s)
  end
end

end # module