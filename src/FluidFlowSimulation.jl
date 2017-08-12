__precompile__()
module FluidFlowSimulation
export dns, VectorField, Parameters

using InplaceRealFFTW

include("types.jl")
include("vectorfunctions.jl")

function dns(u::VectorField,s::AbstractParameters,Nt::Int64,dt::Float64)

  rhs = similar(u)
  aux = similar(rhs)
  s.p*u

  for t=1:Nt
    calculate_rhs!(rhs,u,aux,s)
    time_step!(u,rhs,dt)
  end

  return u
end

function calculate_rhs!(rhs::VectorField,u::VectorField,aux::VectorField,s::AbstractParameters)
  compute_nonlinear!(rhs,u,aux,s)  
  add_viscosity!(rhs,u,s.ν,s.kx,s.ky,s.kz)
  addpressure!(rhs,aux,s.kx,s.ky,s.kz)
end

function compute_nonlinear!(rhs::VectorField,u::VectorField,aux::VectorField,s::AbstractParameters)
    curl!(rhs,u,s.kx,s.ky,s.kz)
    s.p\u

    A_mul_B!(real(aux),s.ip,complex(rhs))

    rcross!(rhs,u,aux)
    s.p*rhs
    dealias!(rhs)
    s.p*u
end

function dealias!(u::VectorField)
  nx,ny,nz = size(u.cx)
  for l=1:3
  for k in (div(nz,3)+2):(div(2nz,3)+1)
    for j in (div(ny,3)+2):(div(2ny,3)+1)
      for i in (div(2nx,3)+1):nx
      @inbounds  u[i,j,k,l] = 0.0 + 0.0im
      end
    end
  end
  end
end

function add_viscosity!(rhs::VectorField,u::VectorField,ν::Real,kx::AbstractVector,ky::AbstractVector,kz::AbstractVector)
  nx,ny,nz,_ = size(rhs) 
  for l =1:3 
    for k = 1:nz
      for j=1:ny
        for i=1:nx
          @inbounds rhs[i,j,k,l] = (kx[i]*kx[i] + ky[j]*ky[j] + kz[k]*kz[k])*ν*u[i,j,k,l]
        end
      end
    end
  end
end

function addpressure!(rhs::VectorField,aux::VectorField,kx::AbstractVector,ky::AbstractVector,kz::AbstractVector)
  nx,ny,nz,_ = size(rhs)  
  for k in 1:nz
    for j in 1:ny
      for i in 1:nx
        @inbounds p1 = (kx[i]*rhs.cx[i,j,k] + ky[j]*rhs.cy[i,j,k] + kz[k]*rhs.cz[i,j,k])
        @inbounds p1 = p1/(kx[i]*kx[i] + ky[j]*ky[j] + kz[k]*kz[k])
        @inbounds aux.cx[i,j,k] = kx[i]*p1
        @inbounds aux.cy[i,j,k] = ky[j]*p1
        @inbounds aux.cz[i,j,k] = kz[k]*p1
      end
    end
  end

  for i in 1:length(rhs)
    @inbounds rhs[i] = rhs[i] - aux[i]
  end
end

function time_step!(u::VectorField,rhs::VectorField,dt::Real)
  for i in length(u)
    @inbounds u[i] += dt*rhs[i]
  end
end

end # module