__precompile__()
module FluidFlowSimulation
export dns, VectorField

using InplaceRealFFTW

include("types.jl")
include("vectorfunctions.jl")

function dns(u::VectorField{T,N,A},Nt::Int,dt::Real,ν::Real,lx::Real,ly::Real,lz::Real) where {T,N,A}

  rhs = similar(u)
  aux = similar(rhs)
  nx::Int ,ny::Int,nz::Int = size(u.data.r)
  global const kx = rfftfreq(nx,lx)
  global const ky = fftfreq(ny,ly)
  global const kz = fftfreq(nz,lz)

  global const p = plan_rfft!(rhs,1:3,flags=FFTW.MEASURE)
  p.pinv = plan_irfft!(rhs,1:3,flags=FFTW.MEASURE&FFTW.PRESERVE_INPUT)

  global const ip = Base.DFT.ScaledPlan(FFTW.rFFTWPlan{Complex{T},FFTW.BACKWARD,false,N}(complex(rhs), real(aux), 1:3, FFTW.MEASURE&FFTW.DESTROY_INPUT,FFTW.NO_TIMELIMIT),Base.DFT.normalization(T, size(real(aux)), 1:3)) 

  p*u
  for t=1:Nt
    calculate_rhs!(rhs,u,aux,ν)
    time_step!(u,rhs,dt)
  end

  return u
end

function calculate_rhs!(rhs::VectorField,u::VectorField,aux::VectorField,ν::Real)
  compute_nonlinear!(rhs,u,aux)  
  add_viscosity!(rhs,u,ν)
  addpressure!(rhs,aux)
end

function compute_nonlinear!(rhs::VectorField,u::VectorField,aux::VectorField)
    curl!(rhs,u)
    p\u

    A_mul_B!(real(aux),ip,complex(rhs))

    rcross!(rhs,u,aux)
    p*rhs
    dealias!(rhs)
    p*u
end

function dealias!(u::VectorField)
  nx,ny,nz = size(u.cx)
  for l=1:3
  for k in div(2nz,3):nz #Fix
    for j in div(2ny,3):ny #Fix
      for i in div(2nx,3):nx
      @inbounds  u[i,j,k,l] = 0.0 + 0.0im
      end
    end
  end
  end
end

@fastmath function addpressure!(rhs::VectorField,aux::VectorField)
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

@fastmath function time_step!(u::VectorField,rhs::VectorField,dt::Real)
  for i in length(u)
    @inbounds u[i] += dt*rhs[i]
  end
end

function add_viscosity!(rhs::VectorField,u::VectorField,ν::Real)
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

end # module