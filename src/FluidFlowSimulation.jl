module FluidFlowSimulation

using InplaceRealFFTW

include("types.jl")
include("vectorfunctions.jl")

function dns(u::VectorField,Nt::Int,dt::Real)

  rhs = similar(u)
  aux = similar(rhs)

  plan_rfft!(rhs,1:3,flags=FFTW.MEASURE)
  p = Base.DFT.ScaledPlan(FFTW.rFFTWPlan{eltype(rhs.r),FFTW.BACKWARD,false,4}(complex(rhs), real(aux), 1:3, FFTW.MEASURE&FFTW.DESTROY_INPUT,FFTW.NO_TIMELIMIT),Base.DFT.normalization(eltype(real(aux)), size(real(aux)), 1:3)) 

  for t=1:Nt
    calculate_rhs!(rhs,u,aux)
    time_step!(u,rhs,dt)
  end

  return u
end

function calculate_rhs!(rhs::VectorField,u::VectorField,aux::VectorField)
  compute_nonlinear!(rhs,u,aux)  
  compute_viscosity!(rhs,u,aux)
  addpressure!(rhs,aux)
end

function compute_nonlinear!(rhs::VectorField{T,N,A,N1,C,R},u::VectorField{T,N,A,N1,C,R},aux::VectorField{T,N,A,N1,C,R}) where {T,N,A,N1,C,R}
    curl!(rhs,u)
    irfft!(u)

    p = Base.DFT.ScaledPlan(FFTW.rFFTWPlan{Complex{T},FFTW.BACKWARD,false,N}(rhs.c, aux.r, 1:N1, FFTW.DESTROY_INPUT,FFTW.NO_TIMELIMIT),Base.DFT.normalization(eltype(aux.r), size(aux.r), 1:N1)) 
    A_mul_B!(real(aux),p,complex(rhs))

    rcross!(rhs,u,aux)
    rfft!(rhs)
    dealias!(rhs)
    rfft!(u)
end

function dealias!(u::FieldVector)
  nx,ny,nz = size(u.cx)
  @inbounds for k in div(2nz,3):nz
    @inbounds for j in div(2ny,3):ny
      @inbounds for i in div(2nx,3):nz
        u.cx[i,j,k] = 0.0 + 0.0im
        u.cy[i,j,k] = 0.0 + 0.0im
        u.cz[i,j,k] = 0.0 + 0.0im
      end
    end
  end
end

function addpressure!(rhs::VectorField,aux::VectorField)
  @inbounds for k in 1:nz
    @inbounds for j in 1:ny
        @inbounds for i in 1:nz
            p1 = (kx[i]*rhs.cx[i,j,k] + ky[j]*rhs.cy[i,j,k] + kz[k]*rhs.cz[i,j,k])
            p1 = p1/(kx[i]*kx[i] + ky[j]*ky[j] + kz[k]*kz[k])
            aux.cx[i,j,k] = kx[i]*p1
            aux.cy[i,j,k] = ky[j]*p1
            aux.cz[i,j,k] = kz[k]*p1
        end
    end
  end

  @inbounds for i in 1:lenght(rhs)
    rhs[i] = rhs[i] - aux[i]
  end
end

function timestep!(u::VectorField,rhs::Vector::Field,dt::Real)
  @inbounds for i in length(u)
    u[i] += dt*rhs[i]
  end
end

end # module