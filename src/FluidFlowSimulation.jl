__precompile__()
module FluidFlowSimulation
export run_simulation, advance_in_time!, VectorField, Parameters

using InplaceRealFFTW
using StaticArrays

include("ReadGlobal.jl")
include("macros.jl")
include("types.jl")
include("vectorfunctions.jl")
include("time_step_functions.jl")
include("stats.jl")
include("output.jl")

using .ReadGlobal

function run_simulation()
  par = readglobal()
  s = Parameters(par)
  run_simulation(s,parse(Int,par["dtStat"]),parse(Int,par["writeTime"]),parse(Int,par["Nt"]),parse(Float64,par["dt"]))
end

function run_simulation(s::AbstractParameters,dtStats::Integer,dtOutput::Integer,totalnsteps::Integer,dt::Real)
  init=0
  stats(s,init,dt)

  @assert totalnsteps > dtOutput > dtStats
  for i = 1:(totalnsteps ÷ dtOutput)
    for j = 1:(dtOutput ÷ dtStats)
      init = advance_in_time!(s,init,dtStats,dt)
      stats(s,init,dt)
    end
    writeoutput(s,init)
  end
end

function advance_in_time!(s::AbstractParameters,init::Int64,Nsteps::Int64,dt::Float64)

  s.p*s.u

  if Integrator(s) !== :Euller
    if init==0
      calculate_rhs!(s)
      copy!(s.rm2,complex(s.rhs))
      copy!(s.rm1,s.rm2)
    end
  end

  for t=1:Nsteps
    init += 1
    calculate_rhs!(s)
    time_step!(s,dt)
  end

  s.p\s.u
  return init
end

function calculate_rhs!(s::AbstractParameters)
  compute_nonlinear!(s)  
  calculate_pressure!(s.aux.cx,s.aux.cy,s.aux.cz,s.rhs.cx,s.rhs.cy,s.rhs.cz,s.kx,s.ky,s.kz,s)
  addpressure!(complex(s.rhs),complex(s.aux),s)
  add_viscosity!(complex(s.rhs),complex(s.u),s.ν,s.kx,s.ky,s.kz,s)
end

function compute_nonlinear!(s::AbstractParameters)
    curl!(s.rhs,s.u,s)
    s.p\s.u

    A_mul_B!(real(s.aux),s.ip,complex(s.rhs))

    rcross!(s.rhs,s.u,s.aux,s)
    s.p*s.rhs
    dealias!(complex(s.rhs),s)
    s.p*s.u
end

function dealias!(rhs::AbstractArray{T,4},s::AbstractParameters{Nx,Ny,Nz,Lcs,Lcv,Nrx,Lrs,Lrv}) where {T,Nx,Ny,Nz,Lcs,Lcv,Nrx,Lrs,Lrv} 
  for l=1:3
  for k in (div(Nz,3)+2):(div(2Nz,3)+1)
    for j in (div(Ny,3)+2):(div(2Ny,3)+1)
      for i in (div(2Nx,3)+1):Nx
      @inbounds  rhs[i,j,k,l] = zero(Complex128)
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

function time_step!(s::AbstractParameters,dt::Real)
  if Integrator(s) == :Euller
    Euller!(complex(s.u),complex(s.rhs),dt,s)
  elseif Integrator(s) == :Adams_Bashforth3rdO
    Adams_Bashforth3rdO!(complex(s.u),complex(s.rhs),dt,s.rm1,s.rm2,s)
  end
end

end # module