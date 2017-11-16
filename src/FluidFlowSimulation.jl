__precompile__()
module FluidFlowSimulation
export run_simulation, advance_in_time!, VectorField, parameters, readglobal

using InplaceRealFFTW
function Base.unsafe_getindex(A::Tuple, I)
  Base.@_inline_meta
  @inbounds r = getindex(A, I)
  r
end

include("ReadGlobal.jl")
include("macros.jl")
include("types.jl")
include("vectorfunctions.jl")
include("time_step_functions.jl")
include("stats.jl")
include("output.jl")
include("models.jl")

using .ReadGlobal

function run_simulation()
  par = readglobal()
  s = parameters(par)
  run_simulation(s,parse(Int,par[:dtStat]),parse(Int,par[:writeTime]),parse(Int,par[:nt]),parse(Float64,par[:dt]))
end

@par function run_simulation(s::@par(AbstractParameters),dtStats::Integer,dtOutput::Integer,totalnsteps::Integer,dt::Real)
  init=0
  time=0.
  writeheader(s)
  s.p*s.u
  typeof(s)<:ScalarParameters && s.ps*s.ρ
  
  if Integrator !== :Euller
    calculate_rhs!(s)
    mycopy!(s.rm2x,s.rhs.rx,s)
    copy!(s.rm1x,s.rm2x)
    mycopy!(s.rm2y,s.rhs.ry,s)
    copy!(s.rm1y,s.rm2y)
    mycopy!(s.rm2z,s.rhs.rz,s)
    copy!(s.rm1z,s.rm2z)
    if typeof(s) <: ScalarParameters
      mycopy!(s.rrm1,rawreal(s.ρrhs),s)
      copy!(s.rrm2,s.rrm1)
    end
  end

  stats(s,init,time)

  @assert totalnsteps >= dtOutput >= dtStats
  for i = 1:(totalnsteps ÷ dtOutput)
    for j = 1:(dtOutput ÷ dtStats)
      init, dt, time = advance_in_time!(s,init,dtStats,dt,time)
      stats(s,init,time)
    end
    writeoutput(s,init)
  end
end

end # module