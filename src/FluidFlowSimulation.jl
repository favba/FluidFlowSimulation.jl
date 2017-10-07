__precompile__()
module FluidFlowSimulation
export run_simulation, advance_in_time!, VectorField, parameters, readglobal

using InplaceRealFFTW
using StaticArrays
using MacroTools.prewalk

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

function run_simulation(s::AbstractParameters,dtStats::Integer,dtOutput::Integer,totalnsteps::Integer,dt::Real)
  init=0
  writeheader(s)
  stats(s,init,dt)

  @assert totalnsteps >= dtOutput >= dtStats
  for i = 1:(totalnsteps ÷ dtOutput)
    for j = 1:(dtOutput ÷ dtStats)
      init = advance_in_time!(s,init,dtStats,dt)
      stats(s,init,dt)
    end
    writeoutput(s,init)
  end
end

end # module