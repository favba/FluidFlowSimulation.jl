__precompile__()
module FluidFlowSimulation
export run_simulation, advance_in_time!, VectorField, parameters, readglobal

import FFTW
using InplaceRealFFT
function Base.unsafe_getindex(A::Tuple, I)
  Base.@_inline_meta
  @inbounds r = getindex(A, I)
  r
end

include("ReadGlobal.jl")
include("macros.jl")
include("fieldtypes.jl")
include("simulationtypes.jl")
include("utils.jl")
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

@par function run_simulation(s::@par(AbstractSimulation),dtStats::Integer,dtOutput::Integer,totalnsteps::Integer,dt::Real)
  info("Simulation started.")
  init=0
  ttime=0.

  initialize!(s)

  @assert totalnsteps >= dtOutput 
  @assert totalnsteps >= dtStats

  for i = 1:totalnsteps
    advance_in_time!(s,dt)
    ttime+=dt
    mod(i,dtOutput) == 0 && writeoutput(s,i)
    mod(i,dtStats) == 0 && writestats(s,i,ttime)
  end
end

function initialize!(s::AbstractSimulation)
  writeheader(s)
  s.p*s.u
  haspassivescalar(s) && s.passivescalar.ps * s.passivescalar.ρ
  hasdensity(s) && s.densitystratification.ps * s.densitystratification.ρ

  calculate_rhs!(s)

  initialize!(s.timestep,s.rhs,s)
  initialize!(s.passivescalar,s)
  initialize!(s.densitystratification,s)

  writestats(s,0,0)
end

end # module