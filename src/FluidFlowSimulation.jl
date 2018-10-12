__precompile__(false)
module FluidFlowSimulation
export run_simulation, advance_in_time!, VectorField, parameters, readglobal

using FFTW, InplaceRealFFT, FluidTensors, FluidFields

function Base.unsafe_getindex(A::Tuple, I)
  Base.@_inline_meta
  @inbounds r = getindex(A, I)
  r
end

include("ReadGlobal.jl")
include("Globals.jl")

using .ReadGlobal
using .Globals

include("macros.jl")
include("simulationtypes.jl")
include("spectrum.jl")
include("forcing.jl")
include("utils.jl")
include("vectorfunctions.jl")
include("time_step_functions.jl")
include("stats.jl")
include("output.jl")
include("models.jl")


function run_simulation()
    par = readglobal()
    s = parameters(par)
    init = haskey(par,:start) ? parse(Int,par[:start]) : 0
    ttime = haskey(par,:startTime) ? parse(Float64,par[:startTime]) : 0.0
    run_simulation(s,init,ttime,parse(Int,par[:dtStat]),parse(Int,par[:writeTime]),parse(Int,par[:nt]))
end

@par function run_simulation(s::@par(AbstractSimulation),init::Int,itime::Real,dtStats::Integer,dtOutput::Integer,totalnsteps::Integer)
    @info("Simulation started.")

    initialize!(s,init)

    @assert totalnsteps >= dtOutput 
    @assert totalnsteps >= dtStats
    finalstep = init + totalnsteps

    ttime = itime
    for i = (init+1):finalstep
        advance_in_time!(s)
        ttime+=get_dt(s)
        mod(i,dtOutput) == 0 && writeoutput(s,i)
        mod(i,dtStats) == 0 && writestats(s,i,ttime)
    end
end

function initialize!(s::AbstractSimulation,init::Integer)
    init == 0 && writeheader(s)
    rfft_and_scale!(s.u)
    #dealias!(s.u,s)

    #haspassivescalar(s) && s.passivescalar.ps * s.passivescalar.ρ
    if haspassivescalar(s)
        rfft_and_scale!(s.passivescalar.ρ) 
        #dealias!(s.passivescalar.ρ,s)
    end
    #hasdensity(s) && s.densitystratification.ps * s.densitystratification.ρ
    if hasdensity(s) 
        rfft_and_scale!(s.densitystratification.ρ) 
        #dealias!(s.densitystratification.ρ,s)
    end

    calculate_rhs!(s)

    initialize!(s.timestep,s.rhs,s)
    initialize!(s.passivescalar,s)
    initialize!(s.densitystratification,s)
  
    initialize!(s.forcing,s)

    init == 0 && writestats(s,0,0)
    return nothing
end

end # module