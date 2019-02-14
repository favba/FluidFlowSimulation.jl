__precompile__(false)
module FluidFlowSimulation
export run_simulation, advance_in_time!, parameters, readglobal

using FFTW, InplaceRealFFT, FluidTensors, FluidFields, LinearAlgebra, GlobalFileHelper, Interpolations, DelimitedFiles

include("Globals.jl")

using .Globals

include("macros.jl")
include("simulationtypes.jl")
include("Forcing_methods/spectrum.jl")
include("Forcing_methods/forcing.jl")
include("utils.jl")
include("vectorfunctions.jl")
include("time_step_functions/time_step_functions.jl")
include("stats.jl")
include("output.jl")
include("models.jl")


function run_simulation()
    par = readglobal()
    s = parameters(par)
    lengthtime = haskey(par,:timeDuration) ? parse(Float64,par[:timeDuration]) : Inf
    nt = haskey(par,:nt) ? parse(Int,par[:nt]) : (typemax(Int) - s.iteration[])
    run_simulation(s,nt,lengthtime)
end

@par function run_simulation(s::@par(AbstractSimulation),totalnsteps::Integer,lenghtime::Real)
    @info("Simulation started.")

    initialize!(s)

    @assert totalnsteps + s.iteration[] >= s.dtoutput 
    @assert totalnsteps + s.iteration[] >= s.dtstats
    @show finalstep = s.iteration[] + totalnsteps
    @show finaltime = s.time[]+lenghtime

    while (s.iteration[]<finalstep && s.time[]<finaltime)
        advance_in_time!(s)
        mod(s.iteration[],s.dtstats) == 0 && writestats(s)
    end
    
    calculate_rhs!(s)
    mod(s.iteration[],s.dtstats) == 0 || writestats(s)
    if !(mod(s.iteration[],s.dtoutput) == 0)
        real!(s.u)
        hasdensity(s) && real!(s.densitystratification.ρ)
        haspassivescalar(s) && real!(s.passivescalar.φ)
        hasles(s) && real!(s.lesmodel.tau)

        writeoutput(s)
    end

    if typeof(s.forcing) <: RfForcing
        writedlm("R.$(s.iteration[])",s.forcing.R)
    end

    return s
end

function initialize!(s::AbstractSimulation)
    s.iteration[] == 0 && writeheader(s)
    myfourier!(s.u)

    if haspassivescalar(s)
        myfourier!(s.passivescalar.φ) 
    end

    if hasdensity(s) 
        myfourier!(s.densitystratification.ρ) 
    end

    calculate_rhs!(s)

    initialize!(s.timestep,s.rhs,s)
    initialize!(s.passivescalar,s)
    initialize!(s.densitystratification,s)
  
    initialize!(s.forcing,s)

    s.iteration[] == 0 && writestats(s)
    return nothing
end

end # module