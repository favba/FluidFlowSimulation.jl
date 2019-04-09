macro def(name, definition) # thanks to http://www.stochasticlifestyle.com/type-dispatch-design-post-object-oriented-programming-julia/
    return esc(quote
        macro $name()
            esc($(Expr(:quote, definition)))
        end
    end)
end

macro cthreads(condition::Symbol,loop::Expr) #does not work well because of #15276, but seems to work on Julia v0.7
    return esc(quote
        if $condition
            Threads.@threads $loop
        else
            $loop
        end
    end)
end

macro mthreads(ex)
    if THR
       return esc(:(Threads.@threads $ex))
    else
        return esc(ex)
    end
end

macro msimd(ex)
    return esc(ex)
    #return esc(:(@simd $ex))
end

macro gen(ex)
    ex2 = macroexpand(ex)
    return esc(:(@generated $ex2))
end

const sim_par = (:EquationType,:VelocityTimeStepType,:PassiveScalarType,:DensityStratificationType,:LESModelType,:ForcingType,:HyperViscosityType)
# Nx,Ny,Nz size of the grid in Fourier Space
# Lcs = Nx*Ny*Nz; Lcv = Lcs*3
# Nrx size of x direction in Real Space skipping padding. The same as 1:(:nx) in global file
# Nrrx size of x direction in Real Space NOT skipping padding.
# Lrs = (Nrx+2)*Ny*Nz; Lrv = 3*Nrx
# kx,ky,kz wavenumber vectors.
# Gdirec can be :x, :y, :z and gives the gravity direction. Odly, only if :z seems to work
# Thr is true or false for threads.
# Nt number of threads.

const sim_fields = (:passivescalar,:densitystratification,:lesmodel,:forcing,:hyperviscosity)

"""
    @par(t::Symbol)
Add the parameters defined in `sim_par` to the Symbol.

# Examples
```julia-repl
julia> sim_par = (:Nx,:Ny,:Nz,:Lcs,:Lcv,:Nrx,:Lrs,:Lrv,:Tr,:Integrator)
(:Nx, :Ny, :Nz, :Lcs, :Lcv, :Nrx, :Lrs, :Lrv, :Tr, :Integrator)

julia> @macroexpand @par(Parameters)
:(Parameters{Nx, Ny, Nz, Lcs, Lcv, Nrx, Lrs, Lrv, Tr, Integrator})

julia> @macroexpand @par(BoussinesqParameters)
:(BoussinesqParameters{Nx, Ny, Nz, Lcs, Lcv, Nrx, Lrs, Lrv, Tr, Integrator})
```
"""
macro par(t::Symbol)
    return esc(Expr(:curly,t,sim_par...))
end

"""
---
    @par(ex:Expr)
Add the parameters defined in `sim_par` after a `where` on a function definition.

# Examples
```julia-repl
julia> sim_par = (:Nx,:Ny,:Nz,:Lcs,:Lcv,:Nrx,:Lrs,:Lrv,:Tr,:Integrator)
(:Nx, :Ny, :Nz, :Lcs, :Lcv, :Nrx, :Lrs, :Lrv, :Tr, :Integrator)

julia> @macroexpand @par function test(s::@par(Parameters),b::Array{T,N}) where {T,N}; return 1; end
:(function test(s::Parameters{Nx, Ny, Nz, Lcs, Lcv, Nrx, Lrs, Lrv, Tr, Integrator}, b::Array{T, N}) where {Nx, Ny, Nz, Lcs, Lcv, Nrx, Lrs, Lrv, Tr, Integrator, T, N} # REPL[1], line 1:
return 1
end)
```
"""
macro par(ex::Expr)
    if ex.head == :function || ex.head == :(=)
        if ex.args[1].head == :where
            # append!(ex.args[1].args,sim_par)
            ex.args[1].args = vcat(ex.args[1].args[1], sim_par..., ex.args[1].args[2:end]) # This way the parameters are available to the existing parameters
        elseif ex.args[1].head == :call
            ex.args[1] = Expr(:where, ex.args[1], sim_par...) 
        end
    end
    return esc(ex)
end