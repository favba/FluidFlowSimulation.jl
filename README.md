# FluidFlowSimulation

Pseudo-spectral numerical simulation of incompressible fluid flow.
Requires [InplaceRealFFTW.jl](https://github.com/favba/InplaceRealFFTW.jl) to perform in-place real-to-complex Fourier transforms.

Parameters should be written in a file named *global* an initial conditions on files *u1.0*, *u2.0*, *u3.0* .

## Example
Building Taylor-Green vortices initial conditions and global file.
On some empty folder do:
```julia-repl
julia> using InplaceRealFFTW;
julia> x = linspace(0,2π*(1-1/64),64);
julia> y = reshape(x,1,64,1);
julia> z = reshape(x,1,1,64);
julia> u3 = zeros(64,64,64);
julia> u1 = similar(u3);
julia> u2 = similar(u1);
julia> @. u1 =  cos(x)*sin(y)*sin(z);
julia> @. u2 =  -sin(x)*cos(y)*sin(z);
julia> write("u1.0",PaddedArray(u1));
julia> write("u2.0",PaddedArray(u2));
julia> write("u3.0",PaddedArray(u3));
julia> par = """
       kinematicViscosity 0.03333333
       xDomainSize 1.0
       yDomainSize 1.0
       zDomainSize 1.0
       nx 64
       ny 64
       nz 64

       Nt 500
       dt 0.05
       dtStat 20
       writeTime 100

       """;
julia> write("global",par);
julia> exit()
```
Start julia in this folder and run simulation with:

```julia-repl
julia> using FluidFlowSimulation;
julia> run_simulation();
```
or just use the shell command ```julia -O 3 -e "using FluidFlowSimulation; run_simulation()"```.

