# FluidFlowSimulation

Pseudo-spectral numerical simulation of incompressible fluid flow.
Requires [InplaceRealFFTW.jl](https://github.com/favba/InplaceRealFFTW.jl) to perform in-place real-to-complex Fourier transforms.

Parameters should be written in a file named *global* an initial conditions on files *u1.0*, *u2.0*, *u3.0* .

## Example
Building Taylor-Green vortices initial conditions and global file.
On some empty folder do:
```julia
using InplaceRealFFTW;
x = linspace(0,2π*(1-1/64),64);
y = reshape(x,1,64,1);
z = reshape(x,1,1,64);
u3 = zeros(64,64,64);
u1 = similar(u3);
u2 = similar(u1);
@. u1 =  cos(x)*sin(y)*sin(z);
@. u2 =  -sin(x)*cos(y)*sin(z);
write("u1.0",PaddedArray(u1));
write("u2.0",PaddedArray(u2));
write("u3.0",PaddedArray(u3));
par = """
kinematicViscosity 0.03333333
xDomainSize 1.0
yDomainSize 1.0
zDomainSize 1.0
nx 64
ny 64
nz 64

nt 500
dt 0.05
dtStat 20
writeTime 100

""";
write("global",par);
exit()
```
Start julia in this folder and run simulation with:

```julia
using FluidFlowSimulation;
run_simulation();
```
or just use the shell command ```julia -O 3 -e "using FluidFlowSimulation; run_simulation()"```.

