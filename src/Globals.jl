__precompile__(false)
module Globals

using ..ReadGlobal

export Lx,Ly,Lz,Nx,Ny,Nz,Lcs,Lcv,Nrrx,Nrx,Lrs,Lrv,ν,Dealias,kx,ky,kz,Thr,Nt,RealRanges

function splitrange(lr,nt)
    a = UnitRange{Int}[]
    sizehint!(a,nt)
    n = lr÷nt
    r = lr%nt
    stop = 0
    init = 1
    for i=1:r
        stop=init+n
        push!(a,init:stop)
        init = stop+1
    end
    for i=1:(nt-r)
        stop=init+n-1
        push!(a,init:stop)
        init = stop+1
    end
    return (a...,)
end

function rfftfreq(n::Integer,s::Real)::Vector{Float64}
    Float64[(n/2 - i)/s for i = n/2:-1:0]
end
  
function fftfreq(n::Integer,s::Real)::Vector{Float64}
    if iseven(n)
        return vcat(Float64[(n/2 - i)/s for i = n/2:-1:1],Float64[-i/s for i = n/2:-1:1])
    else return vcat(Float64[(n/2 - i)/s for i = n/2:-1:0],Float64[-i/s for i = (n-1)/2:-1:1])
    end
end

    d = readglobal()
    const global Nrx = parse(Int,d[:nx])
    const global Ny = parse(Int,d[:ny])
    const global Nz = parse(Int,d[:nz])
  
    const global Nx = div(Nrx,2)+1
    const global Nrrx = 2Nx
    const global Lcs = Nx*Ny*Nz
    const global Lcv = 3*Lcs
    const global Lrs = 2*Lcs
    const global Lrv = 2*Lcv
  
    const global Lx = Float64(eval(Meta.parse(d[:xDomainSize])))
    const global Ly = Float64(eval(Meta.parse(d[:yDomainSize])))
    const global Lz = Float64(eval(Meta.parse(d[:zDomainSize])))
    const global ν = Float64(eval(Meta.parse(d[:kinematicViscosity])))
  
    kxp = reshape(rfftfreq(Nrx,Lx),(Nx,1,1))
    kyp = reshape(fftfreq(Ny,Ly),(1,Ny,1))
    kzp = reshape(fftfreq(Nz,Lz),(1,1,Nz))
  
    haskey(d,:threaded) ? (tr = parse(Bool,d[:threaded])) : (tr = true)
    const global Thr = tr
  
    const global Nt = tr ? Threads.nthreads() : 1 
  
    const global RealRanges = splitrange(Lrs, Nt)
  
    haskey(d,:dealias) ? (Dealiastype = Symbol(d[:dealias])) : (Dealiastype = :sphere)
    haskey(d,:cutoff) ? (cutoffr = Float64(eval(Meta.parse(d[:cutoff])))) : (cutoffr = 15/16)
    const global Dealias = (Dealiastype,cutoffr) 

    cutoff = (cutoffr*kxp[end])^2

    const global dealias = BitArray(undef,(Nx,Ny,Nz))

    if Dealiastype == :sphere
        @. dealias = (kxp^2 + kyp^2 + kzp^2) > cutoff
    elseif Dealiastype == :cube
        @. dealias = (kxp^2 > cutoff) | (kyp^2 > cutoff) | (kzp^2 > cutoff)
    end

    const global kx = (kxp...,)
    const global ky = (kyp...,)
    const global kz = (kzp...,)
  
end