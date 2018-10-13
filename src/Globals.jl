__precompile__(false)
module Globals

using ..ReadGlobal, FluidTensors

export LX,LY,LZ,NX,NY,NZ,NRRX,NRX,ν,DEALIAS_TYPE,KX,KY,KZ,THR,NT,TRANGE,REAL_RANGES,DEALIAS,K,RXRANGE,XRANGE,YRANGE,ZRANGE,RANGEC

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
    const global NRX = parse(Int,d[:nx])
    const global NY = parse(Int,d[:ny])
    const global NZ = parse(Int,d[:nz])
  
    const global NX = div(NRX,2)+1
    const global NRRX = 2NX
  
    const global LX = Float64(eval(Meta.parse(d[:xDomainSize])))
    const global LY = Float64(eval(Meta.parse(d[:yDomainSize])))
    const global LZ = Float64(eval(Meta.parse(d[:zDomainSize])))
    const global ν = Float64(eval(Meta.parse(d[:kinematicViscosity])))
  
    kxp = reshape(rfftfreq(NRX,LX),(NX,1,1))
    kyp = reshape(fftfreq(NY,LY),(1,NY,1))
    kzp = reshape(fftfreq(NZ,LZ),(1,1,NZ))
  
    haskey(d,:threaded) ? (tr = parse(Bool,d[:threaded])) : (tr = true)
    const global THR = tr
  
    const global NT = tr ? Threads.nthreads() : 1 
    const global TRANGE = Base.OneTo(NT)
  
    const global REAL_RANGES = splitrange(NRRX*NY*NZ, NT)
  
    haskey(d,:dealias) ? (Dealiastype = Symbol(d[:dealias])) : (Dealiastype = :sphere)
    haskey(d,:cutoff) ? (cutoffr = Float64(eval(Meta.parse(d[:cutoff])))) : (cutoffr = 15/16)
    const global DEALIAS_TYPE = (Dealiastype,cutoffr) 

    cutoff = (cutoffr*kxp[end])^2

    const global DEALIAS = BitArray(undef,(NX,NY,NZ))

    if Dealiastype == :sphere
        @. DEALIAS = (kxp^2 + kyp^2 + kzp^2) > cutoff
    elseif Dealiastype == :cube
        @. DEALIAS = (kxp^2 > cutoff) | (kyp^2 > cutoff) | (kzp^2 > cutoff)
    end

    const global KX = (kxp...,)
    const global KY = (kyp...,)
    const global KZ = (kzp...,)

    s = (NX,NY,NZ)
    const global K = VecArray(HomogeneousArray{1}(KX,s),HomogeneousArray{2}(KY,s),HomogeneousArray{3}(KZ,s))

    const global XRANGE = Base.OneTo(NX)
    const global RXRANGE = Base.OneTo(NRX)
    const global YRANGE = Base.OneTo(NY)
    const global ZRANGE = Base.OneTo(NZ)
    const global RANGEC = Base.OneTo(NX*NY*NZ)
 
end