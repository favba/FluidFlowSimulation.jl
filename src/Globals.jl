__precompile__(false)
module Globals

using GlobalFileHelper, FluidTensors, FluidFields, StaticArrays

export LX,LY,LZ,NX,NY,NZ,NRRX,NRX,ν,DEALIAS_TYPE,KX,KY,KZ,THR,NT,TRANGE,REAL_RANGES,DEALIAS,K,RXRANGE,XRANGE,YRANGE,ZRANGE,RANGEC, COMPLEX_RANGES

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
    const global TRANGE = SOneTo(NT)
  
    const global REAL_RANGES = splitrange(NRRX*NY*NZ, NT)
    const global COMPLEX_RANGES = splitrange(NX*NY*NZ, NT)
  
    haskey(d,:cutoff) ? (cutoffr = Float64(eval(Meta.parse(d[:cutoff])))) : (cutoffr = 2/3)
    const global DEALIAS_TYPE = cutoffr

    cutoff = (cutoffr*kxp[end])^2
    cutoffx = (cutoffr*kxp[end])^2
    cutoffy = (cutoffr*maximum(abs,kyp))^2
    cutoffz = (cutoffr*maximum(abs,kzp))^2

    const global DEALIAS = BitArray(undef,(NX,NY,NZ))

    @. DEALIAS = (kxp^2 > cutoffx) | (kyp^2 > cutoffy) | (kzp^2 > cutoffz)

    const global KX = FluidFields.SRKvec(NRX,LX)#(kxp...,)
    const global KY = FluidFields.SKvec(NY,LY)#(kyp...,)
    const global KZ = FluidFields.SKvec(NZ,LZ)#(kzp...,)

    s = (NX,NY,NZ)
    const global K = VecArray(HomogeneousArray{1}(KX,s),HomogeneousArray{2}(KY,s),HomogeneousArray{3}(KZ,s))

    const global XRANGE = StaticArrays.SOneTo(NX)
    const global RXRANGE = StaticArrays.SOneTo(NRX)
    const global YRANGE = StaticArrays.SOneTo(NY)
    const global ZRANGE = StaticArrays.SOneTo(NZ)
    const global RANGEC = StaticArrays.SOneTo(NX*NY*NZ)
 
end