__precompile__(false)
module Globals

using GlobalFileHelper, FluidTensors, FluidFields, StaticArrays

export LX,LY,LZ,NX,NY,NZ,NRRX,NRX,ν,DEALIAS_TYPE,KX,KY,KZ,THR,NT,TRANGE,REAL_RANGES,DEALIAS,K,RXRANGE,XRANGE,YRANGE,ZRANGE,RANGEC, COMPLEX_RANGES
export RYRANGE, RZRANGE, XRANGE2X, KH, MAXDKH, KRZ, DKZ

include("CatVec.jl")

function nshells(kx,ky,kz)
    maxdk = max(kx[2],ky[2],kz[2])
    n = round(Int,sqrt(kx[end]^2 + maximum(x->x*x,ky) + maximum(x->x*x,kz))/maxdk) + 1
    return n, maxdk
end

function nshells2D(kx,ky)
    maxdk = max(kx[2],ky[2])
    n = round(Int,sqrt(kx[end]^2 + maximum(x->x*x,ky))/maxdk) + 1
    return n, maxdk
end

function compute_shells(kx::AbstractVector{T},ky::AbstractVector,kz::AbstractVector) where {T}
    Nx = length(kx)
    Ny = length(ky)
    Nz = length(kz)
    nShells, maxdk = nshells(kx,ky,kz)
    kh = zeros(T,nShells)
    numPtsInShell = zeros(Int,nShells)

    @inbounds for k in 1:Nz
        kz2 = kz[k]^2
        for j=1:Ny
            kzy2 = kz2 + ky[j]^2
            for i=1:Nx
                K = sqrt(muladd(kx[i],kx[i],kzy2))
                ii = round(Int,K/maxdk)+1
                kh[ii] += K
                numPtsInShell[ii] += 1
            end
        end
    end
  
    @inbounds @simd for i in 1:length(kh)
        kh[i] = kh[i]/numPtsInShell[i]
    end

    return kh
end

function compute_shells2D(kx::AbstractVector{T},ky) where {T}
    Nx = length(kx)
    Ny = length(ky)
    nShells2D, maxdk2D = nshells2D(kx,ky)
    kh = zeros(T,nShells2D)
    numPtsInShell2D = zeros(Int,nShells2D)

    @inbounds for j=1:Ny
        ky2 = ky[j]^2
        for i=1:Nx
            K = sqrt(muladd(kx[i],kx[i], ky2))
            ii = round(Int,K/maxdk2D)+1
            kh[ii] += K
            numPtsInShell2D[ii] += 1
        end
    end
  
    @inbounds @simd for i in 1:length(kh)
        kh[i] = kh[i]/numPtsInShell2D[i]
    end

    return kh
end

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

    const global KH = (KX[2] >= KY[2]) ? KX : FluidFields.SRKvec(NY,LY)
    const global MAXDKH = KH[2]

    const global KRZ = FluidFields.SRKvec(NZ,LZ)
    const global DKZ = KRZ[2]


    s = (NX,NY,NZ)
    const global K = VecArray(HomogeneousArray{1}(KX,s),HomogeneousArray{2}(KY,s),HomogeneousArray{3}(KZ,s))

    const global RXRANGE = StaticArrays.SOneTo(NRX)
    const global RYRANGE = StaticArrays.SOneTo(NY)
    const global RZRANGE = StaticArrays.SOneTo(NZ)
    const global RANGEC = StaticArrays.SOneTo(NX*NY*NZ)

    const global XRANGE = StaticArrays.SOneTo(findfirst(view(DEALIAS,:,1,1))-1)
    const global XRANGE2X = StaticArrays.SOneTo(2length(XRANGE))
    const global YRANGE = CatVec(StaticArrays.SOneTo(findfirst(view(DEALIAS,1,:,1))-1), 
                                 StaticArrays.SUnitRange(NY-findfirst(view(DEALIAS,1,:,1)) ,NY))

    const global ZRANGE = CatVec(StaticArrays.SOneTo(findfirst(view(DEALIAS,1,1,:))-1), 
                                 StaticArrays.SUnitRange(NZ-findfirst(view(DEALIAS,1,1,:)) ,NZ))
 
end