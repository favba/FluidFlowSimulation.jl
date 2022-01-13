abstract type AbstractHyperViscosity end

#statsheader(a::AbstractForcing) = ""

struct NoHyperViscosity <: AbstractHyperViscosity end

    statsheader(a::AbstractHyperViscosity) = ""

    stats(a::AbstractHyperViscosity,s::AbstractSimulation) = ()

    msg(a::NoHyperViscosity) = "\nHyper viscosity: no hyperviscosity\n\n"

    @inline nuh(::Type{NoHyperViscosity}) = nothing
    @inline nuh(a::AbstractHyperViscosity) = nuh(typeof(a))

    @inline get_hyperviscosity_exponent(::Type{NoHyperViscosity}) = nothing
    @inline get_hyperviscosity_exponent(a::AbstractHyperViscosity) = get_hyperviscosity_exponent(typeof(a))

struct HyperViscosity{νh,M} <: AbstractHyperViscosity
end

    @inline nuh(::Type{<:HyperViscosity{n,M}}) where {n,M} = n
    @inline get_hyperviscosity_exponent(::Type{<:HyperViscosity{n,M}}) where {n,M} = M

    statsheader(a::HyperViscosity) = "hdissh,hdissv,hdiss"

    stats(a::HyperViscosity,s::AbstractSimulation) = hyperviscosity_stats(s.reductionh,s.reductionv,s.u,s)

    msg(a::HyperViscosity{nh,M}) where {nh,M} = "\nHyper viscosity: νh = $(nh), m = $(M)\n\n"


struct SpectralBarrier{initp,endp,F} <: AbstractHyperViscosity
    func::F
    function SpectralBarrier(func,initk,endpk)
        @assert endpk >= initk
        return new{initk,endpk,typeof(func)}(func)
    end
end

SpectralBarrier(initk,endpk) = SpectralBarrier(x->log(esp(x,initk,endpk,Val{1/4}())),initk,endpk)

    statsheader(a::SpectralBarrier) = "hdissh,hdissv,hdiss"

    stats(a::SpectralBarrier,s::AbstractSimulation) = spectral_barrier_stats(s.reductionh,s.reductionv,s.u,a)

    msg(a::SpectralBarrier{initp,endp}) where {initp,endp} = "\nSpectral barrier: init = $(initp), cutoff = $(endp)\n\n"

@inline esp(k,i,p,::Val{n}) where n = abs(k) <= i ? 1.0 : abs(k) < p ? exp(-((abs(k) - i)^2)/((p)^n - abs(k)^n)) : 0.0
#@inline esp(k,i,p) = abs(k) <= i ? 1.0 : abs(k) < p ? exp(-((abs(k) - i)^2)/((p)^2 - k^2)) : 0.0

function spectral_barrier_stats(reductionh,reductionv,u::VectorField{T},hv::SpectralBarrier{ini,cut,F}) where {T,F,ini,cut}
    isrealspace(u) && fourier!(u)
    resulth = fill!(reductionh,zero(T))
    resultv = fill!(reductionv,zero(T))
    @mthreads for ind in ZYRANGE
        jj,kk = Tuple(ind)
        l = ZRANGE[kk]
        j = YRANGE[jj]
        @inbounds begin
            f = hv.func
            eeh = zero(T)
            eev = zero(T)
            ii = Threads.threadid()
            kz2 = KZ[l]*KZ[l]
                kyz2 = muladd(KY[j], KY[j], kz2)
                for i in XRANGE
                    k2 = muladd(KX[i], KX[i], kyz2)
                    k = sqrt(k2)
                    uh = u[i,j,l]
                    fk = f(k)
                    fk = ifelse(fk == -Inf, 0.0,fk)
                    eeh += (1 + (i>1)) * fk*(proj(uh.x,uh.x) + proj(uh.y,uh.y)) 
                    eev += (1 + (i>1)) * fk*proj(uh.z,uh.z) 
                end
            resulth[ii] += eeh
            resultv[ii] += eev
        end
    end
    dissh = -sum(resulth)
    dissv = -sum(resultv)
    return dissh,dissv,dissh+dissv
end