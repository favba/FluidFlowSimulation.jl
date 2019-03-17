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

    statsheader(a::HyperViscosity) = "hdiss"

    stats(a::HyperViscosity,s::AbstractSimulation) = (hyperviscosity_stats(s.reduction,s.u,s),)

    msg(a::HyperViscosity{nh,M}) where {nh,M} = "\nHyper viscosity: νh = $(nh), m = $(M)\n\n"


struct SpectralBarrier{initp,endp,F} <: AbstractHyperViscosity
    func::F
    function SpectralBarrier(func,initk,endpk)
        @assert endpk >= initk
        return new{initk,endpk,typeof(func)}(func)
    end
end

SpectralBarrier(initk,endpk) = SpectralBarrier(x->log(esp(x,initk,endpk,Val{1/4}())),initk,endpk)

    statsheader(a::SpectralBarrier) = "hdiss"

    stats(a::SpectralBarrier,s::AbstractSimulation) = (spectral_barrier_stats(s.reduction,s.u,a),)

    msg(a::SpectralBarrier{initp,endp}) where {initp,endp} = "\nSpectral barrier: init = $(initp), cutoff = $(endp)\n\n"

@inline esp(k,i,p,::Val{n}) where n = abs(k) <= i ? 1.0 : abs(k) < p ? exp(-((abs(k) - i)^2)/((p)^n - abs(k)^n)) : 0.0
#@inline esp(k,i,p) = abs(k) <= i ? 1.0 : abs(k) < p ? exp(-((abs(k) - i)^2)/((p)^2 - k^2)) : 0.0

function spectral_barrier_stats(reduction,u::VectorField{T},hv::SpectralBarrier{ini,cut,F}) where {T,F,ini,cut}
    isrealspace(u) && fourier!(u)
    result = fill!(reduction,zero(T))
    @mthreads for l in ZRANGE
        @inbounds begin
            f = hv.func
            ee = zero(T)
            ii = Threads.threadid()
            kz2 = KZ[l]*KZ[l]
            for j in YRANGE
                kyz2 = muladd(KY[j], KY[j], kz2)
                for i in XRANGE
                    k2 = muladd(KX[i], KX[i], kyz2)
                    k = sqrt(k2)
                    magsq = mag2(u[i,j,l])
                    fk = f(k)
                    fk = ifelse(fk == -Inf, 0.0,fk)
                    ee += (1 + (i>1)) * fk*magsq 
                end
            end
            result[ii] += ee
        end
    end
    return -sum(result)
end