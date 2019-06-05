function calc_cstar!(c,cm1,Δt,Δt2)
    @mthreads for j in TRANGE
        r = (Δt/Δt2)
        @inbounds @simd for i in REAL_RANGES[j]
            cstar = muladd(r,(c[i]-cm1[i]),c[i])
            cm1[i] = c[i]
            #c[i] = max(0.0,cstar)
            c[i] = cstar
        end
    end
end

function average_c!(c,Δ2)
    fullfourier!(c)

    @mthreads for k in Base.OneTo(NZ)
        @inbounds for j in Base.OneTo(NY)
            @msimd for i in Base.OneTo(NX)
                G = Gaussfilter(Δ2,i,j,k)
                c[i,j,k] *= G
            end
        end
    end

    real!(c)
    return nothing
end