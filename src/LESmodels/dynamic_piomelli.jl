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