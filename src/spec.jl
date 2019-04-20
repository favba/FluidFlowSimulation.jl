
@par function write_spec(s::A) where {A<:@par(Simulation)}
    i = s.iteration[]
    hout = s.hspec
    vout = s.vspec

    spectrum_non_linear(hout,vout,s.u,s.rhs)
    write("nl_h.spec.$i",hout)
    write("nl_v.spec.$i",vout)

    spectrum_viscosity(hout,vout,s.u,s)
    write("vis_h.spec.$i",hout)
    write("vis_v.spec.$i",vout)

    spectrum_pressure(hout,vout,s)
    write("press_h.spec.$i",hout)
    write("press_v.spec.$i",vout)

    if hasdensity(A)
        spectrum_buoyancy(vout,s.u,s.densitystratification.ρ,gravity(s))
        write("buoyancy_v.spec.$i",vout)
    end

    if hasles(A)
        spectrum_les(hout,vout,s.u,s.lesmodel.tau)
        write("les_h.spec.$i",hout)
        write("les_v.spec.$i",vout)
    end

    if hashyperviscosity(A)
        if HyperViscosityType <: SpectralBarrier
            spectrum_spectral_barrier(hout,vout,s.u,s.hyperviscosity)
        else
            spectrum_hyperviscosity(hout,vout,s.u,s)
        end
        write("hvis_h.spec.$i",hout)
        write("hvis_v.spec.$i",vout)
    end

end


vecouterproj(a,b) = Vec(proj(a.x,b.x),
                        proj(a.y,b.y),
                        proj(a.z,b.z))

function spectrum_non_linear(hout,vout,u,nl)
    @mthreads for k in ZRANGE
        @inbounds for j in YRANGE
            @simd for i in XRANGE
                out = vecouterproj(nl[i,j,k],u[i,j,k])
                hout[i,j,k] = out.x + out.y
                vout[i,j,k] = out.z
            end
        end
    end
end

function spectrum_les(hout,vout,u,τ)
    @mthreads for k in ZRANGE
        @inbounds for j in YRANGE
            @simd for i in XRANGE
                kh = K[i,j,k]
                out = vecouterproj((im*kh)⋅τ[i,j,k],u[i,j,k])
                hout[i,j,k] = out.x + out.y
                vout[i,j,k] = out.z
            end
        end
    end
end

function spectrum_buoyancy(out,u,ρ,g)
    @mthreads for k in ZRANGE
        @inbounds for j in YRANGE
            @simd for i in XRANGE
                out[i,j,k] = vecouterproj(u[i,j,k],g*ρ[i,j,k]).z
            end
        end
    end
end

function spectrum_spectral_barrier(hout,vout,u::VectorField{T},hv::SpectralBarrier{ini,cut,F}) where {T,F,ini,cut}
    @mthreads for l in ZRANGE
        @inbounds begin
            f = hv.func
            kz2 = KZ[l]*KZ[l]
            for j in YRANGE
                kyz2 = muladd(KY[j], KY[j], kz2)
                for i in XRANGE
                    k2 = muladd(KX[i], KX[i], kyz2)
                    k = sqrt(k2)
                    fk = f(k)
                    fk = ifelse(fk == -Inf, 0.0,fk)
                    v = u[i,j,l] 
                    out = fk*vecouterproj(v,v) 
                    hout[i,j,l] = out.x + out.y 
                    vout[i,j,l] = out.z
                end
            end
        end
    end
end

function spectrum_hyperviscosity(hout,vout,u::VectorField{T},s) where {T}
    @mthreads for l in ZRANGE
        mν = nuh(s)
        @inbounds begin
            M::Int = get_hyperviscosity_exponent(s)
            kz2 = KZ[l]*KZ[l]
            for j in YRANGE
                kyz2 = muladd(KY[j], KY[j], kz2)
                @simd for i in XRANGE
                    k2 = muladd(KX[i], KX[i], kyz2)
                    out = mν*(k2^M)*vecouterproj(u[i,j,l], u[i,j,l])
                    hout[i,j,l] = out.x + out.y 
                    vout[i,j,l] = out.z
                end
            end
        end
    end
end

function spectrum_viscosity(hout,vout,u::VectorField{T},s) where {T}
    @mthreads for l in ZRANGE
        mν = -ν
        @inbounds begin
            kz2 = KZ[l]*KZ[l]
            for j in YRANGE
                kyz2 = muladd(KY[j], KY[j], kz2)
                @simd for i in XRANGE
                    k2 = muladd(KX[i], KX[i], kyz2)
                    out = mν*k2*vecouterproj(u[i,j,l], u[i,j,l])
                    hout[i,j,l] = out.x + out.y 
                    vout[i,j,l] = out.z
                end
            end
        end
    end
end

@par function spectrum_pressure(hout,vout,s::@par(Simulation))
    @mthreads for k in ZRANGE
        spectrum_pressure(k,hout,vout,s)
    end
end

@par function spectrum_pressure(k::Int,hout,vout,s::A) where {A<:@par(Simulation)}
    u = s.u.c
    rhsv = s.rhs.c

    if hasles(A)
        τ = s.lesmodel.tau.c
    end

    if hasdensity(A)
        ρ = s.densitystratification.ρ.field
        g = gravity(s.densitystratification)
    end
    
    @inbounds for j in YRANGE
        @msimd for i in XRANGE

            kh = K[i,j,k]
            K2 = kh⋅kh
            v = u[i,j,k]
            rhs = rhsv[i,j,k]

            if hasles(A)
                if !(!is_SandP(A) && is_FakeSmagorinsky(A))
                    rhs += (im*kh)⋅τ[i,j,k]
                end
            end

            if hasdensity(A)
                rhs += ρ[i,j,k]*g
            end
            
            p1 = ifelse(k==j==i==1,zero(ComplexF64),-(kh⋅rhs)/K2)
            pressure = p1*kh
            out = vecouterproj(v,pressure)

            outxy = out.x + out.y
            outz = out.z

            hout[i,j,k] = outxy
            vout[i,j,k] = outz
        end
    end
end