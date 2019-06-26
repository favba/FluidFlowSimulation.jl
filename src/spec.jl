
@par function write_spec(s::A) where {A<:@par(Simulation)}
    i = s.iteration[]
    hout = s.hspec
    vout = s.vspec

    outH = s.specH
    outV = s.specV
    tspecH = s.tspecH

    spectrum_u(hout,vout,s.u)
    writespectrum("u",i,hout,vout,outH,outV,tspecH)

    spectrum_viscosity(hout,vout,hout,vout,s)
    writespectrum("vis",i,hout,vout,outH,outV,tspecH)

    spectrum_non_linear(hout,vout,s.u,s.rhs)
    writespectrum("nl",i,hout,vout,outH,outV,tspecH)

    spectrum_pressure(hout,vout,s)
    writespectrum("press",i,hout,vout,outH,outV,tspecH)

    if hasdensity(A)
        spectrum_buoyancy(vout,s.u,s.densitystratification.ρ,gravity(s))
        write("buoyancy_v.spec3D.$i",vout)

        calculate_specHV(outH,outV,vout)

        writedlm("buoyancy_v.specH.$i",zip(KH,outH))
        writedlm("buoyancy_v.specV.$i",zip(KRZ,outV))

        outH .= (-).(outH ./ gravity(s.densitystratification).z) .* meangradient(s.densitystratification)
        outV .= (-).(outV ./ gravity(s.densitystratification).z) .* meangradient(s.densitystratification)

        writedlm("rhosource.specH.$i",zip(KH,outH))
        writedlm("rhosource.specV.$i",zip(KRZ,outV))
        
        spectrum_rho(hout,s.densitystratification.ρ)
        write("rho.spec3D.$i",hout)
    
        calculate_specHV(outH,outV,hout)
    
        writedlm("rho.specH.$i",zip(KH,outH))
        writedlm("rho.specV.$i",zip(KRZ,outV))

        spectrum_rhodiss(hout,hout,s)

        write("rhodiss.spec3D.$i",hout)
    
        calculate_specHV(outH,outV,hout)
    
        writedlm("rhodiss.specH.$i",zip(KH,outH))
        writedlm("rhodiss.specV.$i",zip(KRZ,outV))

        spectrum_rhonl(hout,s.densitystratification.ρ,s.densitystratification.flux)
        write("rhonl.spec3D.$i",hout)
    
        calculate_specHV(outH,outV,hout)
    
        writedlm("rhonl.specH.$i",zip(KH,outH))
        writedlm("rhonl.specV.$i",zip(KRZ,outV))


 
    end

    if hasles(A)
        spectrum_les(hout,vout,s.u,s.lesmodel.tau)
        writespectrum("les",i,hout,vout,outH,outV,tspecH)
    end

    if hashyperviscosity(A)
        if HyperViscosityType <: SpectralBarrier
            spectrum_spectral_barrier(hout,vout,s.u,s.hyperviscosity)
        else
            spectrum_hyperviscosity(hout,vout,s.u,s)
        end
        writespectrum("hvis",i,hout,vout,outH,outV,tspecH)
    end

end

@par function write_spec_forcing(s::A) where {A<:@par(Simulation)}
    i = s.iteration[]
    hout = s.hspec

    outH = s.specH
    outV = s.specV
    tspecH = s.tspecH

    fx = s.forcing.forcex
    fy = s.forcing.forcey
    ux = s.u.c.x
    uy = s.u.c.y
    dt = get_dt(s)
    spectrum_forcing(hout,ux,uy,fx,fy,dt)
    write("forcing_h.spec3D.$i",hout)

    calculate_specHV(outH,outV,hout)

    writedlm("forcing_h.specH.$i",zip(KH,outH))
    writedlm("forcing_h.specV.$i",zip(KRZ,outV))

    return nothing
end


function writespectrum(n::String,i::Integer,hout,vout,specH,specV,tspecH)
    write("$(n)_h.spec3D.$i",hout)
    write("$(n)_v.spec3D.$i",vout)

    calculate_specHV(specH,specV,hout)

    copy!(tspecH,specH)

    writedlm("$(n)_h.specH.$i",zip(KH,specH))
    writedlm("$(n)_h.specV.$i",zip(KRZ,specV))

    calculate_specHV(specH,specV,vout)

    tspecH .+= specH
    writedlm("$(n).specH.$i",zip(KH,tspecH))

    writedlm("$(n)_v.specH.$i",zip(KH,specH))
    writedlm("$(n)_v.specV.$i",zip(KRZ,specV))


    return nothing
end

function writedlm2D(n,out)
    open(n,"w+") do f
        for k in RZRANGE
            for j in eachindex(KH)
                write(f,join((KZ[k],KH[j],out[j,k]),'\t'),'\n')
            end
        end
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

function spectrum_rhonl(hout,rho,flux)
    @mthreads for k in ZRANGE
        @inbounds for j in YRANGE
            @simd for i in XRANGE
                kh = K[i,j,k]
                out = proj((im*kh)⋅flux[i,j,k],rho[i,j,k])
                hout[i,j,k] = out
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

function spectrum_forcing(out,ux,uy,fx,fy,dt)
    @mthreads for k in ZRANGE
        @inbounds for j in YRANGE
            @simd for i in XRANGE
                out[i,j,k] = 0.5*(proj(ux[i,j,k],fx[i,j,k]) + proj(uy[i,j,k],fy[i,j,k]))/dt
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

function spectrum_u(hout,vout,u::VectorField{T}) where {T}
    @mthreads for l in ZRANGE
        @inbounds begin
            for j in YRANGE
                @simd for i in XRANGE
                    out = vecouterproj(u[i,j,l], u[i,j,l])
                    hout[i,j,l] = (out.x + out.y)/2
                    vout[i,j,l] = out.z/2
                end
            end
        end
    end
end

function spectrum_rho(hout,rho::ScalarField{T}) where {T}
    @mthreads for l in ZRANGE
        @inbounds begin
            for j in YRANGE
                @simd for i in XRANGE
                    out = proj(rho[i,j,l], rho[i,j,l])
                    hout[i,j,l] = out
                end
            end
        end
    end
end

function spectrum_viscosity(hout,vout,hin,vin,s) where {T}
    @mthreads for l in ZRANGE
        mν = -ν
        @inbounds begin
            kz2 = KZ[l]*KZ[l]
            for j in YRANGE
                kyz2 = muladd(KY[j], KY[j], kz2)
                @simd for i in XRANGE
                    k2 = muladd(KX[i], KX[i], kyz2)
                    hout[i,j,l] = 2*mν*k2*hin[i,j,l]
                    vout[i,j,l] = 2*mν*k2*vin[i,j,l]
                end
            end
        end
    end
end

function spectrum_rhodiss(hout,hin,s) where {T}
    @mthreads for l in ZRANGE
        mν = -diffusivity(s.densitystratification)
        @inbounds begin
            kz2 = KZ[l]*KZ[l]
            for j in YRANGE
                kyz2 = muladd(KY[j], KY[j], kz2)
                @simd for i in XRANGE
                    k2 = muladd(KX[i], KX[i], kyz2)
                    hout[i,j,l] = mν*k2*hin[i,j,l]
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

function calculate_specHV(h,v,out)
    fill!(h,0.0)
    fill!(v,0.0)
    @inbounds for l in ZRANGE
        KZ2 = KZ[l]^2
        nk = round(Int, fsqrt(KZ2)/DKZ) + 1
        @inbounds for j in YRANGE
            KY2 = KY[j]^2
            nj = round(Int, fsqrt(KY2)/MAXDKH) + 1
            ee = out[1,j,l]
            h[1] += 0.5*ee
            h[nj] += 0.5*ee
            v[nk] += ee
            @msimd for i in 2:NX
                k = fsqrt(KX[i]*KX[i])

                nx = round(Int, k/MAXDKH) + 1

                ee = 2*out[i,j,l]
                h[nx] += 0.5*ee*MAXDKH
                h[nj] += 0.5*ee*MAXDKH
                v[nk] += ee*DKZ
 
            end
        end
    end
end