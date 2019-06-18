
@par function write_spec(s::A) where {A<:@par(Simulation)}
    i = s.iteration[]
    hout = s.hspec
    vout = s.vspec

    out1D = s.spec1D
    out2D = s.spec2D
    tspec1D = s.tspec1D

    spectrum_u(hout,vout,s.u)
    writespectrum("u",i,hout,vout,out1D,out2D,tspec1D)

    spectrum_viscosity(hout,vout,hout,vout,s)
    writespectrum("vis",i,hout,vout,out1D,out2D,tspec1D)

    spectrum_non_linear(hout,vout,s.u,s.rhs)
    writespectrum("nl",i,hout,vout,out1D,out2D,tspec1D)

    spectrum_pressure(hout,vout,s)
    writespectrum("press",i,hout,vout,out1D,out2D,tspec1D)

    if hasdensity(A)
        spectrum_buoyancy(vout,s.u,s.densitystratification.ρ,gravity(s))
        write("buoyancy_v.spec.$i",vout)

        calculate_spec12D(out1D,out2D,vout)

        writedlm("buoyancy_v.spec1D.$i",zip(K1D,out1D))
        writedlm2D("buoyancy_v.spec2D.$i",out2D)
    end

    if hasles(A)
        spectrum_les(hout,vout,s.u,s.lesmodel.tau)
        writespectrum("les",i,hout,vout,out1D,out2D,tspec1D)
    end

    if hashyperviscosity(A)
        if HyperViscosityType <: SpectralBarrier
            spectrum_spectral_barrier(hout,vout,s.u,s.hyperviscosity)
        else
            spectrum_hyperviscosity(hout,vout,s.u,s)
        end
        writespectrum("hvis",i,hout,vout,out1D,out2D,tspec1D)
    end

end

@par function write_spec_forcing(s::A) where {A<:@par(Simulation)}
    i = s.iteration[]
    hout = s.hspec

    out1D = s.spec1D
    out2D = s.spec2D
    tspec1D = s.tspec1D

    fx = s.forcing.forcex
    fy = s.forcing.forcey
    ux = s.u.c.x
    uy = s.u.c.y
    dt = get_dt(s)
    spectrum_forcing(hout,ux,uy,fx,fy,dt)
    write("forcing_h.spec3D.$i",hout)

    calculate_spec12D(out1D,out2D,hout)

    writedlm("forcing_h.spec1D.$i",zip(K1D,out1D))
    writedlm2D("forcing_h.spec2D.$i",out2D)

    return nothing
end


function writespectrum(n::String,i::Integer,hout,vout,spec1D,spec2D,tspec1D)
    write("$(n)_h.spec3D.$i",hout)
    write("$(n)_v.spec3D.$i",vout)

    calculate_spec12D(spec1D,spec2D,hout)

    copy!(tspec1D,spec1D)

    writedlm("$(n)_h.spec1D.$i",zip(K1D,spec1D))
    writedlm2D("$(n)_h.spec2D.$i",spec2D)

    calculate_spec12D(spec1D,spec2D,vout)

    tspec1D .+= spec1D

    writedlm("$(n)_v.spec1D.$i",zip(K1D,spec1D))
    writedlm2D("$(n)_v.spec2D.$i",spec2D)

    writedlm("$(n).spec1D.$i",zip(K1D,tspec1D))

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
                    hout[i,j,l] = out.x + out.y 
                    vout[i,j,l] = out.z
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
                    hout[i,j,l] = mν*k2*hin[i,j,l]
                    vout[i,j,l] = mν*k2*vin[i,j,l]
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

function calculate_spec12D(s1,s2,out)
    fill!(s1,0.0)
    fill!(s2,0.0)
    @inbounds for l in ZRANGE
        KZ2 = KZ[l]^2
        @inbounds for j in YRANGE
            KY2 = KY[j]^2
            KYZ2 = KY[j]^2 + KZ2
            n1 = round(Int, fsqrt(KYZ2)/MAXDK1D) + 1
            n2 = round(Int, fsqrt(KY2)/MAXDKH) + 1
            ee = out[1,j,l]
            s1[n1] += ee
            s2[n2,l] += ee
            @msimd for i in 2:NX
                k = fsqrt(muladd(KX[i],KX[i], KYZ2))
                kh = fsqrt(muladd(KX[i],KX[i], KY2))

                n1 = round(Int, k/MAXDK1D) + 1
                n2 = round(Int, kh/MAXDKH) + 1

                ee = 2*out[i,j,l]
                s1[n1] += ee
                s2[n2,l] += ee
            end
        end
    end
end