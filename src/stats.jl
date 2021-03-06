function statsheader(s::AbstractSimulation)
    simulationheader = "iteration,time,u1,u2,u3,u1p2,u2p2,u3p2,kh,kv,kaniso,k,dissh,dissv,diss,nlh,nlv,pressh,pressv,tnlh,tnlv,tdiss"
    header = join(Iterators.filter(x->x !== "",
      (simulationheader,statsheader.(getfield.(Ref(s),sim_fields))...,"\n")),
      ",","")
    return header
end 

function writeheader(s::AbstractSimulation)
    open("Stats.txt","w") do f
        write(f,statsheader(s))
    end

    open("Scales.txt","w") do f
        write(f,"iteration,time,Lk,Lt,lamg,laml,lamt,Lhh,Lvv,Lhv,Lvh,lam12,lam21,L11,L12,L13,L21,L22,L23,L31,L32,L33")
        if hasdensity(s)
            write(f,",Lb,Lo,Lom,Tk,Tt,Thh,Tb,Tbm\n")
        else
            write(f,",Tk,Tt,Thh\n")
        end
    end

    open("Numbers.txt","w") do f
        write(f,"iteration,time,Reh,Reh1,Relamg,Relaml,Relamt,Relamg1,Relaml1,Relamt1")
        if hasdensity(s)
            write(f,",Reb,Rebm,Frh,Frh1\n")
        else
            write(f,"\n")
        end
    end
end

function writestats(s::AbstractSimulation)
    init = s.iteration[]
    time = s.time[]
    results, lens, anum = stats(s)

    open("Stats.txt","a+") do file 
        join(file,(init, time, results..., "\n"), ",","")
    end

    open("Scales.txt","a+") do file 
        join(file,(init, time, lens..., "\n"), ",","")
    end

    open("Numbers.txt","a+") do file 
        join(file,(init, time, anum..., "\n"), ",","")
    end

end

function stats(s::AbstractSimulation)
    vstats = velocity_stats(s)

    u1p2 = vstats[4]
    u2p2 = vstats[5]
    u3p2 = vstats[6]
    uh = sqrt((u1p2+u2p2)/2)
    urms = sqrt((u1p2 + u2p2 + u3p2)/3)
    k = vstats[10]

    L11 = integral_lenght_x(s.u.c.x,u1p2)
    L12 = integral_lenght_y(s.u.c.x,u1p2)
    L13 = integral_lenght_z(s.u.c.x,u1p2)

    #λ12 = taylor_lenght_y(s.yspec,s.hspec,s.u.c.x,u1p2)
    du1dyp2 = dy_squared_mean(s.reductionh,s.u.c.x)
    λ12 = sqrt(2*u1p2/du1dyp2)
    #λ11 = taylor_lenght_x(s.xspec,s.u.c.x,u1p2)
    du1dxp2 = dx_squared_mean(s.reductionh,s.u.c.x)
    λ11 = sqrt(2*u1p2/du1dxp2)

    L21 = integral_lenght_x(s.u.c.y,u2p2)
    L22 = integral_lenght_y(s.u.c.y,u2p2)
    L23 = integral_lenght_z(s.u.c.y,u2p2)

    #λ21 = taylor_lenght_x(s.xspec,s.u.c.y,u2p2)
    du2dyp2 = dy_squared_mean(s.reductionh,s.u.c.y)
    λ22 = sqrt(2*u2p2/du2dyp2)
    #λ22 = taylor_lenght_y(s.yspec,s.hspec,s.u.c.y,u2p2)
    du2dxp2 = dx_squared_mean(s.reductionh,s.u.c.y)
    λ21 = sqrt(2*u2p2/du2dxp2)


    L31 = integral_lenght_x(s.u.c.z,u3p2)
    L32 = integral_lenght_y(s.u.c.z,u3p2)
    L33 = integral_lenght_z(s.u.c.z,u3p2)

    Lhh = (L11 + L22)/2
    Thh = Lhh/urms
    #λl = (λ11 + λ22)/2
    λl = sqrt(2*(u1p2+u2p2)/(du1dxp2+du2dyp2))
    #λt = (λ12 + λ21)/2
    λt = sqrt(2*(u1p2+u2p2)/(du1dyp2+du2dxp2))
    Lhv = (L13 + L23)/2
    Lvv = L33
    Lvh = (L31 + L32)/2

    otherstats = stats.(getfield.(Ref(s),sim_fields),Ref(s))

    tdiss = vstats[13]

    hashyperviscosity(s) && (tdiss += otherstats[5][3])
    hasles(s) && (tdiss += otherstats[3][4])

    Lt = sqrt(k*k*k)/tdiss # outer scale
    Tt = k/tdiss # outer time scale
    Lk = sqrt(sqrt(ν*ν*ν/tdiss)) # Kolmogorov length scale
    Tk = sqrt(ν/tdiss) # Kolmogorov timescale
    λg = sqrt(15*ν/tdiss)*urms

    lensv = (Lk,Lt,λg, λl,λt,Lhh,Lvv,Lhv,Lvh,λ12,λ21,L11,L12,L13,L21,L22,L23,L31,L32,L33)

    Reh = uh*Lhh/ν
    Reh1 = (Lhh/Lk)^(4/3)

    Reλg = urms*λg/ν
    Reλg1 = (λg/Lk)^(4/3)

    Reλl = urms*λl/ν
    Reλl1 = (λl/Lk)^(4/3)

    Reλt = urms*λt/ν
    Reλt1 = (λt/Lk)^(4/3)


    if hasdensity(s)
        Nb = calculate_N(s.densitystratification)
        Lb = 2π*urms/Nb # Buoyancy lenght scale
        Lo = sqrt(tdiss/(Nb*Nb*Nb)) # Ozmidov lenght scale
        Lom = sqrt(tdiss/(Nb*Nb*Nb/(8*pi*pi*pi))) # Ozmidov lenght scale
        Tb = 2π/Nb # Buoyancy time scale
        Tbm = 1/Nb # Buoyancy time scale
        Reb = (Lo/Lk)^(4/3)
        Rebm = (Lom/Lk)^(4/3)
        Frh = 2π*uh/(Nb*Lhh)
        Frh1 = Lhv/Lhh
        lens = (lensv...,Lb,Lo,Lom,Tk,Tt,Thh,Tb,Tbm)
        anums = (Reh,Reh1, Reλg, Reλl, Reλt, Reλg1, Reλl1, Reλt1, Reb, Rebm, Frh,Frh1)
    else
        lens = (lensv...,Tk,Tt,Thh)
        anums = (Reh, Reh1, Reλl, Reλt, Reλl1, Reλt1)
    end

    results = (vstats..., tdiss, flatten(otherstats)...)
    return results, lens, anums
end
#  (velocity_stats(s)..., stats(s.passivescalar,s)..., stats(s.densitystratification,s)..., stats(s.lesmodel,s)..., stats(s.forcing,s)...)

@par function velocity_stats(s::@par(AbstractSimulation))
    u1 = real(s.u.c.x[1,1,1])
    u2 = real(s.u.c.y[1,1,1])
    u3 = real(s.u.c.z[1,1,1])

    u12 = squared_mean(s.reductionh, s.u.c.x)
    u22 = squared_mean(s.reductionh, s.u.c.y)
    u32 = squared_mean(s.reductionh, s.u.c.z)
    kh = (u12+u22)/2
    kv = u32/2
    k = (kh+kv)
    kaniso = 2*kv/kh

#    d1d1 = dx_squared_mean(s.reduction, s.u.c.x)
#    d1d2 = dy_squared_mean(s.reduction, s.u.c.x)
#    d1d3 = dz_squared_mean(s.reduction, s.u.c.x)
#
#    d2d1 = dx_squared_mean(s.reduction, s.u.c.y)
#    d2d2 = dy_squared_mean(s.reduction, s.u.c.y)
#    d2d3 = dz_squared_mean(s.reduction, s.u.c.y)
#
#    d3d1 = dx_squared_mean(s.reduction, s.u.c.z)
#    d3d2 = dy_squared_mean(s.reduction, s.u.c.z)
#    d3d3 = dz_squared_mean(s.reduction, s.u.c.z)

    εh,εv,ε = viscosity_stats(s.reductionh,s.reductionv,s.u)

    #nlh, nlv = non_linear_stats(s.reductionh,s.reductionv,s.u,s.rhs)
    nlh, nlv = s.nlstats[]

    #pressh, pressv = pressure_stats(s.reductionh,s.reductionv,s)
    pressh, pressv = s.pressstats[]

    return u1, u2, u3, u12, u22, u32, kh, kv, kaniso, k, εh, εv, ε, nlh, nlv, pressh, pressv, -(nlh+pressh),(nlv+pressv)
end

@par function scalar_stats(ρ,s1,s::@par(AbstractSimulation))
    rho = real(ρ[1,1,1])
    rho2 = squared_mean(s1.reduction,ρ)

    drd1 = dx_squared_mean(s1.reduction, ρ)
    drd2 = dy_squared_mean(s1.reduction, ρ)
    drd3 = dz_squared_mean(s1.reduction, ρ)

    rhodiss = diffusivity(s1)*(drd1 + drd2 + drd3)

    return rho, rho2, drd1, drd2, drd3, rhodiss
end

#ape(s::AbstractSimulation) = tmean(x->x^2,parent(real(s.ρ)),s)

#@par function tmean(f::F,x::AbstractArray{T,3},s::@par(AbstractSimulation)) where {F<:Function,T<:Number}
#
#    result = fill!(s.reduction,0.0)
#    @mthreads for k in ZRANGE
#        for j in YRANGE
#            @inbounds @msimd for i in RXRANGE
#                result[Threads.threadid()] += f(x[i,j,k])
#            end
#        end
#    end
#
#    return sum(result)/(NRX*NY*NZ)
#end

#tmean(x::AbstractArray,s::AbstractSimulation) = tmean(identity,x,s)

#@generated function getind(f::F,x::NT,i::Int,j::Int,k::Int) where {F<:Function,N,NT<:NTuple{N,AbstractArray}}
#    args = Array{Any,1}(undef,N)
#    for l=1:N 
#       args[l] = :(x[$l][i,j,k])
#    end
#    ex = Expr(:call,:f,args...) 
#    return Expr(:block,Expr(:meta,:inline),Expr(:meta,:propagate_inbounds),ex)
#end
#
#@par function tmean(f::F,x::NT,reduction::L) where {F<:Function,N,NT<:NTuple{N,AbstractArray},L}
#    result = fill!(reduction,0.0)
#    @mthreads for k in ZRANGE
#        for j in YRANGE
#            @inbounds @msimd for i in RXRANGE
#                result[Threads.threadid()] += getind(f,x,i,j,k)
#            end
#        end
#    end
#    return sum(result)/(NRX*NY*NZ)
#end

function squared_mean(reduction,u::ScalarField{T}) where {T}
    isrealspace(u) && fourier!(u)
    result = fill!(reduction,zero(T))
    @mthreads for l in ZRANGE
        @inbounds begin
            ee = zero(T)
            ii = Threads.threadid()
            @inbounds for j in YRANGE
                @simd for i in XRANGE
                    magsq = abs2(u[i,j,l])
                    ee += (1 + (i>1))*magsq 
                end
            end
            result[ii] += ee
        end
    end
    return sum(result)
end

function dx_squared_mean(reduction,u::ScalarField{T}) where {T}
    isrealspace(u) && fourier!(u)
    result = fill!(reduction,zero(T))
    @mthreads for l in ZRANGE
        @inbounds begin
            ee = zero(T)
            ii = Threads.threadid()
            @inbounds for j in YRANGE
                @simd for i in XRANGE
                    magsq = abs2(im*KX[i]*u[i,j,l])
                    ee += (1 + (i>1))*magsq 
                end
            end
            result[ii] += ee
        end
    end
    return sum(result)
end

function dy_squared_mean(reduction,u::ScalarField{T}) where {T}
    isrealspace(u) && fourier!(u)
    result = fill!(reduction,zero(T))
    @mthreads for l in ZRANGE
        @inbounds begin
            ee = zero(T)
            ii = Threads.threadid()
            @inbounds for j in YRANGE
                ky = KY[j]
                @simd for i in XRANGE
                    magsq = abs2(im*ky*u[i,j,l])
                    ee += (1 + (i>1))*magsq 
                end
            end
            result[ii] += ee
        end
    end
    return sum(result)
end

function dz_squared_mean(reduction,u::ScalarField{T}) where {T}
    isrealspace(u) && fourier!(u)
    result = fill!(reduction,zero(T))
    @mthreads for l in ZRANGE
        @inbounds begin
            ee = zero(T)
            ii = Threads.threadid()
            kz = KZ[l]
            @inbounds for j in YRANGE
                @simd for i in XRANGE
                    magsq = abs2(im*kz*u[i,j,l])
                    ee += (1 + (i>1))*magsq 
                end
            end
            result[ii] += ee
        end
    end
    return sum(result)
end

@inline proj(u::Vec{<:Complex},v::Vec{<:Complex}) = proj(u.x,v.x) + proj(u.y,v.y) + proj(u.z,v.z)
@inline proj(a::SymTen{<:Complex},b::SymTen{<:Complex}) =
    proj(a.xx,b.xx) + proj(a.yy,b.yy) + proj(a.zz,b.zz) + 2*(proj(a.xy,b.xy) + proj(a.xz,b.xz) + proj(a.yz,b.yz))

function proj_mean(reduction,u::AbstractField{T},v::AbstractField{T}) where {T}
    isrealspace(u) && fourier!(u)
    isrealspace(v) && fourier!(v)
    result = fill!(reduction,zero(T))
    @mthreads for l in ZRANGE
        @inbounds begin
            ee = zero(T)
            ii = Threads.threadid()
            @inbounds for j in YRANGE
                @simd for i in XRANGE
                    magsq = proj(u[i,j,l],v[i,j,l])
                    ee += (1 + (i>1))*magsq 
                end
            end
            result[ii] += ee
        end
    end
    return sum(result)
end

@inline mag2(u::Vec{T}) where T<:Complex = muladd(u.x.re, u.x.re, muladd(u.x.im, u.x.im,
                                   muladd(u.y.re, u.y.re, muladd(u.y.im, u.y.im,
                                   muladd(u.z.re, u.z.re, u.z.im*u.z.im)))))

@inline mag2(u::Vec{T}) where T<:Real = u⋅u

function hyperviscosity_stats(reductionh,reductionv,u::VectorField{T},s) where {T}
    isrealspace(u) && fourier!(u)
    resulth = fill!(reductionh,zero(T))
    resultv = fill!(reductionv,zero(T))
    @mthreads for l in ZRANGE
        @inbounds begin
            eeh = zero(T)
            eev = zero(T)
            ii = Threads.threadid()
            M::Int = get_hyperviscosity_exponent(s)
            kz2 = KZ[l]*KZ[l]
            for j in YRANGE
                kyz2 = muladd(KY[j], KY[j], kz2)
                @simd for i in XRANGE
                    k2 = muladd(KX[i], KX[i], kyz2)
                    uh = u[i,j,l]
                    eeh += (1 + (i>1)) * (k2^M)*(proj(uh.x,uh.x)+proj(uh.y,uh.y)) 
                    eev += (1 + (i>1)) * (k2^M)*proj(uh.z,uh.z) 
                end
            end
            resulth[ii] += eeh
            resultv[ii] += eev
        end
    end
    mν = nuh(s)
    eh = mν*sum(resulth)
    ev = mν*sum(resultv)
    return eh,ev,eh+ev
end

function viscosity_stats(reductionh,reductionv,u::VectorField{T}) where {T}
    isrealspace(u) && fourier!(u)
    resulth = fill!(reductionh,zero(T))
    resultv = fill!(reductionv,zero(T))
    @mthreads for l in ZRANGE
        @inbounds begin
            eeh = zero(T)
            eev = zero(T)
            ii = Threads.threadid()
            kz2 = KZ[l]*KZ[l]
            for j in YRANGE
                kyz2 = muladd(KY[j], KY[j], kz2)
                @simd for i in XRANGE
                    k2 = muladd(KX[i], KX[i], kyz2)
                    uh = u[i,j,l]
                    eeh += (1 + (i>1)) * k2*(proj(uh.x,uh.x)+proj(uh.y,uh.y)) 
                    eev += (1 + (i>1)) * k2*proj(uh.z,uh.z) 
                end
            end
            resulth[ii] += eeh
            resultv[ii] += eev
        end
    end
    eh = ν*sum(resulth)
    ev = ν*sum(resultv)
    return eh,ev,eh+ev
end

function les_stats(reductionh,reductionv,τ::AbstractField{T},u::VectorField{T}) where {T}
    isrealspace(u) && fourier!(u)
    isrealspace(τ) && fourier!(τ)
    resulth = fill!(reductionh,zero(T))
    resultv = fill!(reductionv,zero(T))
    @mthreads for l in ZRANGE
        @inbounds begin
            eeh = zero(T)
            eev = zero(T)
            ii = Threads.threadid()
            @inbounds for j in YRANGE
                @simd for i in XRANGE
                    kh = K[i,j,l]
                    out = vecouterproj((im*kh)⋅τ[i,j,l],u[i,j,l])
                    eeh += (1 + (i>1))*(out.x + out.y) 
                    eev += (1 + (i>1))*out.z 
                end
            end
            resulth[ii] += eeh
            resultv[ii] += eev
        end
    end
    prh = -sum(resulth)
    prv = -sum(resultv)
    return prh,prv,(2prv)/prh,prh+prv
end


@par function pressure_stats(resulth,resultv,s::@par(Simulation))
    resulth = fill!(resulth,0.0)
    resultv = fill!(resultv,0.0)

    @mthreads for k in ZRANGE
        pressure_stats(k,resulth,resultv,s)
    end

    return sum(resulth), sum(resultv)
end

@par function pressure_stats(k::Int,hr,vr,s::A) where {A<:@par(Simulation)}
    eeh = 0.0
    eev = 0.0
    ii = Threads.threadid()

    u = s.u.c
    rhsv = s.rhs.c
    ke = s.kspec.c

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
            rhs = rhsv[i,j,k] - (im*ke[i,j,k])*kh

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

            eeh += (1 + (i>1))*outxy 
            eev += (1 + (i>1))*outz 
        end
    end

    hr[ii] += eeh
    vr[ii] += eev

    return nothing
end

function non_linear_stats(hout,vout,u,nl,ke)
    fill!(hout,0.0)
    fill!(vout,0.0)
    @mthreads for k in ZRANGE
        eeh = 0.0
        eev = 0.0
        ii = Threads.threadid()
        @inbounds for j in YRANGE
            @simd for i in XRANGE
                ∇ = im*K[i,j,k]
                out = vecouterproj(nl[i,j,k] - ∇*ke[i,j,k],u[i,j,k])
                eeh += (1 + (i>1))*(out.x + out.y)
                eev += (1 + (i>1))*out.z
            end
        end
        hout[ii]+=eeh
        vout[ii]+=eev
    end
    return sum(hout), sum(vout)
end

function integral_lenght_x(u,u1p2)
    r = zero(typeof(proj(u[1],u[1])))
    @inbounds for k in ZRANGE
        for j in YRANGE
            r += proj(u[1,j,k],u[1,j,k])
        end
    end
    return LX*r*pi/u1p2
end

function integral_lenght_y(u,u1p2)
    r = zero(typeof(proj(u[1],u[1])))
    @inbounds for k in ZRANGE
        @msimd for i in XRANGE
            r += (1+(i>1))*proj(u[i,1,k],u[i,1,k])
        end
    end
    return LY*r*pi/u1p2
end

function integral_lenght_z(u,u1p2)
    r = zero(typeof(proj(u[1],u[1])))
    @inbounds for j in YRANGE
        @msimd for i in XRANGE
            r += (1+(i>1))*proj(u[i,j,1],u[i,j,1])
        end
    end
    return LZ*r*pi/u1p2
end

function taylor_lenght_x(re,u,u1p2)

    fill!(re.data,0.0)

    @inbounds for k in ZRANGE
        for j in YRANGE
            for i in XRANGE
                re[i] += proj(u[i,j,k],u[i,j,k])/u1p2
            end
        end
    end

    @inbounds for i in XRANGE
        re[i] *= -KX[i]*KX[i]
    end

    brfft!(re)

    return sqrt(-2/re.data[1])
end

function taylor_lenght_y(re,hpart,u,u1p2)

    fill!(re.data,0.0)
    @inbounds for j in YRANGE
        hpart[1,j,1] = 0.0
    end

    @inbounds for k in ZRANGE
        for j in YRANGE
            for i in XRANGE
                hpart[1,j,1] += proj(u[i,j,k],u[i,j,k])/u1p2
            end
        end
    end

    re[1] = 0.0 + 0.0im

    ep = div(NY,2) + 1

    re[ep] = KY[ep]^2*hpart[1,ep,1]
    @inbounds for i in 2:(ep-1)
        re[i] = (-KY[NY-i]*KY[NY-i])*(hpart[1,i,1] + hpart[1,NY-i,1])/2
    end

    brfft!(re)

    return sqrt(-2/re.data[1])
end

function scalar_les_stats(reduction,f::AbstractField{T},ρ::ScalarField{T}) where {T}
    isrealspace(ρ) && fourier!(ρ)
    isrealspace(f) && fourier!(f)
    result = fill!(reduction,zero(T))
    @mthreads for l in ZRANGE
        @inbounds begin
            ee = zero(T)
            ii = Threads.threadid()
            @inbounds for j in YRANGE
                @simd for i in XRANGE
                    kh = K[i,j,l]
                    out = proj((im*kh)⋅f[i,j,l],ρ[i,j,l])
                    ee += (1 + (i>1))*out
                end
            end
            result[ii] += ee
        end
    end
    pr = -sum(result)
    return pr
end

function scalar_hvis_stats(reduction,u::ScalarField{T},s::HyperViscosity) where {T}
    isrealspace(u) && fourier!(u)
    result = fill!(reduction,zero(T))
    @mthreads for l in ZRANGE
        @inbounds begin
            ee = zero(T)
            ii = Threads.threadid()
            M::Int = get_hyperviscosity_exponent(s)
            kz2 = KZ[l]*KZ[l]
            for j in YRANGE
                kyz2 = muladd(KY[j], KY[j], kz2)
                @simd for i in XRANGE
                    k2 = muladd(KX[i], KX[i], kyz2)
                    uh = u[i,j,l]
                    ee += (1 + (i>1)) * (k2^M)*proj(uh,uh) 
                end
            end
            result[ii] += ee
        end
    end
    mν = nuh(s)
    eh = mν*sum(result)
    return eh
end

function scalar_hvis_stats(reduction,u::ScalarField{T},hv::SpectralBarrier{ini,cut,F}) where {T,F,ini,cut}
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
                    uh = u[i,j,l]
                    fk = f(k)
                    fk = ifelse(fk == -Inf, 0.0,fk)
                    ee += (1 + (i>1)) * fk*proj(uh,uh) 
                end
            end
            result[ii] += ee
        end
    end
    diss = -sum(result)
    return diss
end