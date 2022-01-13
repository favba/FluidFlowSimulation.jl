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

    if (SPEC_FIL_D2 != 0.0) 
        open("Stats_fil.txt","w") do f
            write(f,statsheader(s))
        end
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
    results, lens, anum, results_fil = stats(s)

    open("Stats.txt","a+") do file 
        join(file,(init, time, results..., "\n"), ",","")
    end

    if (STAT_FIL)
        open("Stats_fil.txt","a+") do file
            join(file,(init, time, results_fil..., "\n"), ",","")
        end
    end

    open("Scales.txt","a+") do file 
        join(file,(init, time, lens..., "\n"), ",","")
    end

    open("Numbers.txt","a+") do file 
        join(file,(init, time, anum..., "\n"), ",","")
    end

end

function stats(s::AbstractSimulation)
    vstats, vstats_fil = velocity_stats(s)

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

    _otherstats = stats.(getfield.(Ref(s),sim_fields),Ref(s))
    otherstats = getindex.(_otherstats,1)
    otherstats_fil = getindex.(_otherstats,2)

    tdiss = vstats[13]
    tdiss_fil = vstats_fil[13]

    hashyperviscosity(s) && (tdiss += otherstats[5][3])
    hashyperviscosity(s) && (tdiss_fil += otherstats_fil[5][3])
    hasles(s) && (tdiss += otherstats[3][4])
    hasles(s) && (tdiss_fil += otherstats_fil[3][4])

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
    results_fil = (vstats_fil..., tdiss_fil, flatten(otherstats_fil)...)
    return results, lens, anums, results_fil
end
#  (velocity_stats(s)..., stats(s.passivescalar,s)..., stats(s.densitystratification,s)..., stats(s.lesmodel,s)..., stats(s.forcing,s)...)

@par function velocity_stats(s::@par(AbstractSimulation))
    if !STAT_FIL
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

        εh,εv,ε = viscosity_stats(s.reductionh,s.reductionv,s.u)

        nlh, nlv = s.nlstats[]

        pressh, pressv = s.pressstats[]

        return (u1, u2, u3, u12, u22, u32, kh, kv, kaniso, k, εh, εv, ε, nlh, nlv, pressh, pressv, -(nlh+pressh),(nlv+pressv)), ntuple(x->0.0,19)
    else
        u1 = real(s.u.c.x[1,1,1])
        u2 = real(s.u.c.y[1,1,1])
        u3 = real(s.u.c.z[1,1,1])

        u1_fil,u2_fil,u3_fil = (u1,u2,u3)

        u12,u12_fil = squared_mean(s.reductionh,s.reductionv, s.u.c.x)
        u22,u22_fil = squared_mean(s.reductionh,s.reductionv, s.u.c.y)
        u32,u32_fil = squared_mean(s.reductionh,s.reductionv, s.u.c.z)
        kh = (u12+u22)/2
        kh_fil = (u12_fil+u22_fil)/2
        kv = u32/2
        kv_fil = u32_fil/2
        k = (kh+kv)
        k_fil = (kh_fil+kv_fil)
        kaniso = 2*kv/kh
        kaniso_fil = 2*kv_fil/kh_fil

        (εh,εv,ε),(εh_fil,εv_fil,ε_fil) = viscosity_stats(s.reductionh,s.reductionv,s.reductionh_fil,s.reductionv_fil,s.u)

        #nlh, nlv = non_linear_stats(s.reductionh,s.reductionv,s.u,s.rhs)
        nlh, nlv,nlh_fil, nlv_fil = s.nlstats[]

        #pressh, pressv = pressure_stats(s.reductionh,s.reductionv,s)
        pressh, pressv,pressh_fil, pressv_fil = s.pressstats[]

        return ((u1, u2, u3, u12, u22, u32, kh, kv, kaniso, k, εh, εv, ε, nlh, nlv, pressh, pressv, -(nlh+pressh),(nlv+pressv)),
               (u1_fil, u2_fil, u3_fil, u12_fil, u22_fil, u32_fil, kh_fil, kv_fil, kaniso_fil, k_fil, εh_fil, εv_fil, ε_fil, nlh_fil, nlv_fil, pressh_fil, pressv_fil, -(nlh_fil+pressh_fil),(nlv_fil+pressv_fil)))
    end
end

@par function scalar_stats(ρ,s1,s::@par(AbstractSimulation))
    if !STAT_FIL
        rho = real(ρ[1,1,1])
        rho2 = squared_mean(s1.reduction,ρ)

        drd1 = dx_squared_mean(s1.reduction, ρ)
        drd2 = dy_squared_mean(s1.reduction, ρ)
        drd3 = dz_squared_mean(s1.reduction, ρ)

        rhodiss = diffusivity(s1)*(drd1 + drd2 + drd3)

        return (rho, rho2, drd1, drd2, drd3, rhodiss), ntuple(x->0.0,6)
    else
        rho = real(ρ[1,1,1])
        rho_fil = rho

        rho2, rho2_fil = squared_mean(s1.reduction,s.reductionh,ρ)

        drd1, drd1_fil = dx_squared_mean(s1.reduction,s.reductionh, ρ)
        drd2, drd2_fil = dy_squared_mean(s1.reduction,s.reductionh, ρ)
        drd3, drd3_fil = dz_squared_mean(s1.reduction,s.reductionh, ρ)

        rhodiss = diffusivity(s1)*(drd1 + drd2 + drd3)
        rhodiss_fil = diffusivity(s1)*(drd1_fil + drd2_fil + drd3_fil)

        return (rho, rho2, drd1, drd2, drd3, rhodiss), (rho_fil, rho2_fil, drd1_fil, drd2_fil, drd3_fil, rhodiss_fil)
    end
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
    @mthreads for ind in ZYRANGE
        jj,kk = Tuple(ind)
        l = ZRANGE[kk]
        j = YRANGE[jj]
        @inbounds begin
            ee = zero(T)
            ii = Threads.threadid()
                @simd for i in XRANGE
                    magsq = abs2(u[i,j,l])
                    ee += (1 + (i>1))*magsq 
                end
            result[ii] += ee
        end
    end
    return sum(result)
end

function squared_mean(reduction,reduction_fil, u::ScalarField{T}) where {T}
    isrealspace(u) && fourier!(u)
    result = fill!(reduction,zero(T))
    result_fil = fill!(reduction_fil,zero(T))
    @mthreads for ind in ZYRANGE
        jj,kk = Tuple(ind)
        l = ZRANGE[kk]
        j = YRANGE[jj]
        @inbounds begin
            ee = zero(T)
            ee_fil = zero(T)
            ii = Threads.threadid()
                @simd for i in XRANGE
                    magsq = abs2(u[i,j,l])
                    ee += (1 + (i>1))*magsq 
                    ee_fil += (1 + (i>1))*magsq*Gaussfilter(SPEC_FIL_D2,i,j,l)^2
                end
            result[ii] += ee
            result_fil[ii] += ee_fil
        end
    end
    return sum(result), sum(result_fil)
end


function dx_squared_mean(reduction,u::ScalarField{T}) where {T}
    isrealspace(u) && fourier!(u)
    result = fill!(reduction,zero(T))
    @mthreads for ind in ZYRANGE
        jj,kk = Tuple(ind)
        l = ZRANGE[kk]
        j = YRANGE[jj]
        @inbounds begin
            ee = zero(T)
            ii = Threads.threadid()
                @simd for i in XRANGE
                    magsq = abs2(im*KX[i]*u[i,j,l])
                    ee += (1 + (i>1))*magsq 
                end
            result[ii] += ee
        end
    end
    return sum(result)
end

function dx_squared_mean(reduction, reduction_fil, u::ScalarField{T}) where {T}
    isrealspace(u) && fourier!(u)
    result = fill!(reduction,zero(T))
    result_fil = fill!(reduction_fil,zero(T))
    @mthreads for ind in ZYRANGE
        jj,kk = Tuple(ind)
        l = ZRANGE[kk]
        j = YRANGE[jj]
        @inbounds begin
            ee = zero(T)
            ee_fil = zero(T)
            ii = Threads.threadid()
                @simd for i in XRANGE
                    magsq = abs2(im*KX[i]*u[i,j,l])
                    lee = (1 + (i>1))*magsq 
                    ee += lee
                    ee_fil = lee*Gaussfilter(SPEC_FIL_D2,i,j,l)^2
                end
            result[ii] += ee
            result_fil[ii] += ee_fil
        end
    end
    return sum(result), sum(result_fil)
end


function dy_squared_mean(reduction,u::ScalarField{T}) where {T}
    isrealspace(u) && fourier!(u)
    result = fill!(reduction,zero(T))
    @mthreads for ind in ZYRANGE
        jj,kk = Tuple(ind)
        l = ZRANGE[kk]
        j = YRANGE[jj]
        @inbounds begin
            ee = zero(T)
            ii = Threads.threadid()
                ky = KY[j]
                @simd for i in XRANGE
                    magsq = abs2(im*ky*u[i,j,l])
                    ee += (1 + (i>1))*magsq 
                end
            result[ii] += ee
        end
    end
    return sum(result)
end

function dy_squared_mean(reduction,reduction_fil, u::ScalarField{T}) where {T}
    isrealspace(u) && fourier!(u)
    result = fill!(reduction,zero(T))
    result_fil = fill!(reduction_fil,zero(T))
    @mthreads for ind in ZYRANGE
        jj,kk = Tuple(ind)
        l = ZRANGE[kk]
        j = YRANGE[jj]
        @inbounds begin
            ee = zero(T)
            ee_fil = zero(T)
            ii = Threads.threadid()
                ky = KY[j]
                @simd for i in XRANGE
                    magsq = abs2(im*ky*u[i,j,l])
                    lee = (1 + (i>1))*magsq 
                    ee += lee
                    ee_fil += lee*Gaussfilter(SPEC_FIL_D2,i,j,l)^2
                end
            result[ii] += ee
            result_fil[ii] += ee_fil
        end
    end
    return sum(result), sum(result_fil)
end


function dz_squared_mean(reduction,u::ScalarField{T}) where {T}
    isrealspace(u) && fourier!(u)
    result = fill!(reduction,zero(T))
    @mthreads for ind in ZYRANGE
        jj,kk = Tuple(ind)
        l = ZRANGE[kk]
        j = YRANGE[jj]
        @inbounds begin
            kz = KZ[l]
            ee = zero(T)
            ii = Threads.threadid()
                @simd for i in XRANGE
                    magsq = abs2(im*kz*u[i,j,l])
                    ee += (1 + (i>1))*magsq 
                end
            result[ii] += ee
        end
    end
    return sum(result)
end

function dz_squared_mean(reduction, reduction_fil, u::ScalarField{T}) where {T}
    isrealspace(u) && fourier!(u)
    result = fill!(reduction,zero(T))
    result_fil = fill!(reduction_fil,zero(T))
    @mthreads for ind in ZYRANGE
        jj,kk = Tuple(ind)
        l = ZRANGE[kk]
        j = YRANGE[jj]
        @inbounds begin
            kz = KZ[l]
            ee = zero(T)
            ee_fil = zero(T)
            ii = Threads.threadid()
                @simd for i in XRANGE
                    magsq = abs2(im*kz*u[i,j,l])
                    lee = (1 + (i>1))*magsq 
                    ee += lee
                    ee_fil += lee*Gaussfilter(SPEC_FIL_D2,i,j,l)^2
                end
            result[ii] += ee
            result_fil[ii] += ee_fil
        end
    end
    return sum(result), sum(result_fil)
end


@inline proj(u::Vec{<:Complex},v::Vec{<:Complex}) = proj(u.x,v.x) + proj(u.y,v.y) + proj(u.z,v.z)
@inline proj(a::SymTen{<:Complex},b::SymTen{<:Complex}) =
    proj(a.xx,b.xx) + proj(a.yy,b.yy) + proj(a.zz,b.zz) + 2*(proj(a.xy,b.xy) + proj(a.xz,b.xz) + proj(a.yz,b.yz))

function proj_mean(reduction,u::AbstractField{T},v::AbstractField{T}) where {T}
    isrealspace(u) && fourier!(u)
    isrealspace(v) && fourier!(v)
    result = fill!(reduction,zero(T))
    @mthreads for ind in ZYRANGE
        jj,kk = Tuple(ind)
        l = ZRANGE[kk]
        j = YRANGE[jj]
        @inbounds begin
            ee = zero(T)
            ii = Threads.threadid()
                @simd for i in XRANGE
                    magsq = proj(u[i,j,l],v[i,j,l])
                    ee += (1 + (i>1))*magsq 
                end
            result[ii] += ee
        end
    end
    return sum(result)
end

function proj_mean(reduction,reduction_fil,u::AbstractField{T},v::AbstractField{T}) where {T}
    isrealspace(u) && fourier!(u)
    isrealspace(v) && fourier!(v)
    result = fill!(reduction,zero(T))
    result_fil = fill!(reduction_fil,zero(T))
    @mthreads for ind in ZYRANGE
        jj,kk = Tuple(ind)
        l = ZRANGE[kk]
        j = YRANGE[jj]
        @inbounds begin
            ee = zero(T)
            ee_fil = zero(T)
            ii = Threads.threadid()
                @simd for i in XRANGE
                    magsq = (1 + (i>1))*proj(u[i,j,l],v[i,j,l])
                    ee += magsq 
                    ee_fil += magsq*Gaussfilter(SPEC_FIL_D2,i,j,l)^2
                end
            result[ii] += ee
            result_fil[ii] += ee_fil
        end
    end
    return sum(result), sum(result_fil)
end

@inline mag2(u::Vec{T}) where T<:Complex = muladd(u.x.re, u.x.re, muladd(u.x.im, u.x.im,
                                   muladd(u.y.re, u.y.re, muladd(u.y.im, u.y.im,
                                   muladd(u.z.re, u.z.re, u.z.im*u.z.im)))))

@inline mag2(u::Vec{T}) where T<:Real = u⋅u

function hyperviscosity_stats(reductionh,reductionv,u::VectorField{T},s) where {T}
    isrealspace(u) && fourier!(u)
    resulth = fill!(reductionh,zero(T))
    resultv = fill!(reductionv,zero(T))
    @mthreads for ind in ZYRANGE
        jj,kk = Tuple(ind)
        l = ZRANGE[kk]
        j = YRANGE[jj]
        @inbounds begin
            eeh = zero(T)
            eev = zero(T)
            ii = Threads.threadid()
            M::Int = get_hyperviscosity_exponent(s)
            kz2 = KZ[l]*KZ[l]
                kyz2 = muladd(KY[j], KY[j], kz2)
                @simd for i in XRANGE
                    k2 = muladd(KX[i], KX[i], kyz2)
                    uh = u[i,j,l]
                    eeh += (1 + (i>1)) * (k2^M)*(proj(uh.x,uh.x)+proj(uh.y,uh.y)) 
                    eev += (1 + (i>1)) * (k2^M)*proj(uh.z,uh.z) 
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

function hyperviscosity_stats(reductionh,reductionv,reductionh_fil,reductionv_fil,u::VectorField{T},s) where {T}
    isrealspace(u) && fourier!(u)
    resulth = fill!(reductionh,zero(T))
    resultv = fill!(reductionv,zero(T))
    resulth_fil = fill!(reductionh_fil,zero(T))
    resultv_fil = fill!(reductionv_fil,zero(T))
    @mthreads for ind in ZYRANGE
        jj,kk = Tuple(ind)
        l = ZRANGE[kk]
        j = YRANGE[jj]
        @inbounds begin
            eeh = zero(T)
            eev = zero(T)
            eeh_fil = zero(T)
            eev_fil = zero(T)
            ii = Threads.threadid()
            M::Int = get_hyperviscosity_exponent(s)
            kz2 = KZ[l]*KZ[l]
                kyz2 = muladd(KY[j], KY[j], kz2)
                @simd for i in XRANGE
                    k2 = muladd(KX[i], KX[i], kyz2)
                    uh = u[i,j,l]
                    leeh = (1 + (i>1)) * (k2^M)*(proj(uh.x,uh.x)+proj(uh.y,uh.y)) 
                    leev = (1 + (i>1)) * (k2^M)*proj(uh.z,uh.z) 
                    eeh += leeh
                    eev += leev

                    G2 = Gaussfilter(SPEC_FIL_D2,k2)^2
                    eeh_fil += leeh*G2
                    eev_fil += leev*G2
                end
            resulth[ii] += eeh
            resultv[ii] += eev
            resulth_fil[ii] += eeh_fil
            resultv_fil[ii] += eev_fil
        end
    end
    mν = nuh(s)
    eh = mν*sum(resulth)
    ev = mν*sum(resultv)
    eh_fil = mν*sum(resulth_fil)
    ev_fil = mν*sum(resultv_fil)
    return (eh,ev,eh+ev),(eh_fil,ev_fil,eh_fil+ev_fil)
end

function viscosity_stats(reductionh,reductionv,u::VectorField{T}) where {T}
    isrealspace(u) && fourier!(u)
    resulth = fill!(reductionh,zero(T))
    resultv = fill!(reductionv,zero(T))
    @mthreads for ind in ZYRANGE
        jj,kk = Tuple(ind)
        l = ZRANGE[kk]
        j = YRANGE[jj]
        @inbounds begin
            eeh = zero(T)
            eev = zero(T)
            ii = Threads.threadid()
            kz2 = KZ[l]*KZ[l]
                kyz2 = muladd(KY[j], KY[j], kz2)
                @simd for i in XRANGE
                    k2 = muladd(KX[i], KX[i], kyz2)
                    uh = u[i,j,l]
                    eeh += (1 + (i>1)) * k2*(proj(uh.x,uh.x)+proj(uh.y,uh.y)) 
                    eev += (1 + (i>1)) * k2*proj(uh.z,uh.z) 
                end
            resulth[ii] += eeh
            resultv[ii] += eev
        end
    end
    eh = ν*sum(resulth)
    ev = ν*sum(resultv)
    return eh,ev,eh+ev
end

function viscosity_stats(reductionh,reductionv,reductionh_fil,reductionv_fil,u::VectorField{T}) where {T}
    isrealspace(u) && fourier!(u)
    resulth = fill!(reductionh,zero(T))
    resultv = fill!(reductionv,zero(T))
    resulth_fil = fill!(reductionh_fil,zero(T))
    resultv_fil = fill!(reductionv_fil,zero(T))
    @mthreads for ind in ZYRANGE
        jj,kk = Tuple(ind)
        l = ZRANGE[kk]
        j = YRANGE[jj]
        @inbounds begin
            eeh = zero(T)
            eev = zero(T)
            eeh_fil = zero(T)
            eev_fil = zero(T)
            ii = Threads.threadid()
            kz2 = KZ[l]*KZ[l]
                kyz2 = muladd(KY[j], KY[j], kz2)
                @simd for i in XRANGE
                    k2 = muladd(KX[i], KX[i], kyz2)
                    uh = u[i,j,l]
                    leeh = (1 + (i>1)) * k2*(proj(uh.x,uh.x)+proj(uh.y,uh.y)) 
                    leev = (1 + (i>1)) * k2*proj(uh.z,uh.z) 
                    eeh += leeh
                    eev += leev
                    G2 = Gaussfilter(SPEC_FIL_D2,i,j,l)^2
                    eeh_fil += leeh*G2
                    eev_fil += leev*G2
                end
            resulth[ii] += eeh
            resultv[ii] += eev
            resulth_fil[ii] += eeh_fil
            resultv_fil[ii] += eev_fil
        end
    end
    eh = ν*sum(resulth)
    ev = ν*sum(resultv)
    eh_fil = ν*sum(resulth_fil)
    ev_fil = ν*sum(resultv_fil)
    return (eh,ev,eh+ev),(eh_fil,ev_fil,eh_fil+ev_fil)
end

function les_stats(reductionh,reductionv,τ::AbstractField{T},u::VectorField{T}) where {T}
    isrealspace(u) && fourier!(u)
    isrealspace(τ) && fourier!(τ)
    resulth = fill!(reductionh,zero(T))
    resultv = fill!(reductionv,zero(T))
    @mthreads for ind in ZYRANGE
        jj,kk = Tuple(ind)
        l = ZRANGE[kk]
        j = YRANGE[jj]
        @inbounds begin
            eeh = zero(T)
            eev = zero(T)
            ii = Threads.threadid()
                @simd for i in XRANGE
                    kh = K[i,j,l]
                    out = vecouterproj((im*kh)⋅τ[i,j,l],u[i,j,l])
                    eeh += (1 + (i>1))*(out.x + out.y) 
                    eev += (1 + (i>1))*out.z 
                end
            resulth[ii] += eeh
            resultv[ii] += eev
        end
    end
    prh = -sum(resulth)
    prv = -sum(resultv)
    return prh,prv,(2prv)/prh,prh+prv
end

function les_stats(reductionh,reductionv,reductionh_fil,reductionv_fil,τ::AbstractField{T},u::VectorField{T}) where {T}
    isrealspace(u) && fourier!(u)
    isrealspace(τ) && fourier!(τ)
    resulth = fill!(reductionh,zero(T))
    resultv = fill!(reductionv,zero(T))
    resulth_fil = fill!(reductionh_fil,zero(T))
    resultv_fil = fill!(reductionv_fil,zero(T))
    @mthreads for ind in ZYRANGE
        jj,kk = Tuple(ind)
        l = ZRANGE[kk]
        j = YRANGE[jj]
        @inbounds begin
            eeh = zero(T)
            eev = zero(T)
            eeh_fil = zero(T)
            eev_fil = zero(T)
            ii = Threads.threadid()
                @simd for i in XRANGE
                    kh = K[i,j,l]
                    out = vecouterproj((im*kh)⋅τ[i,j,l],u[i,j,l])
                    leeh = (1 + (i>1))*(out.x + out.y) 
                    leev = (1 + (i>1))*out.z 
                    eeh += leeh
                    eev += leev

                    G2 = Gaussfilter(SPEC_FIL_D2,i,j,l)^2
                    eeh_fil += leeh*G2
                    eev_fil += leev*G2
                end
            resulth[ii] += eeh
            resultv[ii] += eev
            resulth_fil[ii] += eeh_fil
            resultv_fil[ii] += eev_fil
        end
    end
    prh = -sum(resulth)
    prv = -sum(resultv)
    prh_fil = -sum(resulth_fil)
    prv_fil = -sum(resultv_fil)
    return (prh,prv,(2prv)/prh,prh+prv), (prh_fil,prv_fil,(2prv_fil)/prh_fil,prh_fil+prv_fil)
end


@par function pressure_stats(resulth,resultv,s::@par(Simulation))
    resulth = fill!(resulth,0.0)
    resultv = fill!(resultv,0.0)

    @mthreads for ind in ZYRANGE
        jj,kk = Tuple(ind)
        k = ZRANGE[kk]
        j = YRANGE[jj]
        pressure_stats(j,k,resulth,resultv,s)
    end

    return sum(resulth), sum(resultv), 0.0, 0.0
end

@par function pressure_stats(j::Int,k::Int,hr,vr,s::A) where {A<:@par(Simulation)}
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
    
        @inbounds @msimd for i in XRANGE

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

    hr[ii] += eeh
    vr[ii] += eev

    return nothing
end

@par function pressure_stats(resulth,resultv,resulth_fil,resultv_fil,s::@par(Simulation))
    resulth = fill!(resulth,0.0)
    resultv = fill!(resultv,0.0)
    resulth_fil = fill!(resulth_fil,0.0)
    resultv_fil = fill!(resultv_fil,0.0)

    @mthreads for ind in ZYRANGE
        jj,kk = Tuple(ind)
        k = ZRANGE[kk]
        j = YRANGE[jj]
        pressure_stats(j,k,resulth,resultv,resulth_fil,resultv_fil,s)
    end

    return sum(resulth), sum(resultv), sum(resulth_fil), sum(resultv_fil)
end

@par function pressure_stats(j::Int,k::Int,hr,vr,hr_fil,vr_fil,s::A) where {A<:@par(Simulation)}
    eeh = 0.0
    eev = 0.0
    eeh_fil = 0.0
    eev_fil = 0.0
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
    
        @inbounds @msimd for i in XRANGE

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

            leeh = (1 + (i>1))*outxy 
            leev = (1 + (i>1))*outz 

            eeh += leeh
            eev += leev

            G2 = Gaussfilter(SPEC_FIL_D2,K2)^2

            eeh += leeh*G2
            eev += leev*G2
        end

    hr[ii] += eeh
    vr[ii] += eev
    hr_fil[ii] += eeh_fil
    vr_fil[ii] += eev_fil

    return nothing
end


function non_linear_stats(hout,vout,u,nl,ke)
    fill!(hout,0.0)
    fill!(vout,0.0)
    @mthreads for ind in ZYRANGE
        jj,kk = Tuple(ind)
        k = ZRANGE[kk]
        j = YRANGE[jj]
        eeh = 0.0
        eev = 0.0
        ii = Threads.threadid()
            @simd for i in XRANGE
                ∇ = im*K[i,j,k]
                out = vecouterproj(nl[i,j,k] - ∇*ke[i,j,k],u[i,j,k])
                eeh += (1 + (i>1))*(out.x + out.y)
                eev += (1 + (i>1))*out.z
            end
        hout[ii]+=eeh
        vout[ii]+=eev
    end
    return sum(hout), sum(vout), 0.0, 0.0
end

function non_linear_stats(hout,vout,hout_fil,vout_fil,u,nl,ke)
    fill!(hout,0.0)
    fill!(vout,0.0)
    fill!(hout_fil,0.0)
    fill!(vout_fil,0.0)
    @mthreads for ind in ZYRANGE
        jj,kk = Tuple(ind)
        k = ZRANGE[kk]
        j = YRANGE[jj]
        eeh = 0.0
        eev = 0.0
        eeh_fil = 0.0
        eev_fil = 0.0
        ii = Threads.threadid()
            @simd for i in XRANGE
                ∇ = im*K[i,j,k]
                out = vecouterproj(nl[i,j,k] - ∇*ke[i,j,k],u[i,j,k])
                leeh = (1 + (i>1))*(out.x + out.y)
                leev = (1 + (i>1))*out.z
                eeh += leeh
                eev += leev
                G2 = Gaussfilter(SPEC_FIL_D2,i,j,k)^2
                eeh_fil += leeh*G2
                eev_fil += leev*G2
            end
        hout[ii]+=eeh
        vout[ii]+=eev
        hout_fil[ii]+=eeh_fil
        vout_fil[ii]+=eev_fil
    end
    return sum(hout), sum(vout), sum(hout_fil), sum(vout_fil)
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
    @mthreads for ind in ZYRANGE
        jj,kk = Tuple(ind)
        l = ZRANGE[kk]
        j = YRANGE[jj]
        @inbounds begin
            ee = zero(T)
            ii = Threads.threadid()
                @simd for i in XRANGE
                    kh = K[i,j,l]
                    out = proj((im*kh)⋅f[i,j,l],ρ[i,j,l])
                    ee += (1 + (i>1))*out
                end
            result[ii] += ee
        end
    end
    pr = -sum(result)
    return pr
end

function scalar_les_stats(reduction,reduction_fil, f::AbstractField{T},ρ::ScalarField{T}) where {T}
    isrealspace(ρ) && fourier!(ρ)
    isrealspace(f) && fourier!(f)
    result = fill!(reduction,zero(T))
    result_fil = fill!(reduction_fil,zero(T))
    @mthreads for ind in ZYRANGE
        jj,kk = Tuple(ind)
        l = ZRANGE[kk]
        j = YRANGE[jj]
        @inbounds begin
            ee = zero(T)
            ee_fil = zero(T)
            ii = Threads.threadid()
                @simd for i in XRANGE
                    kh = K[i,j,l]
                    out = proj((im*kh)⋅f[i,j,l],ρ[i,j,l])
                    lee = (1 + (i>1))*out
                    ee += lee
                    G2 = Gaussfilter(SPEC_FIL_D2,kh⋅kh)^2
                    ee_fil += lee*G2
                end
            result[ii] += ee
            result_fil[ii] += ee_fil
        end
    end
    pr = -sum(result)
    pr_fil = -sum(result_fil)
    return pr, pr_fil
end


function scalar_hvis_stats(reduction,u::ScalarField{T},s::HyperViscosity) where {T}
    isrealspace(u) && fourier!(u)
    result = fill!(reduction,zero(T))
    @mthreads for ind in ZYRANGE
        jj,kk = Tuple(ind)
        l = ZRANGE[kk]
        j = YRANGE[jj]
        @inbounds begin
            ee = zero(T)
            ii = Threads.threadid()
            M::Int = get_hyperviscosity_exponent(s)
            kz2 = KZ[l]*KZ[l]
                kyz2 = muladd(KY[j], KY[j], kz2)
                @simd for i in XRANGE
                    k2 = muladd(KX[i], KX[i], kyz2)
                    uh = u[i,j,l]
                    ee += (1 + (i>1)) * (k2^M)*proj(uh,uh) 
                end
            result[ii] += ee
        end
    end
    mν = nuh(s)
    eh = mν*sum(result)
    return eh
end

function scalar_hvis_stats(reduction,reduction_fil, u::ScalarField{T},s::HyperViscosity) where {T}
    isrealspace(u) && fourier!(u)
    result = fill!(reduction,zero(T))
    result_fil = fill!(reduction_fil,zero(T))
    @mthreads for ind in ZYRANGE
        jj,kk = Tuple(ind)
        l = ZRANGE[kk]
        j = YRANGE[jj]
        @inbounds begin
            ee = zero(T)
            ee_fil = zero(T)
            ii = Threads.threadid()
            M::Int = get_hyperviscosity_exponent(s)
            kz2 = KZ[l]*KZ[l]
                kyz2 = muladd(KY[j], KY[j], kz2)
                @simd for i in XRANGE
                    k2 = muladd(KX[i], KX[i], kyz2)
                    uh = u[i,j,l]
                    lee = (1 + (i>1)) * (k2^M)*proj(uh,uh) 
                    ee += lee
                    ee_fil += lee*Gaussfilter(SPEC_FIL_D2,i,j,l)^2
                end
            result[ii] += ee
            result_fil[ii] += ee_fil
        end
    end
    mν = nuh(s)
    eh = mν*sum(result)
    eh_fil = mν*sum(result_fil)
    return eh, eh_fil
end


function scalar_hvis_stats(reduction,u::ScalarField{T},hv::SpectralBarrier{ini,cut,F}) where {T,F,ini,cut}
    isrealspace(u) && fourier!(u)
    result = fill!(reduction,zero(T))
    @mthreads for ind in ZYRANGE
        jj,kk = Tuple(ind)
        l = ZRANGE[kk]
        j = YRANGE[jj]
        @inbounds begin
            f = hv.func
            ee = zero(T)
            ii = Threads.threadid()
            kz2 = KZ[l]*KZ[l]
                kyz2 = muladd(KY[j], KY[j], kz2)
                for i in XRANGE
                    k2 = muladd(KX[i], KX[i], kyz2)
                    k = sqrt(k2)
                    uh = u[i,j,l]
                    fk = f(k)
                    fk = ifelse(fk == -Inf, 0.0,fk)
                    ee += (1 + (i>1)) * fk*proj(uh,uh) 
                end
            result[ii] += ee
        end
    end
    diss = -sum(result)
    return diss
end

function scalar_hvis_stats(reduction, reduction_fil, u::ScalarField{T},hv::SpectralBarrier{ini,cut,F}) where {T,F,ini,cut}
    isrealspace(u) && fourier!(u)
    result = fill!(reduction,zero(T))
    result_fil = fill!(reduction_fil,zero(T))
    @mthreads for ind in ZYRANGE
        jj,kk = Tuple(ind)
        l = ZRANGE[kk]
        j = YRANGE[jj]
        @inbounds begin
            f = hv.func
            ee = zero(T)
            ee_fil = zero(T)
            ii = Threads.threadid()
            kz2 = KZ[l]*KZ[l]
                kyz2 = muladd(KY[j], KY[j], kz2)
                for i in XRANGE
                    k2 = muladd(KX[i], KX[i], kyz2)
                    k = sqrt(k2)
                    uh = u[i,j,l]
                    fk = f(k)
                    fk = ifelse(fk == -Inf, 0.0,fk)
                    lee = (1 + (i>1)) * fk*proj(uh,uh) 
                    ee += lee
                    ee_fil += lee*Gaussfilter(SPEC_FIL_D2,k2)^2
                end
            result[ii] += ee
            result_fil[ii] += ee_fil
        end
    end
    diss = -sum(result)
    diss_fil = -sum(result_fil)
    return diss, diss_fil
end