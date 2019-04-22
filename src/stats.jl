function statsheader(s::AbstractSimulation)
    simulationheader = "iteration,time,u1,u2,u3,u1p2,u2p2,u3p2,k,du1dx1p2,du1dx2p2,du1dx3p2,du2dx1p2,du2dx2p2,du2dx3p2,du3dx1p2,du3dx2p2,du3dx3p2,diss"
    header = join(Iterators.filter(x->x !== "",
      (simulationheader,statsheader.(getfield.(Ref(s),sim_fields))...,"\n")),
      ",","")
    return header
end 

function writeheader(s::AbstractSimulation)
    open("Stats.txt","w") do f
        write(f,statsheader(s))
    end
end

function writestats(s::AbstractSimulation)
    init = s.iteration[]
    time = s.time[]
    results = stats(s)
    open("Stats.txt","a+") do file 
        join(file,(init, time, results..., "\n"), ",","")
    end
end

stats(s::AbstractSimulation) = 
    (velocity_stats(s)..., flatten(stats.(getfield.(Ref(s),sim_fields),Ref(s)))...)
#  (velocity_stats(s)..., stats(s.passivescalar,s)..., stats(s.densitystratification,s)..., stats(s.lesmodel,s)..., stats(s.forcing,s)...)

@par function velocity_stats(s::@par(AbstractSimulation))
    u1 = real(s.u.c.x[1,1,1])
    u2 = real(s.u.c.y[1,1,1])
    u3 = real(s.u.c.z[1,1,1])

    u12 = squared_mean(s.reduction, s.u.c.x)
    u22 = squared_mean(s.reduction, s.u.c.y)
    u32 = squared_mean(s.reduction, s.u.c.z)
    k = (u12+u22+u32)/2

    d1d1 = dx_squared_mean(s.reduction, s.u.c.x)
    d1d2 = dy_squared_mean(s.reduction, s.u.c.x)
    d1d3 = dz_squared_mean(s.reduction, s.u.c.x)

    d2d1 = dx_squared_mean(s.reduction, s.u.c.y)
    d2d2 = dy_squared_mean(s.reduction, s.u.c.y)
    d2d3 = dz_squared_mean(s.reduction, s.u.c.y)

    d3d1 = dx_squared_mean(s.reduction, s.u.c.z)
    d3d2 = dy_squared_mean(s.reduction, s.u.c.z)
    d3d3 = dz_squared_mean(s.reduction, s.u.c.z)

    ε = ν*((d1d1+d2d2+d3d3) + (d1d2+d2d1) + (d1d3+d3d1) + (d2d3+d3d2))

    return u1, u2, u3, u12, u22, u32, k, d1d1, d1d2, d1d3, d2d1, d2d2, d2d3, d3d1, d3d2, d3d3, ε
end

@par function scalar_stats(ρ,s1,s::@par(AbstractSimulation))
    rho = real(ρ[1,1,1])
    rho2 = squared_mean(s1.reduction,ρ)

    drd1 = dx_squared_mean(s1.reduction, ρ)
    drd2 = dy_squared_mean(s1.reduction, ρ)
    drd3 = dz_squared_mean(s1.reduction, ρ)

    rhodiss = diffusivity(s1)*(drd1^2 + drd2^2 + drd3^2)

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

function hyperviscosity_stats(reduction,u::VectorField{T},s) where {T}
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
                    magsq = mag2(u[i,j,l])
                    ee += (1 + (i>1)) * (k2^M)*magsq 
                end
            end
            result[ii] += ee
        end
    end
    mν = nuh(s)
    return mν*sum(result)
end

function les_stats(reduction,τ::AbstractField{T},u::VectorField{T}) where {T}
    isrealspace(u) && fourier!(u)
    isrealspace(τ) && fourier!(τ)
    result = fill!(reduction,zero(T))
    @mthreads for l in ZRANGE
        @inbounds begin
            ee = zero(T)
            ii = Threads.threadid()
            @inbounds for j in YRANGE
                @simd for i in XRANGE
                    S = symouter(im*K[i,j,l],u[i,j,l])
                    magsq = proj(τ[i,j,l],S)
                    ee += (1 + (i>1))*magsq 
                end
            end
            result[ii] += ee
        end
    end
    return sum(result)
end