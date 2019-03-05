function statsheader(s::AbstractSimulation)
    simulaitionheader = "iteration,time,u1,u2,u3,u1p2,u2p2,u3p2,k,du1dx1p2,du1dx2p2,du1dx3p2,du2dx1p2,du2dx2p2,du2dx3p2,du3dx1p2,du3dx2p2,du3dx3p2,diss"
    header = join(Iterators.filter(x->x !== "",
      (simulaitionheader,statsheader.(getfield.(Ref(s),sim_fields))...,"\n")),
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

    setfourier!(s.rhs)
    grad!(s.rhs,s.u.c.x,s)
    d1d1 = squared_mean(s.reduction, s.rhs.c.x)
    d1d2 = squared_mean(s.reduction, s.rhs.c.y)
    d1d3 = squared_mean(s.reduction, s.rhs.c.z)

    grad!(s.rhs,s.u.c.y,s)
    d2d1 = squared_mean(s.reduction, s.rhs.c.x)
    d2d2 = squared_mean(s.reduction, s.rhs.c.y)
    d2d3 = squared_mean(s.reduction, s.rhs.c.z)

    grad!(s.rhs,s.u.c.z,s)
    d3d1 = squared_mean(s.reduction, s.rhs.c.x)
    d3d2 = squared_mean(s.reduction, s.rhs.c.y)
    d3d3 = squared_mean(s.reduction, s.rhs.c.z)

    ε = 2ν*(d1d1+d2d2+d3d3+ 2*((d1d2+d2d1)/2 + (d1d3+d3d1)/2 + (d2d3+d3d2)/2))

    return u1, u2, u3, u12, u22, u32, k, d1d1, d1d2, d1d3, d2d1, d2d2, d2d3, d3d1, d3d2, d3d3, ε
end

@par function scalar_stats(ρ,s1,s::@par(AbstractSimulation))
    rho = real(ρ[1,1,1])
    rho2 = squared_mean(s1.reduction,ρ)

    setfourier!(s.rhs)
    grad!(s.rhs,ρ,s)
    drd1 = squared_mean(s1.reduction, s.rhs.c.x)
    drd2 = squared_mean(s1.reduction, s.rhs.c.y)
    drd3 = squared_mean(s1.reduction, s.rhs.c.z)

    return rho, rho2, drd1, drd2, drd3
end

#ape(s::AbstractSimulation) = tmean(x->x^2,parent(real(s.ρ)),s)

@par function tmean(f::F,x::AbstractArray{T,3},s::@par(AbstractSimulation)) where {F<:Function,T<:Number}

    result = fill!(s.reduction,0.0)
    @mthreads for k in ZRANGE
        for j in YRANGE
            @inbounds @msimd for i in RXRANGE
                result[Threads.threadid()] += f(x[i,j,k])
            end
        end
    end

    return sum(result)/(NRX*NY*NZ)
end

tmean(x::AbstractArray,s::AbstractSimulation) = tmean(identity,x,s)

@generated function getind(f::F,x::NT,i::Int,j::Int,k::Int) where {F<:Function,N,NT<:NTuple{N,AbstractArray}}
    args = Array{Any,1}(undef,N)
    for l=1:N 
       args[l] = :(x[$l][i,j,k])
    end
    ex = Expr(:call,:f,args...) 
    return Expr(:block,Expr(:meta,:inline),Expr(:meta,:propagate_inbounds),ex)
end

@par function tmean(f::F,x::NT,reduction::L) where {F<:Function,N,NT<:NTuple{N,AbstractArray},L}
    result = fill!(reduction,0.0)
    @mthreads for k in ZRANGE
        for j in YRANGE
            @inbounds @msimd for i in RXRANGE
                result[Threads.threadid()] += getind(f,x,i,j,k)
            end
        end
    end
    return sum(result)/(NRX*NY*NZ)
end

function squared_mean(reduction,u::ScalarField{T}) where {T}
    isrealspace(u) && fourier!(u)
    result = fill!(reduction,zero(T))
    @mthreads for l in ZRANGE
        ee = zero(T)
        ii = Threads.threadid()
        @inbounds for j in YRANGE
            magsq = abs2(u[1,j,l])
            ee += magsq
            @simd for i in 2:NX
                magsq = abs2(u[i,j,l])
                ee += 2*magsq 
            end
        end
        result[ii] += ee
    end
    return sum(result)
end

function proj_mean(reduction,u::ScalarField{T},v) where {T}
    isrealspace(u) && fourier!(u)
    #isrealspace(v) && fourier!(v)
    result = fill!(reduction,zero(T))
    @mthreads for l in ZRANGE
        ee = zero(T)
        ii = Threads.threadid()
        @inbounds for j in YRANGE
            magsq = proj(u[1,j,l],v[1,j,l])
            ee += magsq
            @simd for i in 2:NX
                magsq = proj(u[i,j,l],v[i,j,l])
                ee += 2*magsq 
            end
        end
        result[ii] += ee
    end
    return sum(result)
end