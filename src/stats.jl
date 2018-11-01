function statsheader(s::AbstractSimulation)
    simulaitionheader = "iteration,time,u1,u2,u3,u1^2,u2^2,u3^2,du1dx1^2,du1dx2^2,du1dx3^2,du2dx1^2,du2dx2^2,du2dx3^2,du3dx1^2,du3dx2^2,du3dx3^2"
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

function writestats(s::AbstractSimulation,init::Integer,time::Real)
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

    mycopy!(s.aux,s.u)
    brfft!(s.aux)
    u12 = tmean(x->x^2,s.aux.rr.x,s)
    u22 = tmean(x->x^2,s.aux.rr.y,s)
    u32 = tmean(x->x^2,s.aux.rr.z,s)

    grad!(s.aux,s.u.c.x,s)
    brfft!(s.aux)
    d1d1 = tmean(x->x^2,s.aux.rr.x,s)
    d1d2 = tmean(x->x^2,s.aux.rr.y,s)
    d1d3 = tmean(x->x^2,s.aux.rr.z,s)

    grad!(s.aux,s.u.c.y,s)
    brfft!(s.aux)
    d2d1 = tmean(x->x^2,s.aux.rr.x,s)
    d2d2 = tmean(x->x^2,s.aux.rr.y,s)
    d2d3 = tmean(x->x^2,s.aux.rr.z,s)

    grad!(s.aux,s.u.c.z,s)
    brfft!(s.aux)
    d3d1 = tmean(x->x^2,s.aux.rr.x,s)
    d3d2 = tmean(x->x^2,s.aux.rr.y,s)
    d3d3 = tmean(x->x^2,s.aux.rr.z,s)

    return u1, u2, u3, u12, u22, u32, d1d1, d1d2, d1d3, d2d1, d2d2, d2d3, d3d1, d3d2, d3d3
end

@par function scalar_stats(s1,s::@par(AbstractSimulation))
    rho = real(s1.ρ[1,1,1])
    mycopy!(s1.ρrhs,s1.ρ)
    brfft!(s1.ρrhs)
    rho2 = tmean(x->x^2,parent(real(s1.ρrhs)),s)

    grad!(s.aux,complex(s1.ρ),s)
    brfft!(s.aux)
    drd1 = tmean(x->x^2,s.aux.rr.x,s)
    drd2 = tmean(x->x^2,s.aux.rr.y,s)
    drd3 = tmean(x->x^2,s.aux.rr.z,s)

    return rho, rho2, drd1, drd2, drd3
end

#ape(s::AbstractSimulation) = tmean(x->x^2,parent(real(s.ρ)),s)

@par function tmean(f::F,x::AbstractArray{T,3},s::@par(AbstractSimulation)) where {F<:Function,T<:Number}

    result = fill!(s.reduction,0.0)
    @mthreads for k in ZRANGE
        for j in YRANGE
            @inbounds @msimd for i in RXRANGE
                result[Threads.threadid()] += f(x[i,j,k])::T
            end
        end
    end

    return sum(result)/(NRX*NY*NZ)
end

tmean(x::AbstractArray,s::AbstractSimulation) = tmean(identity,x,s)

@generated function getind(f::F,x::NTuple{N,AbstractArray{T,3}},i::Int,j::Int,k::Int) where {F,T,N}
    args = Array{Any,1}(N)
    for l=1:N 
       args[l] = :(x[$l][i,j,k])
    end
    return Base.pushmeta!(Expr(:call,:f,args...),:inline)
end

@par function tmean(f::F,x::NTuple{N,AbstractArray{T,3}},s::@par(AbstractSimulation)) where {F,T,N}
    result = fill!(s.reduction,0.0)
    @mthreads for k in ZRANGE
        for j in YRANGE
            @inbounds @msimd for i in RXRANGE
                result[Threads.threadid()] += getind(f,x,i,j,k)::T
            end
        end
    end
    return sum(result)/(NRX*NY*NZ)
end
