_append(x::Tuple, y) = (x..., y) 
_append(x::Tuple, y::Tuple) = (x..., y...)
flatten(x::Tuple) = _flatten((), x)
_flatten(result::Tuple, x::Tuple) = _flatten(_append(result, first(x)), Base.tail(x)) 
_flatten(result::Tuple, x::Tuple{}) = result

function rfftfreq(n::Integer,s::Real)::Vector{Float64}
    Float64[(n/2 - i)/s for i = n/2:-1:0]
end

@inline fsqrt(x) = @fastmath(sqrt(x))

function fftfreq(n::Integer,s::Real)::Vector{Float64}
    if iseven(n)
        return vcat(Float64[(n/2 - i)/s for i = n/2:-1:1],Float64[-i/s for i = n/2:-1:1])
    else return vcat(Float64[(n/2 - i)/s for i = n/2:-1:0],Float64[-i/s for i = (n-1)/2:-1:1])
    end
end

function mycopy!(out::Array{<:Real,N},inp::Array{<:Real,N}) where N
    @mthreads for j in TRANGE
        l = REAL_RANGES[j]
        for i in l
            @inbounds out[i] = inp[i]
        end
    end
end

mycopy!(o::ScalarField,i::ScalarField) = mycopy!(o.field.data,i.field.data)

mycopy!(o::VectorField,i::VectorField) = (mycopy!(o.rr.x,i.rr.x);
                                          mycopy!(o.rr.y,i.rr.y);
                                          mycopy!(o.rr.z,i.rr.z))

mycopy!(o::SymTrTenField,i::SymTrTenField) = (mycopy!(o.rr.xx,i.rr.xx);
                                              mycopy!(o.rr.xy,i.rr.xy);
                                              mycopy!(o.rr.xz,i.rr.xz);
                                              mycopy!(o.rr.yy,i.rr.yy);
                                              mycopy!(o.rr.yz,i.rr.yz))

@par function myscale!(field::AbstractArray{<:Real,N}) where N
    @mthreads for ind in ZYRANGE
        jj,kk = Tuple(ind)
        k = ZRANGE[kk]
        j = YRANGE[jj]
        x = 1/(NRX*NY*NZ)
            @msimd for i in XRANGE2X
                @inbounds field[i,j,k] = x*field[i,j,k]
            end
    end
end 

@par function fullmyscale!(field::AbstractArray{<:Real,N}) where N
    @mthreads for l in TRANGE
        x = 1/(NRX*NY*NZ)
        @msimd for i in REAL_RANGES[l]
            @inbounds field[i] = x*field[i]
        end
    end
end 


myscale!(o::ScalarField) = myscale!(o.field.data)

myscale!(o::VectorField) = (myscale!(o.rr.x);
                            myscale!(o.rr.y);
                            myscale!(o.rr.z))

myscale!(o::SymTrTenField) = (myscale!(o.rr.xx);
                              myscale!(o.rr.xy);
                              myscale!(o.rr.xz);
                              myscale!(o.rr.yy);
                              myscale!(o.rr.yz))

myscale!(o::SymTenField) = (myscale!(o.rr.xx);
                            myscale!(o.rr.xy);
                            myscale!(o.rr.xz);
                            myscale!(o.rr.yy);
                            myscale!(o.rr.yz);
                            myscale!(o.rr.zz))

fullmyscale!(o::ScalarField) = fullmyscale!(o.field.data)

fullmyscale!(o::VectorField) = (fullmyscale!(o.rr.x);
                            fullmyscale!(o.rr.y);
                            fullmyscale!(o.rr.z))
                            
fullmyscale!(o::SymTrTenField) = (fullmyscale!(o.rr.xx);
                              fullmyscale!(o.rr.xy);
                              fullmyscale!(o.rr.xz);
                              fullmyscale!(o.rr.yy);
                              fullmyscale!(o.rr.yz))
                            
fullmyscale!(o::SymTenField) = (fullmyscale!(o.rr.xx);
                            fullmyscale!(o.rr.xy);
                            fullmyscale!(o.rr.xz);
                            fullmyscale!(o.rr.yy);
                            fullmyscale!(o.rr.yz);
                            fullmyscale!(o.rr.zz))

function myfourier!(field::A) where {T,N,N2,L,A<:Union{<:ScalarField{T,N,N2,L},<:VectorField{T,N,N2,L},<:SymTrTenField{T,N,N2,L},<:SymTenField{T,N,N2,L}}}
    rfft!(field)
    dealias!(field)
    myscale!(field)
    return nothing
end

function fullfourier!(field::A) where {T,N,N2,L,A<:Union{<:ScalarField{T,N,N2,L},<:VectorField{T,N,N2,L},<:SymTrTenField{T,N,N2,L},<:SymTenField{T,N,N2,L}}}
    rfft!(field)
    fullmyscale!(field)
    return nothing
end

function dealias!(f::AbstractArray{T,3}) where {T<:Complex}
    @mthreads for i in RANGEC
        @inbounds begin
            DEALIAS[i] && (f[i] = zero(T))
        end
   end
end

dealias!(o::VectorField) = (dealias!(o.c.x);
                            dealias!(o.c.y);
                            dealias!(o.c.z))

dealias!(o::SymTrTenField) = (dealias!(o.c.xx);
                              dealias!(o.c.xy);
                              dealias!(o.c.xz);
                              dealias!(o.c.yy);
                              dealias!(o.c.yz))

dealias!(o::SymTenField) = (dealias!(o.c.xx);
                              dealias!(o.c.xy);
                              dealias!(o.c.xz);
                              dealias!(o.c.yy);
                              dealias!(o.c.yz);
                              dealias!(o.c.zz))



function splitrange(lr,nt)
    a = UnitRange{Int}[]
    sizehint!(a,nt)
    n = lr÷nt
    r = lr%nt
    stop = 0
    init = 1
    for i=1:r
        stop=init+n
        push!(a,init:stop)
        init = stop+1
    end
    for i=1:(nt-r)
        stop=init+n-1
        push!(a,init:stop)
        init = stop+1
    end
    return (a...,)
end

function compute_shells2D(kx,ky,Nx,Ny)
    nShells2D = min(Nx,Ny÷2)
    maxdk2D = max(kx[2],ky[2])
    kh = zeros(nShells2D)
    numPtsInShell2D = zeros(Int,nShells2D)

    @inbounds for j=1:Ny
        for i=1:Nx
            K = fsqrt(kx[i]^2 + ky[j]^2)
            ii = round(Int,K/maxdk2D)+1
            ii > nShells2D && break
            kh[ii] += K
            numPtsInShell2D[ii] += 1
        end
    end
  
    @inbounds @simd for i in 1:length(kh)
        kh[i] = kh[i]/numPtsInShell2D[i]
    end

    return nShells2D, maxdk2D, numPtsInShell2D, kh
end

function calculate_Zf(d,kf,kh)
    Zf = zeros(size(kh))
    zft = haskey(d,:Zf) ? Symbol(d[:Zf]) : :cutoff
    if zft == :cutoff
        for (i,k) in enumerate(kh)
            Zf[i] = k <= kf
        end
    else
        for (i,k) in enumerate(kh)
            Zf[i] = tanh((kf-k) / (0.25*kf)) * (((kf-k) <= 0.0) ? 0.0 : 1.0)
        end
    end
    return Zf
end

function calculate_Zf(d,kf,kx,ky)
    Zf = zeros(length(kx),length(ky))
    zft = haskey(d,:Zf) ? Symbol(d[:Zf]) : :cutoff
    if zft == :tanh
        for j in eachindex(ky)
            for i in eachindex(kx)
                k = sqrt(kx[i]^2 + ky[j]^2)
                Zf[i,j] = tanh((kf-k) / (0.25*kf)) * (((kf-k) <= 0.0) ? 0.0 : 1.0)
            end
        end
    else
        for j in eachindex(ky)
            for i in eachindex(kx)
                k = sqrt(kx[i]^2 + ky[j]^2)
                Zf[i,j] = k <= kf
            end
        end
    end
    return Zf
end

@inline function Gaussfilter(Δ²::Real, k2::Real)
    aux = -Δ²/24
    return exp(aux*k2)
end

@inline Gaussfilter(Δ²::Real, i::Integer,j::Integer,k::Integer) = Gaussfilter(Δ²,K[i,j,k]⋅K[i,j,k])

@inline function boxfilter(Δ²::Real, k2::Real)
    aux = 0.5*fsqrt(Δ²*k2)
    return k2 == zero(k2) ? oneunit(k2) : sin(aux)/aux
end

@inline boxfilter(Δ::Real,i::Integer,j::Integer,k::Integer) = boxfilter(Δ,K[i,j,k]⋅K[i,j,k])

@inline function cutofffilter(Δ²::Real, k2::Real)
    aux = inv(Δ²)*π^2
    return k2 < aux
end

@inline cutofffilter(Δ²::Real,i::Integer,j::Integer,k::Integer) = boxfilter(Δ²,K[i,j,k]⋅K[i,j,k])

mymax(x::Float64,y::Float64) = max(abs(x),abs(y))
mymax(x::Float64,v::Vec{Float64}) = mymax(x,max(abs(v.x),abs(v.y),abs(v.z)))
mymax(v::Vec{Float64},x::Float64) = mymax(x,v)
mymax(v::Vec{Float64}) = mymax(v.x,mymax(v.y,v.z))
mymax(x::Float64) = abs(x)

function find_max(reduction,v)
    @mthreads for k in RZRANGE
        ti = Threads.threadid()
        umax = mymax(v[1,1,k])
        @inbounds for j in RYRANGE
            for i in RXRANGE
                umax = mymax(umax,mymax(v[i,j,k]))
            end
        end
        reduction[ti] = max(umax,reduction[ti])
    end
end

mysetfourier!(f) = (dealias!(f); setfourier!(f))