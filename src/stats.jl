function writeheader(s::AbstractParameters)
  open("Stats.txt","w") do f
    write(f,"iteration  time  u12  u22  u32  k  enstrophy \n")
  end
end

function stats(s::AbstractParameters,init::Integer,dt::Real)
  u2,v2,w2 = kinetic_energy(s)
  k = (u2+v2+w2)/2
  ω = enstrophy(s)
  open("Stats.txt","a+") do file 
    join(file,(init, init*dt, u2, v2, w2, k, ω, "\n"), "  ")
  end
end

function kinetic_energy(s::AbstractParameters)
  u2 = tmean(x->x^2,s.u.rx,s)
  v2 = tmean(x->x^2,s.u.ry,s)
  w2 = tmean(x->x^2,s.u.rz,s)
  return u2,v2,w2
end

function enstrophy(s::AbstractParameters)
  a = s.aux
  u = s.u
  rhs = s.rhs
  copy!(real(a),real(u))
  s.p*a
  curl!(rhs,a,s)
  s.p\rhs
  ω = tmean((x,y,z)->(x^2+y^2+z^2),(rhs.rx,rhs.ry,rhs.rz),s)
  return ω
end

function writeheader(s::ScalarParameters)
  open("Stats.txt","w") do f
    write(f,"iteration  time  u12  u22  u32  k  enstrophy rho2 \n")
  end
end


function stats(s::ScalarParameters,init::Integer,dt::Real)
  u2,v2,w2 = kinetic_energy(s)
  k = (u2+v2+w2)/2
  ω = enstrophy(s)
  ρ2 = ape(s)
  open("Stats.txt","a+") do file 
    join(file,(init, init*dt, u2, v2, w2, k, ω, ρ2, "\n"), "  ")
  end
end

ape(s::ScalarParameters) = tmean(x->x^2,rawreal(s.ρ),s)

@par function tmean(f::Function,x::AbstractArray{T,3},s::@par(AbstractParameters)) where {T<:Number}

  result = fill!(s.reduction,0.0)
  Threads.@threads for k in 1:Nz
    for j in 1:Ny
      @simd for i in 1:Nrx
        @inbounds result[Threads.threadid()] += f(x[i,j,k])::T
      end
    end
  end

  return sum(result)/(Nrx*Ny*Nz)
end

tmean(x::AbstractArray,s::AbstractParameters) = tmean(identity,x,s)

@inline @generated function getind(f::Function,x::NTuple{N,AbstractArray{T,3}},i::Int,j::Int,k::Int) where {T,N}
  args = Array{Any,1}(N)
  for l=1:N 
   args[l] = :(x[$l][i,j,k])
  end
  return Expr(:call,:f,args...)
end

@par function tmean(f::Function,x::NTuple{N,AbstractArray{T,3}},s::@par(AbstractParameters)) where {T,N}
   result = fill!(s.reduction,0.0)
   Threads.@threads for k in 1:Nz
     for j in 1:Ny
       @simd for i in 1:Nrx
         @inbounds result[Threads.threadid()] += getind(f,x,i,j,k)::T
       end
     end
   end
  return sum(result)/(Nrx*Ny*Nz)
end
