function writeheader(s::AbstractParameters)
  open("Stats.txt","w") do f
    write(f,"iteration  time  u1  u2  u3  u1^2  u2^2  u3^2  du1dx1^2  du1dx2^2  du1dx3^2 du2dx1^2  du2dx2^2  du2dx3^2 du3dx1^2  du3dx2^2  du3dx3^2 \n")
  end
end

function stats(s::AbstractParameters,init::Integer,time::Real)
  results = velocity_stats(s)
  open("Stats.txt","a+") do file 
    join(file,(init, time, results..., "\n"), "  ")
  end
end

@par function velocity_stats(s::@par(AbstractParameters))
  u1 = real(s.u.cx[1,1,1])/(Nrx*Ny*Nz)
  u2 = real(s.u.cy[1,1,1])/(Nrx*Ny*Nz)
  u3 = real(s.u.cz[1,1,1])/(Nrx*Ny*Nz)

  mycopy!(s.aux,s.u,s)
  s.p\s.aux
  u12 = tmean(x->x^2,s.aux.rx,s)
  u22 = tmean(x->x^2,s.aux.ry,s)
  u32 = tmean(x->x^2,s.aux.rz,s)

  grad!(s.aux,s.u.cx,s)
  s.p\s.aux
  d1d1 = tmean(x->x^2,s.aux.rx,s)
  d1d2 = tmean(x->x^2,s.aux.ry,s)
  d1d3 = tmean(x->x^2,s.aux.rz,s)

  grad!(s.aux,s.u.cy,s)
  s.p\s.aux
  d2d1 = tmean(x->x^2,s.aux.rx,s)
  d2d2 = tmean(x->x^2,s.aux.ry,s)
  d2d3 = tmean(x->x^2,s.aux.rz,s)

  grad!(s.aux,s.u.cz,s)
  s.p\s.aux
  d3d1 = tmean(x->x^2,s.aux.rx,s)
  d3d2 = tmean(x->x^2,s.aux.ry,s)
  d3d3 = tmean(x->x^2,s.aux.rz,s)

  return u1, u2, u3, u12, u22, u32, d1d1, d1d2, d1d3, d2d1, d2d2, d2d3, d3d1, d3d2, d3d3
end

function writeheader(s::ScalarParameters)
  open("Stats.txt","w") do f
    write(f,"iteration  time  u1  u2  u3  u1^2  u2^2  u3^2  du1dx1^2  du1dx2^2  du1dx3^2 du2dx1^2  du2dx2^2  du2dx3^2 du3dx1^2  du3dx2^2  du3dx3^2 rho rho^2 drhodx^2 drhody^3 drhodz^2\n")
  end
end


function stats(s::ScalarParameters,init::Integer,time::Real)
  results = velocity_stats(s)
  results2 = scalar_stats(s)  
  open("Stats.txt","a+") do file 
    join(file,(init, time, results..., results2..., "\n"), "  ")
  end
end

@par function scalar_stats(s::@par(ScalarParameters))
  rho = real(s.ρ[1,1,1])/(Nrx*Ny*Nz)  
  _mycopy!(complex(s.ρrhs),complex(s.ρ),s)
  s.ps\s.ρrhs
  rho2 = tmean(x->x^2,rawreal(s.ρrhs),s)
  dealias!(s.ρrhs,s)

  grad!(s.aux,complex(s.ρ),s)
  s.p\s.aux
  drd1 = tmean(x->x^2,s.aux.rx,s)
  drd2 = tmean(x->x^2,s.aux.ry,s)
  drd3 = tmean(x->x^2,s.aux.rz,s)

  return rho, rho2, drd1, drd2, drd3
end

ape(s::ScalarParameters) = tmean(x->x^2,rawreal(s.ρ),s)

@par function tmean(f::Function,x::AbstractArray{T,3},s::@par(AbstractParameters)) where {T<:Number}

  result = fill!(s.reduction,0.0)
  Threads.@threads for k in 1:Nz
    for j in 1:Ny
      @fastmath @inbounds @msimd for i in 1:Nrx
        result[Threads.threadid()] += f(x[i,j,k])::T
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
       @fastmath @inbounds @msimd for i in 1:Nrx
         result[Threads.threadid()] += getind(f,x,i,j,k)::T
       end
     end
   end
  return sum(result)/(Nrx*Ny*Nz)
end
