_append(x::Tuple, y) = (x..., y) 
_append(x::Tuple, y::Tuple) = (x..., y...)
flatten(x::Tuple) = _flatten((), x)
_flatten(result::Tuple, x::Tuple) = _flatten(_append(result, first(x)), Base.tail(x)) 
_flatten(result::Tuple, x::Tuple{}) = result

function rfftfreq(n::Integer,s::Real)::Vector{Float64}
  Float64[(n/2 - i)/s for i = n/2:-1:0]
end

function fftfreq(n::Integer,s::Real)::Vector{Float64}
  if iseven(n)
    return vcat(Float64[(n/2 - i)/s for i = n/2:-1:1],Float64[-i/s for i = n/2:-1:1])
  else return vcat(Float64[(n/2 - i)/s for i = n/2:-1:0],Float64[-i/s for i = (n-1)/2:-1:1])
  end
end

@inbounds @par function mycopy!(rm::AbstractArray{T,3},rhs::AbstractArray{T,3},s::@par(AbstractSimulation)) where T<:Complex
  @mthreads for kk in 1:length(Kzr)
    k = Kzr[kk]
    jj::Int = 1
    for y in Kyr, j in y
      for i in Kxr
        rm[i,jj,kk] = rhs[i,j,k]
      end
    jj+=1
    end
  end
end

@inbounds @par function mycopy!(rm::AbstractArray{T,3},rhs::AbstractArray{T,3},s::@par(AbstractSimulation)) where T<:Real
  @mthreads for kk in 1:length(Kzr)
    k = Kzr[kk]
    jj::Int = 1
    for y in Kyr, j in y
      for i in 1:(2length(Kxr))
        rm[i,jj,kk] = rhs[i,j,k]
      end
    jj+=1
    end
  end
end

@par function mycopy!(out::VectorField,inp::VectorField,s::@par(AbstractSimulation))
  _mycopy!(out.cx,inp.cx,s)
  _mycopy!(out.cy,inp.cy,s)
  _mycopy!(out.cz,inp.cz,s)
end

@par function _mycopy!(out::Array{Complex128,3},inp::Array{Complex128,3},s::@par(AbstractSimulation))
  @mthreads for k in Kzr
    for y in Kyr, j in y
      for i in Kxr
        @inbounds out[i,j,k] = inp[i,j,k]
      end
    end
  end
end

@inline @par function my_scale!(field::AbstractArray{<:Real,3},s::@par(AbstractSimulation))
  x = 1/(Nrx*Ny*Nz)
  @mthreads for k in 1:Nz
    for j in 1:Ny
      @msimd for i in 1:Nrx
        @inbounds field[i,j,k] = x*field[i,j,k]
      end
    end
  end
end 

@inline my_scale!(field::PaddedArray,s) = my_scale!(parent(real(field)),s)

@inline function my_scale!(field::VectorField,s)
  my_scale!(field.rx,s)
  my_scale!(field.ry,s)
  my_scale!(field.rz,s)
end

@inline function back_transform!(field,p,s)
  A_mul_B!(real(field),p,complex(field))
  my_scale!(field,s)
  return nothing
end