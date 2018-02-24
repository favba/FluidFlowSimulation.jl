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
      for i in 1:(Kxr[k][j])
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
      for i in 1:(2Kxr[k][j])
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
      for i in 1:(Kxr[k][j])
        @inbounds out[i,j,k] = inp[i,j,k]
      end
    end
  end
end

@inline @par function my_scale_real!(field::AbstractArray{<:Real,3},s::@par(AbstractSimulation))
  x = 1/(Nrx*Ny*Nz)
  @mthreads for k in 1:Nz
    for j in 1:Ny
      @msimd for i in 1:Nrx
        @inbounds field[i,j,k] = x*field[i,j,k]
      end
    end
  end
end 

@inline my_scale_real!(field::PaddedArray,s) = my_scale_real!(parent(real(field)),s)

@inline function my_scale_real!(field::VectorField,s)
  my_scale_real!(field.rx,s)
  my_scale_real!(field.ry,s)
  my_scale_real!(field.rz,s)
end


@inline @par function my_scale_fourier!(field::AbstractArray{<:Real,3},s::@par(AbstractSimulation))
  x = 1/(Nrx*Ny*Nz)
  @mthreads for k in 1:Nz
    for j in 1:Ny
      @inbounds @fastmath @msimd for i in 1:2Nx
        field[i,j,k] = x*field[i,j,k]
      end
    end
  end
end 


@inline my_scale_fourier!(field::PaddedArray,s) = my_scale_fourier!(parent(real(field)),s)

@inline function my_scale_fourier!(field::VectorField,s)
  my_scale_fourier!(field.rx,s)
  my_scale_fourier!(field.ry,s)
  my_scale_fourier!(field.rz,s)
end

@inline function my_scale_fourier!(field::SymmetricTracelessTensor,s)
  my_scale_fourier!(field.rxx,s)
  my_scale_fourier!(field.rxy,s)
  my_scale_fourier!(field.rxz,s)
  my_scale_fourier!(field.ryy,s)
  my_scale_fourier!(field.ryz,s)
end

@inline function irfft!(field,p,s::AbstractSimulation)
  p*field
  #Not doing scaling in real space anymore.
  #my_scale_real!(field,s)
  return nothing
end

@inline function rfft!(field,p,s::AbstractSimulation)
  p*field
  my_scale_fourier!(field,s)
  return nothing
end


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
      K = sqrt(kx[i]^2 + ky[j]^2)
      ii = round(Int,K/maxdk2D)+1
      if ii <= nShells2D
        kh[ii] += K
        numPtsInShell2D[ii] += 1
      end
    end
  end
  
  @inbounds @simd for i in linearindices(kh)
    kh[i] = kh[i]/numPtsInShell2D[i]
  end

  return nShells2D, maxdk2D, numPtsInShell2D, kh
end