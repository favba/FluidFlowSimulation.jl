@par function cross!(outx::AbstractArray{T},outy::AbstractArray{T},outz::AbstractArray{T},
                  ux::AbstractArray{T},uy::AbstractArray{T},uz::AbstractArray{T},
                  vx::AbstractArray{T},vy::AbstractArray{T},vz::AbstractArray{T},
                  s::@par(AbstractParameters)) where {T<:Real}
  @mthreads for i in 1:Lrs
    @fastmath @inbounds begin
     outx[i] = uy[i]*vz[i] - uz[i]*vy[i]
     outy[i] = uz[i]*vx[i] - ux[i]*vz[i]
     outz[i] = ux[i]*vy[i] - uy[i]*vx[i]
    end
  end
end

@par function cross!(outx::AbstractArray{T},outy::AbstractArray{T},outz::AbstractArray{T},
  ux::AbstractArray{T},uy::AbstractArray{T},uz::AbstractArray{T},
  vx::AbstractArray{T},vy::AbstractArray{T},vz::AbstractArray{T},
  s::@par(AbstractParameters)) where {T<:Complex}
  @mthreads for k in Kzr
    for j in Kyr
      @fastmath @inbounds @msimd for i in Kxr
        outx[i] = uy[i]*vz[i] - uz[i]*vy[i]
        outy[i] = uz[i]*vx[i] - ux[i]*vz[i]
        outz[i] = ux[i]*vy[i] - uy[i]*vx[i]
      end
    end
  end
end

@par function crossk!(outx,outy,outz,vx,vy,vz,s::@par(AbstractParameters)) 
  @mthreads for k in Kzr
    for j in Kyr
      @fastmath @inbounds @msimd for i in Kxr 
        outx[i,j,k] = im*(ky[j]*vz[i,j,k] - kz[k]*vy[i,j,k])
        outy[i,j,k] = im*(kz[k]*vx[i,j,k] - kx[i]*vz[i,j,k])
        outz[i,j,k] = im*(kx[i]*vy[i,j,k] - ky[j]*vx[i,j,k])
      end
    end
  end  
end

function rcross!(out::VectorField,u::VectorField,v::VectorField,s::AbstractParameters)
  cross!(out.rx,out.ry,out.rz,u.rx,u.ry,u.rz,v.rx,v.ry,v.rz,s)
  return out  
end

function ccross!(out::VectorField,u::VectorField,v::VectorField,s::AbstractParameters)
  cross!(out.cx,out.cy,out.cz,u.cx,u.cy,u.cz,v.cx,v.cy,v.cz,s)
  return out
end

function curl!(out::VectorField,u::VectorField,s::AbstractParameters)
  crossk!(out.cx,out.cy,out.cz,u.cx,u.cy,u.cz,s)
  return out
end

function grad!(out::VectorField,f::AbstractArray{<:Complex,3},s::AbstractParameters)
  dx!(out.cx,f,s)
  dy!(out.cy,f,s)
  dz!(out.cz,f,s)
  dealias!(out,s)
end

@par function dx!(out::AbstractArray{<:Complex,3},f::AbstractArray{<:Complex,3},s::@par(AbstractParameters)) 
  @mthreads for k in Kzr
    for j in Kyr
      @fastmath @inbounds @msimd for i in Kxr
        out[i,j,k] = f[i,j,k]*im*kx[i]
      end
    end
  end
end

@par function dy!(out::AbstractArray{<:Complex,3},f::AbstractArray{<:Complex,3},s::@par(AbstractParameters)) 
  @mthreads for k in Kzr
    for j in Kyr
      @fastmath @inbounds @msimd for i in Kxr
        out[i,j,k] = f[i,j,k]*im*ky[j]
      end
    end
  end
end

@par function dz!(out::AbstractArray{<:Complex,3},f::AbstractArray{<:Complex,3},s::@par(AbstractParameters)) 
  @mthreads for k in Kzr
    for j in Kyr
      @fastmath @inbounds @msimd for i in Kxr
        out[i,j,k] = f[i,j,k]*im*kz[k]
      end
    end
  end
end

function rfftfreq(n::Integer,s::Real)::Vector{Float64}
  d = 2π*s
  Float64[(n/2 - i)/d for i = n/2:-1:0]
end

function fftfreq(n::Integer,s::Real)::Vector{Float64}
  d = 2π*s
  if iseven(n)
    return vcat(Float64[(n/2 - i)/d for i = n/2:-1:1],Float64[-i/d for i = n/2:-1:1])
  else return vcat(Float64[(n/2 - i)/d for i = n/2:-1:0],Float64[-i/d for i = (n-1)/2:-1:1])
  end
end

function scalar_advection!(out::VectorField,scalar::AbstractArray,v::VectorField,s::ScalarParameters)

  _scalar_advection!(out.rx,rawreal(scalar),v.rx,s)
  _scalar_advection!(out.ry,rawreal(scalar),v.ry,s)
  _scalar_advection!(out.rz,rawreal(scalar),v.rz,s)

end

@par function _scalar_advection!(out::AbstractArray{Float64,3},scalar::AbstractArray{Float64,3},v::AbstractArray{Float64,3},s::@par(ScalarParameters))
  @mthreads for i in 1:Lrs
      @fastmath @inbounds out[i] = scalar[i]*v[i]
  end
end

@par function div!(out::AbstractArray{Complex128,3},ux,uy,uz,w,mdρdz::Real, s::@par(ScalarParameters))
    mim::Complex128 = -im
    @mthreads for k in Kzr
      for j in Kyr
        @fastmath @inbounds @msimd for i in Kxr
         # out[i,j,k] = mim*(kx[i]*ux[i,j,k] + ky[j]*uy[i,j,k] + kz[k]*uz[i,j,k]) + mdœÅdz*w[i,j,k]
          out[i,j,k] = muladd(mim,muladd(kx[i], ux[i,j,k], muladd(ky[j], uy[i,j,k], kz[k]*uz[i,j,k])), mdρdz*w[i,j,k])
        end
      end
    end
end

@par function div!(out::AbstractArray{Complex128,3},ux,uy,uz, s::@par(AbstractParameters))
  mim::Complex128 = -im
    @mthreads for k in Kzr
      for j in Kyr
        @fastmath @inbounds @msimd for i in Kxr
          #out[i,j,k] = -im*(kx[i]*ux[i,j,k] + ky[j]*uy[i,j,k] + kz[k]*uz[i,j,k])
          out[i,j,k] = mim*muladd(kx[i], ux[i,j,k], muladd(ky[j], uy[i,j,k], kz[k]*uz[i,j,k]))
        end
      end
    end
end