@par function cross!(outx::AbstractArray{T},outy::AbstractArray{T},outz::AbstractArray{T},
  ux::AbstractArray{T},uy::AbstractArray{T},uz::AbstractArray{T},
  vx::AbstractArray{T},vy::AbstractArray{T},vz::AbstractArray{T},
  s::@par(AbstractSimulation)) where {T<:Complex}
  @mthreads for k in Kzr
    for y in Kyr, j in y
      @fastmath @inbounds @msimd for i in Kxr
        outx[i,j,k] = uy[i,j,k]*vz[i,j,k] - uz[i,j,k]*vy[i,j,k]
        outy[i,j,k] = uz[i,j,k]*vx[i,j,k] - ux[i,j,k]*vz[i,j,k]
        outz[i,j,k] = ux[i,j,k]*vy[i,j,k] - uy[i,j,k]*vx[i,j,k]
      end
    end
  end
end

@par function crossk!(outx,outy,outz,vx,vy,vz,s::@par(AbstractSimulation)) 
  @mthreads for k in Kzr
    for y in Kyr, j in y
      @fastmath @inbounds @msimd for i in Kxr 
        outx[i,j,k] = im*(ky[j]*vz[i,j,k] - kz[k]*vy[i,j,k])
        outy[i,j,k] = im*(kz[k]*vx[i,j,k] - kx[i]*vz[i,j,k])
        outz[i,j,k] = im*(kx[i]*vy[i,j,k] - ky[j]*vx[i,j,k])
      end
    end
  end  
end

function ccross!(out::VectorField,u::VectorField,v::VectorField,s::AbstractSimulation)
  cross!(out.cx,out.cy,out.cz,u.cx,u.cy,u.cz,v.cx,v.cy,v.cz,s)
  return out
end

function curl!(out::VectorField,u::VectorField,s::AbstractSimulation)
  crossk!(out.cx,out.cy,out.cz,u.cx,u.cy,u.cz,s)
  return out
end

@par function grad!(out::VectorField,f::AbstractArray{<:Complex,3},s::@par(AbstractSimulation))
  outx = out.cx
  outy = out.cy
  outz = out.cz
  @mthreads for k in Kzr
    for y in Kyr, j in y
      @inbounds @msimd for i in Kxr
        outx[i,j,k] = f[i,j,k]*im*kx[i]
        outy[i,j,k] = f[i,j,k]*im*ky[j]
        outz[i,j,k] = f[i,j,k]*im*kz[k]
      end
    end
  end
  #dx!(out.cx,f,s)
  #dy!(out.cy,f,s)
  #dz!(out.cz,f,s)
  #dealias!(out,s)
end

@par function dx!(out::AbstractArray{<:Complex,3},f::AbstractArray{<:Complex,3},s::@par(AbstractSimulation)) 
  @mthreads for k in Kzr
    for y in Kyr, j in y
      @inbounds @msimd for i in Kxr
        out[i,j,k] = f[i,j,k]*im*kx[i]
      end
    end
  end
end

@par function dy!(out::AbstractArray{<:Complex,3},f::AbstractArray{<:Complex,3},s::@par(AbstractSimulation)) 
  @mthreads for k in Kzr
    for y in Kyr, j in y
      @inbounds @msimd for i in Kxr
        out[i,j,k] = f[i,j,k]*im*ky[j]
      end
    end
  end
end

@par function dz!(out::AbstractArray{<:Complex,3},f::AbstractArray{<:Complex,3},s::@par(AbstractSimulation)) 
  @mthreads for k in Kzr
    for y in Kyr, j in y
      @inbounds @msimd for i in Kxr
        out[i,j,k] = f[i,j,k]*im*kz[k]
      end
    end
  end
end

function scalar_advection!(out::VectorField,scalar::AbstractArray,v::VectorField,s::AbstractSimulation)

  _scalar_advection!(out.rx,parent(real(scalar)),v.rx,s)
  _scalar_advection!(out.ry,parent(real(scalar)),v.ry,s)
  _scalar_advection!(out.rz,parent(real(scalar)),v.rz,s)

end

@par function _scalar_advection!(out::AbstractArray{Float64,3},scalar::AbstractArray{Float64,3},v::AbstractArray{Float64,3},s::@par(AbstractSimulation))
  @mthreads for i in 1:Lrs
      @fastmath @inbounds out[i] = scalar[i]*v[i]
  end
end

@par function div!(out::AbstractArray{Complex128,3},ux,uy,uz,w,mdρdz::Real, s::@par(AbstractSimulation))
  mim::Complex128 = -im
  @mthreads for k in Kzr
    for y in Kyr, j in y
      @fastmath @inbounds @msimd for i in Kxr
       # out[i,j,k] = mim*(kx[i]*ux[i,j,k] + ky[j]*uy[i,j,k] + kz[k]*uz[i,j,k]) + mdœÅdz*w[i,j,k]
        out[i,j,k] = muladd(mim,muladd(kx[i], ux[i,j,k], muladd(ky[j], uy[i,j,k], kz[k]*uz[i,j,k])), mdρdz*w[i,j,k])
      end
    end
  end
end

@par function div!(out::AbstractArray{Complex128,3},ux,uy,uz, s::@par(AbstractSimulation))
  mim::Complex128 = -im
  @mthreads for k in Kzr
    for y in Kyr, j in y
      @fastmath @inbounds @msimd for i in Kxr
        #out[i,j,k] = -im*(kx[i]*ux[i,j,k] + ky[j]*uy[i,j,k] + kz[k]*uz[i,j,k])
        out[i,j,k] = mim*muladd(kx[i], ux[i,j,k], muladd(ky[j], uy[i,j,k], kz[k]*uz[i,j,k]))
      end
    end
  end
end