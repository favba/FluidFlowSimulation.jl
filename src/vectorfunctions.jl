@par function realspacecalculation!(s::A) where {A<:@par(AbstractSimulation)}
  @assert !(hasdensity(A) & haspassivescalar(A))
  n = Thr ? Threads.nthreads() : 1
  @mthreads for j in 1:n
    realspacecalculation!(s,j)
  end
  return nothing
end

@par function realspacecalculation!(s::A,j::Integer) where {A<:@par(AbstractSimulation)} 
  scale::Float64 = 1/(Nrx*Ny*Nz)                
  mscale::Float64 = -1/(Nrx*Ny*Nz)                
  ux = s.u.rx
  uy = s.u.ry
  uz = s.u.rz
  vx = s.aux.rx
  vy = s.aux.ry
  vz = s.aux.rz
  outx = s.rhs.rx
  outy = s.rhs.ry
  outz = s.rhs.rz
  haspassivescalar(A) && (
    ρ = parent(real(s.passivescalar.ρ)) )
  hasdensity(A) && (
    ρ = parent(real(s.densitystratification.ρ)) )

  @inbounds @msimd for i in RealRanges[j]
    ux[i] *= scale
    uy[i] *= scale
    uz[i] *= scale

    outx[i] = muladd(scale*uy[i],vz[i], mscale*uz[i]*vy[i])
    outy[i] = muladd(scale*uz[i],vx[i], mscale*ux[i]*vz[i])
    outz[i] = muladd(scale*ux[i],vy[i], mscale*uy[i]*vx[i])

    if haspassivescalar(A) || hasdensity(A)
      ρ[i] *= scale
      vx[i] = ux[i]*ρ[i]
      vy[i] = uy[i]*ρ[i]
      vz[i] = uz[i]*ρ[i]
    end
  end
  return nothing
end

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

function grad!(out::VectorField,f::AbstractArray{<:Complex,3},s::AbstractSimulation)
  dx!(out.cx,f,s)
  dy!(out.cy,f,s)
  dz!(out.cz,f,s)
  dealias!(out,s)
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

function rfftfreq(n::Integer,s::Real)::Vector{Float64}
  Float64[(n/2 - i)/s for i = n/2:-1:0]
end

function fftfreq(n::Integer,s::Real)::Vector{Float64}
  if iseven(n)
    return vcat(Float64[(n/2 - i)/s for i = n/2:-1:1],Float64[-i/s for i = n/2:-1:1])
  else return vcat(Float64[(n/2 - i)/s for i = n/2:-1:0],Float64[-i/s for i = (n-1)/2:-1:1])
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