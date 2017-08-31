function cross!(outx::AbstractArray{T},outy::AbstractArray{T},outz::AbstractArray{T},
                ux::AbstractArray{T},uy::AbstractArray{T},uz::AbstractArray{T},
                vx::AbstractArray{T},vy::AbstractArray{T},vz::AbstractArray{T},
                s::AbstractParameters{Nx,Ny,Nz,Lcs,Lcv,Nrx,Lrs,Lrv,Tr}) where {T,Nx,Ny,Nz,Lcs,Lcv,Nrx,Lrs,Lrv,Tr}
  if Tr
    _tcross!(outx,outy,outz,ux,uy,uz,vx,vy,vz,s)
  else
    for i in 1:(T<:Real ? Lrs : Lcs)
      @inbounds outx[i] = uy[i]*vz[i] - uz[i]*vy[i]
      @inbounds outy[i] = uz[i]*vx[i] - ux[i]*vz[i]
      @inbounds outz[i] = ux[i]*vy[i] - uy[i]*vx[i]
    end
  end
end

function _tcross!(outx::AbstractArray{T},outy::AbstractArray{T},outz::AbstractArray{T},
                  ux::AbstractArray{T},uy::AbstractArray{T},uz::AbstractArray{T},
                  vx::AbstractArray{T},vy::AbstractArray{T},vz::AbstractArray{T},
                  s::AbstractParameters{Nx,Ny,Nz,Lcs,Lcv,Nrx,Lrs,Lrv,Tr}) where {T,Nx,Ny,Nz,Lcs,Lcv,Nrx,Lrs,Lrv,Tr}
  Threads.@threads for i in 1:(T<:Real ? Lrs : Lcs)
    @inbounds outx[i] = uy[i]*vz[i] - uz[i]*vy[i]
    @inbounds outy[i] = uz[i]*vx[i] - ux[i]*vz[i]
    @inbounds outz[i] = ux[i]*vy[i] - uy[i]*vx[i]
  end
end

function crossk!(outx::AbstractArray{T,3},outy::AbstractArray{T,3},outz::AbstractArray{T,3},
                ux::StaticArray,uy::StaticArray,uz::StaticArray,
                vx::AbstractArray{T,3},vy::AbstractArray{T,3},vz::AbstractArray{T,3},s::AbstractParameters{Nx,Ny,Nz,Lcs,Lcv,Nrx,Lrs,Lrv,Tr}) where {T,Nx,Ny,Nz,Lcs,Lcv,Nrx,Lrs,Lrv,Tr}

  if Tr
    _tcrossk!(outx,outy,outz,ux,uy,uz,vx,vy,vz,s)
  else
    for k in 1:Nz
      for j in 1:Ny
        for i in 1:Nx
          @inbounds outx[i,j,k] = im*(uy[j]*vz[i,j,k] - uz[k]*vy[i,j,k])
          @inbounds outy[i,j,k] = im*(uz[k]*vx[i,j,k] - ux[i]*vz[i,j,k])
          @inbounds outz[i,j,k] = im*(ux[i]*vy[i,j,k] - uy[j]*vx[i,j,k])
        end
      end
    end
  end
end

function _tcrossk!(outx,outy,outz,ux,uy,uz,vx,vy,vz,s::AbstractParameters{Nx,Ny,Nz,Lcs,Lcv,Nrx,Lrs,Lrv,Tr}) where {Nx,Ny,Nz,Lcs,Lcv,Nrx,Lrs,Lrv,Tr}
  Threads.@threads for k in 1:Nz
    for j in 1:Ny
      for i in 1:Nx
        @inbounds outx[i,j,k] = im*(uy[j]*vz[i,j,k] - uz[k]*vy[i,j,k])
        @inbounds outy[i,j,k] = im*(uz[k]*vx[i,j,k] - ux[i]*vz[i,j,k])
        @inbounds outz[i,j,k] = im*(ux[i]*vy[i,j,k] - uy[j]*vx[i,j,k])
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
  crossk!(out.cx,out.cy,out.cz,s.kx,s.ky,s.kz,u.cx,u.cy,u.cz,s)
  return out
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

function scalar_advection!(out::VectorField,scalar::AbstractArray,v::VectorField,s::ScalarParameters{Nx,Ny,Nz,Lcs,Lcv,Nrx,Lrs,Lrv,Tr}) where {Nx,Ny,Nz,Lcs,Lcv,Nrx,Lrs,Lrv,Tr}

  _scalar_advection!(out.rx,rawreal(scalar),v.rx,s)
  _scalar_advection!(out.ry,rawreal(scalar),v.ry,s)
  _scalar_advection!(out.rz,rawreal(scalar),v.rz,s)

end

function _scalar_advection!(out::AbstractArray{Float64,3},scalar::AbstractArray{Float64,3},v::AbstractArray{Float64,3},s::ScalarParameters{Nx,Ny,Nz,Lcs,Lcv,Nrx,Lrs,Lrv,Tr}) where {Nx,Ny,Nz,Lcs,Lcv,Nrx,Lrs,Lrv,Tr}
  if Tr
    _tscalar_advection!(out,scalar,v,s)
  else
    for i in 1:Lrs
      @inbounds out[i] = scalar[i]*v[i]
    end
  end
end

function _tscalar_advection!(out::AbstractArray{Float64,3},scalar::AbstractArray{Float64,3},v::AbstractArray{Float64,3},s::ScalarParameters{Nx,Ny,Nz,Lcs,Lcv,Nrx,Lrs,Lrv,Tr}) where {Nx,Ny,Nz,Lcs,Lcv,Nrx,Lrs,Lrv,Tr}
  Threads.@threads for i in 1:Lrs
      @inbounds out[i] = scalar[i]*v[i]
  end
end

function div!(out::AbstractArray{Complex128,3},kx,ky,kz,ux,uy,uz,w,mdρdz::Real, s::ScalarParameters{Nx,Ny,Nz,Lcs,Lcv,Nrx,Lrs,Lrv,Tr}) where {Nx,Ny,Nz,Lcs,Lcv,Nrx,Lrs,Lrv,Tr}
  mim = -im
  if Tr
    _tdiv!(out,kx,ky,kz,ux,uy,uz,w,mdρdz,s,mim)
  else
    for k in 1:Nz
      for j in 1:Ny
        for i in 1:Nx
         # @inbounds out[i,j,k] = mim*(kx[i]*ux[i,j,k] + ky[j]*uy[i,j,k] + kz[k]*uz[i,j,k]) + mdρdz*w[i,j,k]
          @inbounds out[i,j,k] = muladd(mim,muladd(kx[i], ux[i,j,k], muladd(ky[j], uy[i,j,k], kz[k]*uz[i,j,k])), mdρdz*w[i,j,k])
        end
      end
    end
  end
end

function _tdiv!(out::AbstractArray{Complex128,3},kx,ky,kz,ux,uy,uz,w,mdρdz::Real, s::ScalarParameters{Nx,Ny,Nz,Lcs,Lcv,Nrx,Lrs,Lrv,Tr},mim) where {Nx,Ny,Nz,Lcs,Lcv,Nrx,Lrs,Lrv,Tr}
    Threads.@threads for k in 1:Nz
      for j in 1:Ny
        for i in 1:Nx
         # @inbounds out[i,j,k] = mim*(kx[i]*ux[i,j,k] + ky[j]*uy[i,j,k] + kz[k]*uz[i,j,k]) + mdœÅdz*w[i,j,k]
          @inbounds out[i,j,k] = muladd(mim,muladd(kx[i], ux[i,j,k], muladd(ky[j], uy[i,j,k], kz[k]*uz[i,j,k])), mdρdz*w[i,j,k])
        end
      end
    end
end

function div!(out::AbstractArray{Complex128,3},kx,ky,kz,ux,uy,uz, s::AbstractParameters{Nx,Ny,Nz,Lcs,Lcv,Nrx,Lrs,Lrv,Tr}) where {Nx,Ny,Nz,Lcs,Lcv,Nrx,Lrs,Lrv,Tr}
  mim = -im

  if Tr
    _tdiv!(out,kx,ky,kz,ux,uy,uz,s,mim)
  else
    for k in 1:Nz
      for j in 1:Ny
        for i in 1:Nx
          #@inbounds out[i,j,k] = -im*(kx[i]*ux[i,j,k] + ky[j]*uy[i,j,k] + kz[k]*uz[i,j,k])
          @inbounds out[i,j,k] = mim*muladd(kx[i], ux[i,j,k], muladd(ky[j], uy[i,j,k], kz[k]*uz[i,j,k]))
        end
      end
    end
  end
end

function _tdiv!(out::AbstractArray{Complex128,3},kx,ky,kz,ux,uy,uz, s::AbstractParameters{Nx,Ny,Nz,Lcs,Lcv,Nrx,Lrs,Lrv,Tr},mim) where {Nx,Ny,Nz,Lcs,Lcv,Nrx,Lrs,Lrv,Tr}
    Threads.@threads for k in 1:Nz
      for j in 1:Ny
        for i in 1:Nx
          #@inbounds out[i,j,k] = -im*(kx[i]*ux[i,j,k] + ky[j]*uy[i,j,k] + kz[k]*uz[i,j,k])
          @inbounds out[i,j,k] = mim*muladd(kx[i], ux[i,j,k], muladd(ky[j], uy[i,j,k], kz[k]*uz[i,j,k]))
        end
      end
    end
end