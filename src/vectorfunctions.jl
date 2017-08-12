function cross!(outx::AbstractArray,outy::AbstractArray,outz::AbstractArray,
                ux::AbstractArray,uy::AbstractArray,uz::AbstractArray,
                vx::AbstractArray,vy::AbstractArray,vz::AbstractArray)
 for i in 1:length(outx)
    @inbounds outx[i] = uy[i]*vz[i] - uz[i]*vy[i]
    @inbounds outy[i] = uz[i]*vx[i] - ux[i]*vz[i]
    @inbounds outz[i] = ux[i]*vy[i] - uy[i]*vx[i]
  end
end

function cross!(outx::AbstractArray{T,3},outy::AbstractArray{T,3},outz::AbstractArray{T,3},
                ux::AbstractVector,uy::AbstractVector,uz::AbstractVector,
                vx::AbstractArray{T,3},vy::AbstractArray{T,3},vz::AbstractArray{T,3}) where T
  nx,ny,nz = size(outx)
  for k in 1:nz
    for j in 1:ny
      for i in 1:nx
        @inbounds outx[i,j,k] = uy[j]*vz[i,j,k] - uz[k]*vy[i,j,k]
        @inbounds outy[i,j,k] = uz[k]*vx[i,j,k] - ux[i]*vz[i,j,k]
        @inbounds outz[i,j,k] = ux[i]*vy[i,j,k] - uy[j]*vx[i,j,k]
      end
    end
  end
end


function rcross!(out::VectorField,u::VectorField,v::VectorField)
    cross!(out.rx,out.ry,out.rz,u.rx,u.ry,u.rz,v.rx,v.ry,v.rz)
end

function ccross!(out::VectorField,u::VectorField,v::VectorField)
    cross!(out.cx,out.cy,out.cz,u.cx,u.cy,u.cz,v.cx,v.cy,v.cz)
end

function curl!(out::VectorField,u::VectorField,kx::AbstractVector,ky::AbstractVector,kz::AbstractVector)
    cross!(out.cx,out.cy,out.cz,im .* kx,im .* ky,im .* kz,u.cx,u.cy,u.cz)
end

function rfftfreq(n::Integer,s::Real)
  d = 2π*s/n
  [(n/2 - i)/(d*n) for i = n/2:-1:0]
end

function fftfreq(n::Integer,s::Real)
  d = 2π*s/n
  if iseven(n)
    return vcat([(n/2 - i)/(d*n) for i = n/2:-1:1],[-i/(d*n) for i = n/2:-1:1])
  else return vcat([(n/2 - i)/(d*n) for i = n/2:-1:0],[-i/(d*n) for i = (n-1)/2:-1:1])
  end
end