@fastmath function cross!(outx::AbstractArray,outy::AbstractArray,outz::AbstractArray,
                ux::AbstractArray,uy::AbstractArray,uz::AbstractArray,
                vx::AbstractArray,vy::AbstractArray,vz::AbstractArray)
 @inbounds for i in 1:length(outx)
    outx[i] = uy[i]*vz[i] - uz[i]*vy[i]
    outy[i] = uz[i]*vx[i] - ux[i]*vz[i]
    outz[i] = ux[i]*vy[i] - uy[i]*vx[i]
  end
end

@fastmath function cross!(outx::AbstractArray{T,3},outy::AbstractArray{T,3},outz::AbstractArray{T,3},
                ux::AbstractArray{T,1},uy::AbstractArray{T,1},uz::AbstractArray{T,1},
                vx::AbstractArray{T,3},vy::AbstractArray{T,3},vz::AbstractArray{T,3}) where T
  nx,ny,nz = size(outx)
  @inbounds for k in 1:nz
    @inbounds for j in 1:ny
      @inbounds for i in 1:nx
        outx[i,j,k] = uy[j]*vz[i,j,k] - uz[k]*vy[i,j,k]
        outy[i,j,k] = uz[k]*vx[i,j,k] - ux[i]*vz[i,j,k]
        outz[i,j,k] = ux[i]*vy[i,j,k] - uy[j]*vx[i,j,k]
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

function curl!(out::VectorField,u::VectorField)
    cross!(out.cx,out.cy,out.cz,ikx,iky,ikz,u.cx,u.cy,u.cz)
end