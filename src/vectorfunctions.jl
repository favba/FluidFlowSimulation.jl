function cross!(outx::AbstractArray{T},outy::AbstractArray{T},outz::AbstractArray{T},
                ux::AbstractArray{T},uy::AbstractArray{T},uz::AbstractArray{T},
                vx::AbstractArray{T},vy::AbstractArray{T},vz::AbstractArray{T},
                s::AbstractParameters{Nx,Ny,Nz}) where {T<:Real,Nx,Ny,Nz}
 for i in 1:(2*Nx*Ny*Nz)
    @inbounds outx[i] = uy[i]*vz[i] - uz[i]*vy[i]
    @inbounds outy[i] = uz[i]*vx[i] - ux[i]*vz[i]
    @inbounds outz[i] = ux[i]*vy[i] - uy[i]*vx[i]
  end
end

function cross!(outx::AbstractArray,outy::AbstractArray,outz::AbstractArray,
                ux::AbstractArray,uy::AbstractArray,uz::AbstractArray,
                vx::AbstractArray,vy::AbstractArray,vz::AbstractArray,s::AbstractParameters{Nx,Ny,Nz}) where {Nx,Ny,Nz}
 for i in 1:Nx*Ny*Nz
    @inbounds outx[i] = uy[i]*vz[i] - uz[i]*vy[i]
    @inbounds outy[i] = uz[i]*vx[i] - ux[i]*vz[i]
    @inbounds outz[i] = ux[i]*vy[i] - uy[i]*vx[i]
  end
end

function crossk!(outx::AbstractArray{T,3},outy::AbstractArray{T,3},outz::AbstractArray{T,3},
                ux::StaticArray,uy::StaticArray,uz::StaticArray,
                vx::AbstractArray{T,3},vy::AbstractArray{T,3},vz::AbstractArray{T,3},s::AbstractParameters{Nx,Ny,Nz}) where {T,Nx,Ny,Nz}
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


function rcross!(out::VectorField,u::VectorField,v::VectorField,s::AbstractParameters)
    cross!(out.rx,out.ry,out.rz,u.rx,u.ry,u.rz,v.rx,v.ry,v.rz,s)
end

function ccross!(out::VectorField,u::VectorField,v::VectorField,s::AbstractParameters)
    cross!(out.cx,out.cy,out.cz,u.cx,u.cy,u.cz,v.cx,v.cy,v.cz,s)
end

function curl!(out::VectorField,u::VectorField,s::AbstractParameters)
    crossk!(out.cx,out.cy,out.cz,s.kx,s.ky,s.kz,u.cx,u.cy,u.cz,s)
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