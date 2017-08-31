function Euller!(u::AbstractArray{Float64,N},rhs::AbstractArray,dt::Real,s::AbstractParameters{Nx,Ny,Nz,Lcs,Lcv,Nrx,Lrs,Lrv,Tr}) where {N,Nx,Ny,Nz,Lcs,Lcv,Nrx,Lrs,Lrv,Tr}
  if Tr
    _tEuller!(u,rhs,dt,s)
 else
    for i in 1:(N==3 ? Lrs : Lrv)
      #@inbounds u[i] += dt*rhs[i]
      @inbounds u[i] = muladd(dt,rhs[i],u[i])
    end
  end 
  return nothing
end

function _tEuller!(u::AbstractArray{Float64,N},rhs::AbstractArray,dt::Real,s::AbstractParameters{Nx,Ny,Nz,Lcs,Lcv,Nrx,Lrs,Lrv,Tr}) where {N,Nx,Ny,Nz,Lcs,Lcv,Nrx,Lrs,Lrv,Tr}
  Threads.@threads for i in 1:(N==3 ? Lrs : Lrv)
    #@inbounds u[i] += dt*rhs[i]
    @inbounds u[i] = muladd(dt,rhs[i],u[i])
  end
end

function Adams_Bashforth3rdO!(u::AbstractArray{Float64,N}, rhs::AbstractArray, dt::Real, rm1::AbstractArray, rm2::AbstractArray, s::AbstractParameters{Nx,Ny,Nz,Lcs,Lcv,Nrx,Lrs,Lrv,Tr}) where {N,Nx,Ny,Nz,Lcs,Lcv,Nrx,Lrs,Lrv,Tr}
  dt12 = dt/12

  if Tr
    _tAdams_Bashforth3rdO!(u,rhs,dt12,rm1,rm2,s)
  else
    for i in 1:(N==3 ? Lrs : Lrv)
      #@inbounds u[i] += dt12*(23*rhs[i] - 16rm1[i] + 5rm2[i])
      @inbounds u[i] = muladd(muladd(23, rhs[i], muladd(-16, rm1[i], 5rm2[i])), dt12::Float64, u[i])
    end
  end

  copy!(rm2,rm1)
  copy!(rm1,rhs)
  return nothing
end

function _tAdams_Bashforth3rdO!(u::AbstractArray{Float64,N}, rhs::AbstractArray, dt12::Real, rm1::AbstractArray, rm2::AbstractArray, s::AbstractParameters{Nx,Ny,Nz,Lcs,Lcv,Nrx,Lrs,Lrv,Tr}) where {N,Nx,Ny,Nz,Lcs,Lcv,Nrx,Lrs,Lrv,Tr}
    Threads.@threads for i in 1:(N==3 ? Lrs : Lrv)
      #@inbounds u[i] += dt12*(23*rhs[i] - 16rm1[i] + 5rm2[i])
      @inbounds u[i] = muladd(muladd(23, rhs[i], muladd(-16, rm1[i], 5rm2[i])), dt12, u[i])
    end
end
