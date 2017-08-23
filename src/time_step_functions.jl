function Euller!(u::AbstractArray,rhs::AbstractArray,dt::Real,s::AbstractParameters{Nx,Ny,Nz,Lcs,Lcv,Nrx,Lrs,Lrv}) where {Nx,Ny,Nz,Lcs,Lcv,Nrx,Lrs,Lrv}
  for i in 1:Lcv
    @inbounds u[i] += dt*rhs[i]
  end
end

function Adams_Bashforth3rdO!(u::AbstractArray, rhs::AbstractArray, dt::Real, rm1::AbstractArray, rm2::AbstractArray, s::AbstractParameters{Nx,Ny,Nz,Lcs,Lcv,Nrx,Lrs,Lrv}) where {Nx,Ny,Nz,Lcs,Lcv,Nrx,Lrs,Lrv}
  dt12 = dt/12

  for i in 1:Lcv
    @inbounds u[i] += dt12*(23*rhs[i] - 16rm1[i] + 5rm2[i])
  end

  copy!(rm2,rm1)
  copy!(rm1,rhs)
end
