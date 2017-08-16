function Euller!(u::VectorField,rhs::VectorField,dt::Real,s::AbstractParameters{Nx,Ny,Nz}) where {Nx,Ny,Nz}
  for i in 1:3*Nx*Ny*Nz
    @inbounds u[i] += dt*rhs[i]
  end
end

function Adams_Bashforth3rdO!(u::VectorField, rhs::VectorField, dt::Real, rm1::AbstractArray, rm2::AbstractArray, s::AbstractParameters{Nx,Ny,Nz}) where {Nx,Ny,Nz}
  dt12 = dt/12

  for i in 1:3*Nx*Ny*Nz
    @inbounds u[i] += dt12*(23*rhs[i] - 16rm1[i] + 5rm2[i])
  end

  copy!(rm2,rm1)
  copy!(rm1,rhs)
end
