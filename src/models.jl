@inline @par function advance_in_time!(s::A,dt::Real) where {A<:@par(AbstractSimulation)}
  calculate_rhs!(s)
  time_step!(s,dt)
  return nothing
end

@par function calculate_rhs!(s::A) where {A<:@par(AbstractSimulation)}
  fourierspacep1!(s)
  realspace!(s)
  fourierspacep2!(s)
  return nothing
end

@par function fourierspacep1!(s::A) where {A<:@par(AbstractSimulation)}
  @mthreads for k in Kzr
    fourierspacep1!(k,s)
  end
  return nothing
end

@inline @par function fourierspacep1!(k,s::@par(AbstractSimulation)) 
  auxx = s.aux.cx
  auxy = s.aux.cy
  auxz = s.aux.cz
  ux = s.u.cx
  uy = s.u.cy
  uz = s.u.cz
  if hasles(s)
    txx = s.lesmodel.tau.cxx
    txy = s.lesmodel.tau.cxy
    txz = s.lesmodel.tau.cxz
    tyy = s.lesmodel.tau.cyy
    tyz = s.lesmodel.tau.cyz
  end
  #@inbounds for k in Kzr[j]
    @inbounds for y in Kyr, j in y
      @msimd for i in Kxr 
        auxx[i,j,k] = im*(ky[j]*uz[i,j,k] - kz[k]*uy[i,j,k])
        auxy[i,j,k] = im*(kz[k]*ux[i,j,k] - kx[i]*uz[i,j,k])
        auxz[i,j,k] = im*(kx[i]*uy[i,j,k] - ky[j]*ux[i,j,k])
        if hasles(s)
          tr = (im*ky[j]*uy[i,j,k] + im*kx[i]*ux[i,j,k] + im*kz[k]*uz[i,j,k])
          txx[i,j,k] = -im*kx[i]*ux[i,j,k] + tr
          tyy[i,j,k] = -im*ky[j]*uy[i,j,k] + tr
          txy[i,j,k] = -im*kx[i]*uy[i,j,k] -im*ky[j]*ux[i,j,k]
          txz[i,j,k] = -im*kx[i]*uz[i,j,k] -im*kz[k]*ux[i,j,k]
          tyz[i,j,k] = -im*ky[j]*uz[i,j,k] -im*kz[k]*uy[i,j,k]
        end
      end
    end
end

@par function realspace!(s::A) where {A<:@par(AbstractSimulation)}
  #A_mul_B!(real(s.u),s.p.pinv.p,complex(s.u))
  s.pb*s.u
  #A_mul_B!(real(s.aux),s.p.pinv.p,complex(s.aux))
  s.pb*s.aux
  
  #haspassivescalar(A) && A_mul_B!(real(s.passivescalar.ρ),s.passivescalar.ps.pinv.p,complex(s.passivescalar.ρ))
  haspassivescalar(A) && s.passivescalar.pbs * s.passivescalar.ρ
  #hasdensity(A) && A_mul_B!(real(s.densitystratification.ρ),s.densitystratification.ps.pinv.p,complex(s.densitystratification.ρ))
  hasdensity(A) && s.densitystratification.pbs * s.densitystratification.ρ
 
  hasles(s) && s.lesmodel.pt * s.lesmodel.tau
  
  realspacecalculation!(s)

  s.p*s.rhs
  dealias!(s.rhs, s)
  s.p*s.u
  if haspassivescalar(A) 
    s.p*s.aux
    dealias!(s.aux, s)
    s.passivescalar.ps * s.passivescalar.ρ
  elseif hasdensity(A)
    s.p*s.aux
    dealias!(s.aux, s)
    s.densitystratification.ps * s.densitystratification.ρ
  else
    dealias!(s.aux, s)
  end

  if hasles(s)
    s.lesmodel.pt * s.lesmodel.tau
    dealias!(s.lesmodel.tau,s) #Implement!
  end

  return nothing
end

@par function realspacecalculation!(s::A) where {A<:@par(AbstractSimulation)}
  @assert !(hasdensity(A) & haspassivescalar(A))
  n = Nt
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
  if haspassivescalar(A)
    ρ = parent(real(s.passivescalar.ρ))
  end
  if hasdensity(A)
    ρ = parent(real(s.densitystratification.ρ))
  end

  if hasles(s)
    txx = s.lesmodel.tau.rxx
    tyy = s.lesmodel.tau.ryy
    txy = s.lesmodel.tau.rxy
    txz = s.lesmodel.tau.rxz
    tyz = s.lesmodel.tau.ryz
    c = cs(s.lesmodel)
    Δ = Delta(s.lesmodel)
    α = c*c*Δ*Δ 
  end

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

    if hasles(s)
      txx[i] *= scale
      txy[i] *= scale
      txz[i] *= scale
      tyy[i] *= scale
      tyz[i] *= scale

      S = sqrt(2*(txx[i]^2 + tyy[i]^2 +(-txx[i]-tyy[i])^2 + 2*(txy[i]^2 + txz[i]^2 + tyz[i]^2)))
      txx[i] *= α*S
      txy[i] *= α*S
      txz[i] *= α*S
      tyy[i] *= α*S
      tyz[i] *= α*S
    end

  end
  return nothing
end

@par function fourierspacep2!(s::A) where {A<:@par(AbstractSimulation)}
  add_viscosity!(s.rhs,s.u,ν,s)
  hasles(s) && add_residual_tensor!(s.rhs,s.lesmodel.tau,s)
  if hasdensity(A)
    Gdirec = graddir(s.densitystratification)
    gdir = GDirec === :x ? s.rhs.cx : GDirec === :y ? s.rhs.cy : s.rhs.cz 
    addgravity!(gdir, complex(s.densitystratification.ρ), -gravity(s.densitystratification), s)
  end
  pressure_projection!(s.rhs.cx,s.rhs.cy,s.rhs.cz,s)
  if haspassivescalar(A)
    gdir = graddir(s.passivescalar) === :x ? s.u.cx : graddir(s.passivescalar) === :y ? s.u.cy : s.u.cz 
    div!(complex(s.passivescalar.ρrhs), s.aux.cx, s.aux.cy, s.aux.cz, gdir, -meangradient(s.passivescalar), s)
    add_scalar_difusion!(complex(s.passivescalar.ρrhs),complex(s.passivescalar.ρ),diffusivity(s.passivescalar),s)
  end
  if hasdensity(A)
    gdir = graddir(s.densitystratification) === :x ? s.u.cx : graddir(s.densitystratification) === :y ? s.u.cy : s.u.cz 
    div!(complex(s.densitystratification.ρrhs), s.aux.cx, s.aux.cy, s.aux.cz, gdir, -meangradient(s.densitystratification), s)
    add_scalar_difusion!(complex(s.densitystratification.ρrhs),complex(s.densitystratification.ρ),diffusivity(s.densitystratification),s)
  end
  return nothing
end

@par function add_residual_tensor!(rhs::VectorField,τ::SymmetricTracelessTensor,s::@par(AbstractSimulation))
  @mthreads for k in Kzr
    add_residual_tensor!(rhs,τ,k,s)
  end
end

@inline @par function add_residual_tensor!(rhs::VectorField,tau::SymmetricTracelessTensor,k::Int,s::@par(AbstractSimulation))
  rx = rhs.cx
  ry = rhs.cy
  rz = rhs.cz
  txx = tau.cxx
  txy = tau.cxy
  txz = tau.cxz
  tyy = tau.cyy
  tyz = tau.cyz
  @inbounds for y in Kyr, j in y
    @msimd for i in Kxr
      rx[i,j,k] += im*(kx[i]*txx[i,j,k] + ky[j]*txy[i,j,k] + kz[k]*txz[i,j,k])
      ry[i,j,k] += im*(kx[i]*txy[i,j,k] + ky[j]*tyy[i,j,k] + kz[k]*tyz[i,j,k])
      rz[i,j,k] += im*(kx[i]*txz[i,j,k] + ky[j]*tyz[i,j,k] + kz[k]*(-txx[i,j,k]-tyy[i,j,k]))
    end
  end
end


@inline @par function dealias!(rhs::VectorField,s::@par(AbstractSimulation))
  dealias!(rhs.cx,s.dealias,s)
  dealias!(rhs.cy,s.dealias,s)
  dealias!(rhs.cz,s.dealias,s)
end

@inline @par function dealias!(rhs::SymmetricTracelessTensor,s::@par(AbstractSimulation))
  dealias!(rhs.cxx,s.dealias,s)
  dealias!(rhs.cxy,s.dealias,s)
  dealias!(rhs.cxz,s.dealias,s)
  dealias!(rhs.cyy,s.dealias,s)
  dealias!(rhs.cyz,s.dealias,s)
end

dealias!(rhs::AbstractArray{<:Complex,3},s::AbstractSimulation) = dealias!(rhs,s.dealias,s)

@inline @par function dealias!(rhs::AbstractArray{T,3},dealias,s::@par(AbstractSimulation)) where {T<:Complex}
 @mthreads for i = 1:Lcs
  @inbounds begin
    dealias[i] && (rhs[i] = zero(T))
  end
 end
end

function add_viscosity!(rhs::VectorField,u::VectorField,ν::Real,s::AbstractSimulation)
  _add_viscosity!(rhs.cx,u.cx,-ν,s)
  _add_viscosity!(rhs.cy,u.cy,-ν,s)
  _add_viscosity!(rhs.cz,u.cz,-ν,s)
end

@par function _add_viscosity!(rhs::AbstractArray,u::AbstractArray,mν::Real,s::@par(AbstractSimulation))
  @mthreads for k in Kzr
    for y in Kyr, j in y
      @fastmath @inbounds @msimd for i in Kxr
        #rhs[i,j,k] = (kx[i]*kx[i] + ky[j]*ky[j] + kz[k]*kz[k])*mŒΩ*u[i,j,k] + rhs[i,j,k]
        rhs[i,j,k] = muladd(muladd(kx[i], kx[i], muladd(ky[j], ky[j], kz[k]*kz[k])), mν*u[i,j,k], rhs[i,j,k])
      end
    end
  end
end

function add_scalar_difusion!(rhs::AbstractArray,u::AbstractArray,ν::Real,s::AbstractSimulation)
  _add_viscosity!(rhs,u,-ν,s)
end

 @par function pressure_projection!(rhsx,rhsy,rhsz,s::@par(AbstractSimulation))
  @inbounds a = (rhsx[1],rhsy[1],rhsz[1])
  @mthreads for k in Kzr
    for y in Kyr, j in y
      @fastmath @inbounds @msimd for i in Kxr
        #p1 = -(kx[i]*rhsx[i,j,k] + ky[j]*rhsy[i,j,k] + kz[k]*rhsz[i,j,k])/(kx[i]*kx[i] + ky[j]*ky[j] + kz[k]*kz[k])
        p1 = -muladd(kx[i], rhsx[i,j,k], muladd(ky[j], rhsy[i,j,k], kz[k]*rhsz[i,j,k]))/muladd(kx[i], kx[i], muladd(ky[j], ky[j],  kz[k]*kz[k]))
        rhsx[i,j,k] = muladd(kx[i],p1,rhsx[i,j,k])
        rhsy[i,j,k] = muladd(ky[j],p1,rhsy[i,j,k])
        rhsz[i,j,k] = muladd(kz[k],p1,rhsz[i,j,k])
      end
    end
  end
  @inbounds rhsx[1],rhsy[1],rhsz[1] = a
end

@par function addgravity!(rhs,ρ,g::Real,s::@par(BoussinesqSimulation))
  @mthreads for k in Kzr
    for y in Kyr, j in y
      @fastmath @inbounds @msimd for i in Kxr
        rhs[i,j,k] = muladd(ρ[i,j,k],g,rhs[i,j,k])
      end
    end
  end
end

@par function time_step!(s::A,dt::Real) where {A<:@par(AbstractSimulation)}
  s.timestep(s.u,s.rhs,dt,s)
  haspassivescalar(s) && s.passivescalar.timestep(parent(real(s.passivescalar.ρ)),parent(real(s.passivescalar.ρrhs)),dt,s)
  hasdensity(s) && s.densitystratification.timestep(parent(real(s.densitystratification.ρ)),parent(real(s.densitystratification.ρrhs)),dt,s)
end
