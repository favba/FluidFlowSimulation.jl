@inline @par function advance_in_time!(s::A) where {A<:@par(AbstractSimulation)}
  calculate_rhs!(s)
  hasforcing(s) && s.forcing(s)
  time_step!(s)
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
    if scalarmodel(s.lesmodel) === EddyDiffusion
      ρ = hasdensity(s) ? complex(s.densitystratification.ρ) : complex(s.passivescalar.ρ) 
      gρx = s.lesmodel.scalar.gradρ.cx
      gρy = s.lesmodel.scalar.gradρ.cy
      gρz = s.lesmodel.scalar.gradρ.cz
    end
  end
  #@inbounds for k in Kzr[j]
    @inbounds for y in Kyr, j in y
      @msimd for i in 1:(Kxr[k][j]) 
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
          if scalarmodel(s.lesmodel) === EddyDiffusion
            gρx[i,j,k] = -im*kx[i]*ρ[i,j,k]
            gρy[i,j,k] = -im*ky[j]*ρ[i,j,k]
            gρz[i,j,k] = -im*kz[k]*ρ[i,j,k]
          end
        end
      end
    end
end

@par function realspace!(s::A) where {A<:@par(AbstractSimulation)}
  irfft!(s.u,s.pb,s)
  irfft!(s.aux,s.pb,s)
  
  haspassivescalar(A) && irfft!(s.passivescalar.ρ, s.passivescalar.pbs, s)
  hasdensity(A) && irfft!(s.densitystratification.ρ, s.densitystratification.pbs, s)
 
  if hasles(s)
    irfft!(s.lesmodel.tau, s.lesmodel.pbt, s)
    (haspassivescalar(A) | hasdensity(A)) && irfft!(s.lesmodel.scalar.gradρ, s.pb, s)
  end
  
  realspacecalculation!(s)

  rfft!(s.rhs,s.p, s)
  #dealias!(s.rhs, s)
  rfft!(s.u,s.p, s)
  if haspassivescalar(A) 
    rfft!(s.aux,s.p,s)
    #dealias!(s.aux, s)
    rfft!(s.passivescalar.ρ, s.passivescalar.ps, s)
  elseif hasdensity(A)
    rfft!(s.aux, s.p, s)
    #dealias!(s.aux, s)
    rfft!(s.densitystratification.ρ, s.densitystratification.ps, s)
  else
    dealias!(s.aux, s)
  end

  if hasles(s)
    rfft!(s.lesmodel.tau, s.lesmodel.pt, s)
    #dealias!(s.lesmodel.tau,s)
    if (haspassivescalar(A) | hasdensity(A))
      dealias!(s.lesmodel.scalar.gradρ,s)
    end
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
  ux = s.u.rx
  uy = s.u.ry
  uz = s.u.rz
  ωx = s.aux.rx
  ωy = s.aux.ry
  ωz = s.aux.rz
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
    is_SandP(s.lesmodel) && (β = cbeta(s.lesmodel)*Δ*Δ)
    if haspassivescalar(A) | hasdensity(A)
      gradrhox = s.lesmodel.scalar.gradρ.rx
      gradrhoy = s.lesmodel.scalar.gradρ.ry
      gradrhoz = s.lesmodel.scalar.gradρ.rz
    end
  end

  @inbounds @msimd for i in RealRanges[j]

    outx[i] = uy[i]*ωz[i] - uz[i]*ωy[i]
    outy[i] = uz[i]*ωx[i] - ux[i]*ωz[i]
    outz[i] = ux[i]*ωy[i] - uy[i]*ωx[i]

    if hasles(s)

      S = sqrt(2*(txx[i]^2 + tyy[i]^2 +(-txx[i]-tyy[i])^2 + 2*(txy[i]^2 + txz[i]^2 + tyz[i]^2)))
      νt = α*S

      if is_Smagorinsky(s.lesmodel)
        txx[i] *= νt
        txy[i] *= νt
        txz[i] *= νt
        tyy[i] *= νt
        tyz[i] *= νt
      elseif is_SandP(s.lesmodel)
        pxx = ωy[i]*txz[i] - ωz[i]*txy[i]
        #pxy = (-1/2)*ωx[i]*txz[i] + (1/2)*ωz[i]*txx[i] - ((-1/2)*ωy[i]*tyz[i] + (1/2)*ωz[i]*tyy[i])
        pxy = 0.5*(-ωx[i]*txz[i] + ωz[i]*txx[i] + ωy[i]*tyz[i] - ωz[i]*tyy[i])
        #pxz = (1/2)*ωx[i]*txy[i] + (-1/2)*ωy[i]*txx[i] - ((1/2)*ωz[i]*tyz[i] + (-1/2)*(-txx[i] - tyy[i])*ωy[i])
        pxz = 0.5*(ωx[i]*txy[i] - ωz[i]*tyz[i] - tyy[i]*ωy[i]) - ωy[i]*txx[i]
        pyy = -ωx[i]*tyz[i] + ωz[i]*txy[i]
        #pyz = (1/2)*ωx[i]*tyy[i] + (-1/2)*ωy[i]*txy[i] - ((-1/2)*ωz[i]*txz[i] + (1/2)*(-txx[i] - tyy[i])*ωx[i])
        pyz = 0.5*(ωx[i]*txx[i] - ωy[i]*txy[i] + txz[i]*ωz[i]) + ωx[i]*tyy[i]

        txx[i] = νt*txx[i] + β*pxx
        txy[i] = νt*txy[i] + β*pxy
        txz[i] = νt*txz[i] + β*pxz
        tyy[i] = νt*tyy[i] + β*pyy
        tyz[i] = νt*tyz[i] + β*pyz
      end
      
    end

    if haspassivescalar(A) || hasdensity(A)
      ωx[i] = ux[i]*ρ[i]
      ωy[i] = uy[i]*ρ[i]
      ωz[i] = uz[i]*ρ[i]

      if scalarmodel(s.lesmodel) === EddyDiffusion
        ωx[i] += νt*gradrhox[i]
        ωy[i] += νt*gradrhoy[i]
        ωz[i] += νt*gradrhoz[i]
      end
    end

  end
  return nothing
end

@par function fourierspacep2!(s::A) where {A<:@par(AbstractSimulation)}
  add_viscosity!(s.rhs,s.u,ν,s)
  hasles(s) && add_residual_tensor!(s.rhs,s.lesmodel.tau,s)
  if hasdensity(A)
    Gdirec = graddir(s.densitystratification)
    gdir = Gdirec === :x ? s.rhs.cx : Gdirec === :y ? s.rhs.cy : s.rhs.cz 
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
    @msimd for i in 1:(Kxr[k][j])
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
      @fastmath @inbounds @msimd for i in 1:(Kxr[k][j])
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
      @fastmath @inbounds @msimd for i in 1:(Kxr[k][j])
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

@par function addgravity!(rhs,ρ,g::Real,s::@par(AbstractSimulation))
  @mthreads for k in Kzr
    for y in Kyr, j in y
      @fastmath @inbounds @msimd for i in 1:(Kxr[k][j])
        rhs[i,j,k] = muladd(ρ[i,j,k],g,rhs[i,j,k])
      end
    end
  end
end

@par function time_step!(s::A) where {A<:@par(AbstractSimulation)}
  s.timestep(s.u,s.rhs,s)
  haspassivescalar(s) && s.passivescalar.timestep(parent(real(s.passivescalar.ρ)),parent(real(s.passivescalar.ρrhs)),s)
  hasdensity(s) && s.densitystratification.timestep(parent(real(s.densitystratification.ρ)),parent(real(s.densitystratification.ρrhs)),s)
end
