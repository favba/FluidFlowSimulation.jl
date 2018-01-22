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
  curl!(s.aux,s.u,s)
  return nothing
end

@par function realspace!(s::A) where {A<:@par(AbstractSimulation)}
  #A_mul_B!(real(s.u),s.p.pinv.p,complex(s.u))
  s.p.pinv.p*s.u
  #A_mul_B!(real(s.aux),s.p.pinv.p,complex(s.aux))
  s.p.pinv.p*s.aux
  
  #haspassivescalar(A) && A_mul_B!(real(s.passivescalar.ρ),s.passivescalar.ps.pinv.p,complex(s.passivescalar.ρ))
  haspassivescalar(A) && s.passivescalar.ps.pinv.p * s.passivescalar.ρ
  #hasdensity(A) && A_mul_B!(real(s.densitystratification.ρ),s.densitystratification.ps.pinv.p,complex(s.densitystratification.ρ))
  hasdensity(A) && s.densitystratification.ps.pinv.p * s.densitystratification.ρ
  
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
end

@par function fourierspacep2!(s::A) where {A<:@par(AbstractSimulation)}
  add_viscosity!(s.rhs,s.u,ν,s)
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


@par function compute_nonlinear!(s::A) where {A<:@par(AbstractSimulation)}
  A_mul_B!(real(s.u),s.p.pinv.p,complex(s.u))
  A_mul_B!(real(s.aux),s.p.pinv.p,complex(s.aux))
  
  isscalar(A) && A_mul_B!(real(s.ρ),s.ps.pinv.p,complex(s.ρ))
  
  realspace!(s)
  s.p*s.rhs
  dealias!(s.rhs, s)
  s.p*s.u
  if isscalar(A) 
    s.p*s.aux
    dealias!(s.aux, s)
    s.ps*s.ρ
    gdir = GDirec === :x ? s.u.cx : GDirec === :y ? s.u.cy : s.u.cz 
    div!(complex(s.ρrhs), s.aux.cx, s.aux.cy, s.aux.cz, gdir, -s.dρdz, s)
  else
    dealias!(s.aux,s)
  end
  return nothing
end


@inline @par function dealias!(rhs::VectorField,s::@par(AbstractSimulation))
  dealias!(rhs.cx,s.dealias,s)
  dealias!(rhs.cy,s.dealias,s)
  dealias!(rhs.cz,s.dealias,s)
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
