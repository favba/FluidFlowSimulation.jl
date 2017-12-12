@inline @par function advance_in_time!(s::A,dt::Real) where {A<:@par(AbstractParameters)}
  calculate_rhs!(s)
  time_step!(s,dt)
  return nothing
end

@par function calculate_rhs!(s::A) where {A<:@par(AbstractParameters)}
  compute_nonlinear!(s)
  add_viscosity!(s.rhs,s.u,s.ν,s)
  if A<:BoussinesqParameters
    gdir = GDirec === :x ? s.rhs.cx : GDirec === :y ? s.rhs.cy : s.rhs.cz 
    addgravity!(gdir, complex(s.ρ), -s.g, s)
  end
  pressure_projection!(s.rhs.cx,s.rhs.cy,s.rhs.cz,s)
  A<:ScalarParameters && add_scalar_difusion!(complex(s.ρrhs),complex(s.ρ),s.α,s)
end

@par function compute_nonlinear!(s::A) where {A<:@par(AbstractParameters)}
  curl!(s.aux,s.u,s)
  A_mul_B!(real(s.u),s.pinv.p,complex(s.u))
  A_mul_B!(real(s.aux),s.pinv.p,complex(s.aux))
  
  A<:ScalarParameters && A_mul_B!(real(s.ρ),s.ps.pinv.p,complex(s.ρ))
  
  realspace!(s.rhs, s.u, s.aux, s)
  s.p*s.rhs
  dealias!(s.rhs, s)
  s.p*s.u
  if A<:ScalarParameters
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


@inline @par function dealias!(rhs::VectorField,s::@par(AbstractParameters))
  dealias!(rhs.cx,s.dealias,s)
  dealias!(rhs.cy,s.dealias,s)
  dealias!(rhs.cz,s.dealias,s)
end

dealias!(rhs::AbstractArray{<:Complex,3},s::AbstractParameters) = dealias!(rhs,s.dealias,s)

@inline @par function dealias!(rhs::AbstractArray{T,3},dealias,s::@par(AbstractParameters)) where {T<:Complex}
 @mthreads for i = 1:Lcs
  @inbounds begin
    dealias[i] && (rhs[i] = zero(T))
  end
 end
end

function add_viscosity!(rhs::VectorField,u::VectorField,ν::Real,s::AbstractParameters)
  _add_viscosity!(rhs.cx,u.cx,-ν,s)
  _add_viscosity!(rhs.cy,u.cy,-ν,s)
  _add_viscosity!(rhs.cz,u.cz,-ν,s)
end

@par function _add_viscosity!(rhs::AbstractArray,u::AbstractArray,mν::Real,s::@par(AbstractParameters))
  @mthreads for k in Kzr
    for y in Kyr, j in y
      @fastmath @inbounds @msimd for i in Kxr
        #rhs[i,j,k] = (kx[i]*kx[i] + ky[j]*ky[j] + kz[k]*kz[k])*mŒΩ*u[i,j,k] + rhs[i,j,k]
        rhs[i,j,k] = muladd(muladd(kx[i], kx[i], muladd(ky[j], ky[j], kz[k]*kz[k])), mν*u[i,j,k], rhs[i,j,k])
      end
    end
  end
end

function add_scalar_difusion!(rhs::AbstractArray,u::AbstractArray,ν::Real,s::AbstractParameters)
  _add_viscosity!(rhs,u,-ν,s)
end

 @par function pressure_projection!(rhsx,rhsy,rhsz,s::@par(AbstractParameters))
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

@par function addgravity!(rhs,ρ,g::Real,s::@par(BoussinesqParameters))
  @mthreads for k in Kzr
    for y in Kyr, j in y
      @fastmath @inbounds @msimd for i in Kxr
        rhs[i,j,k] = muladd(ρ[i,j,k],g,rhs[i,j,k])
      end
    end
  end
end

@par function time_step!(s::A,dt::Real) where {A<:@par(AbstractParameters)}
  if Integrator === :Euller
    Euller!(rawreal(s.u),rawreal(s.rhs),dt,s)
    A <: ScalarParameters && Euller!(complex(s.ρ),complex(s.ρrhs),dt,s)
  elseif Integrator === :Adams_Bashforth3rdO
    Adams_Bashforth3rdO!(dt,s)
  end
end

@inbounds @par function mycopy!(rm::AbstractArray{T,3},rhs::AbstractArray{T,3},s::@par(AbstractParameters)) where T<:Complex
  @mthreads for kk in 1:length(Kzr)
    k = Kzr[kk]
    jj::Int = 1
    for y in Kyr, j in y
      for i in Kxr
        rm[i,jj,kk] = rhs[i,j,k]
      end
    jj+=1
    end
  end
end

@inbounds @par function mycopy!(rm::AbstractArray{T,3},rhs::AbstractArray{T,3},s::@par(AbstractParameters)) where T<:Real
  @mthreads for kk in 1:length(Kzr)
    k = Kzr[kk]
    jj::Int = 1
    for y in Kyr, j in y
      for i in 1:(2length(Kxr))
        rm[i,jj,kk] = rhs[i,j,k]
      end
    jj+=1
    end
  end
end

@par function mycopy!(out::VectorField,inp::VectorField,s::@par(AbstractParameters))
  _mycopy!(out.cx,inp.cx,s)
  _mycopy!(out.cy,inp.cy,s)
  _mycopy!(out.cz,inp.cz,s)
end

@par function _mycopy!(out::Array{Complex128,3},inp::Array{Complex128,3},s::@par(AbstractParameters))
  @mthreads for k in Kzr
    for y in Kyr, j in y
      for i in Kxr
        @inbounds out[i,j,k] = inp[i,j,k]
      end
    end
  end
end

@inline function back_transform!(field,p,s)
  A_mul_B!(real(field),p.pinv.p,complex(field))
  my_scale!(field,s)
  return nothing
end

@inline @par function my_scale!(field::AbstractArray{<:Real,3},s::@par(AbstractParameters))
  x = 1/(Nrx*Ny*Nz)
  @mthreads for k in 1:Nz
    for j in 1:Ny
      @msimd for i in 1:Nrx
        @inbounds field[i,j,k] = x*field[i,j,k]
      end
    end
  end
end 

@inline my_scale!(field::PaddedArray,s) = my_scale!(rawreal(field),s)

@inline function my_scale!(field::VectorField,s)
  my_scale!(field.rx,s)
  my_scale!(field.ry,s)
  my_scale!(field.rz,s)
end