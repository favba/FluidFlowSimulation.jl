@par function Euller!(u::AbstractArray{Float64,N},rhs::AbstractArray,dt::Real,s::@par(AbstractParameters)) where {N}
  @mthreads for i in 1:(N==3 ? Lrs : Lrv)
    #@inbounds u[i] += dt*rhs[i]
    @inbounds u[i] = muladd(dt,rhs[i],u[i])
  end
  return nothing
end

@par function Adams_Bashforth3rdO!(dt::Real, s::A) where {A<:@par(AbstractParameters)}
  dt12 = dt/12

  _tAdams_Bashforth3rdO!(s.u.rx,s.rhs.rx,dt12,s.rm1x,s.rm2x,s)
  _tAdams_Bashforth3rdO!(s.u.ry,s.rhs.ry,dt12,s.rm1y,s.rm2y,s)
  _tAdams_Bashforth3rdO!(s.u.rz,s.rhs.rz,dt12,s.rm1z,s.rm2z,s)

  copy!(s.rm2x,s.rm1x)
  copy!(s.rm2y,s.rm1y)
  copy!(s.rm2z,s.rm1z)
  mycopy!(s.rm1x,s.rhs.rx,s)
  mycopy!(s.rm1y,s.rhs.ry,s)
  mycopy!(s.rm1z,s.rhs.rz,s)

  if A<:ScalarParameters
    _tAdams_Bashforth3rdO!(rawreal(s.ρ),rawreal(s.ρrhs),dt12,s.rrm1,s.rrm2,s)
    copy!(s.rrm2,s.rrm1)
    mycopy!(s.rrm1,rawreal(s.ρrhs),s)
  end

  return nothing
end

@par function _tAdams_Bashforth3rdO!(u::AbstractArray{Complex128,3}, rhs::AbstractArray, dt12::Real, rm1::AbstractArray, rm2::AbstractArray, s::@par(AbstractParameters)) 
    @mthreads for kk in 1:length(Kzr)
      k = Kzr[kk]
      for (jj,j) in enumerate(Kyr)
        for i in Kxr
          #@inbounds u[i] += dt12*(23*rhs[i] - 16rm1[i] + 5rm2[i])
          @inbounds u[i,j,k] = muladd(muladd(23, rhs[i,j,k], muladd(-16, rm1[i,jj,kk], 5rm2[i,jj,kk])), dt12, u[i,j,k])
        end
      end
    end
end

@par function _tAdams_Bashforth3rdO!(u::AbstractArray{Float64,3}, rhs::AbstractArray, dt12::Real, rm1::AbstractArray, rm2::AbstractArray, s::@par(AbstractParameters)) 
    @mthreads for kk in 1:length(Kzr)
      k = Kzr[kk]
      for (jj,j) in enumerate(Kyr)
        for i in 1:(2length(Kxr))
          #@inbounds u[i] += dt12*(23*rhs[i] - 16rm1[i] + 5rm2[i])
          @inbounds u[i,j,k] = muladd(muladd(23, rhs[i,j,k], muladd(-16, rm1[i,jj,kk], 5rm2[i,jj,kk])), dt12, u[i,j,k])
        end
      end
    end
end
