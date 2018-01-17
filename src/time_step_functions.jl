@par function Euller!(u::AbstractArray{Float64,N},rhs::AbstractArray,dt::Real,s::@par(AbstractSimulation)) where {N}
  @mthreads for i in 1:(N===3 ? Lrs : Lrv)
    #@inbounds u[i] += dt*rhs[i]
    @fastmath @inbounds u[i] = muladd(dt,rhs[i],u[i])
  end
  return nothing
end

@par function Adams_Bashforth3rdO!(dt::Real, s::A) where {A<:@par(AbstractSimulation)}
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

  if isscalar(A)
    _tAdams_Bashforth3rdO!(parent(real(s.ρ)),parent(real(s.ρrhs)),dt12,s.rrm1,s.rrm2,s)
    copy!(s.rrm2,s.rrm1)
    mycopy!(s.rrm1,parent(real(s.ρrhs)),s)
  end

  return nothing
end

@inbounds @par function _tAdams_Bashforth3rdO!(u::AbstractArray{Complex128,3}, rhs::AbstractArray, dt12::Real, rm1::AbstractArray, rm2::AbstractArray, s::@par(AbstractSimulation)) 
  @mthreads for kk in 1:length(Kzr)
    k = Kzr[kk]
    jj::Int = 1
    for y in Kyr, j in y
      @fastmath @msimd for i in Kxr
        #u[i] += dt12*(23*rhs[i] - 16rm1[i] + 5rm2[i])
        u[i,j,k] = muladd(muladd(23, rhs[i,j,k], muladd(-16, rm1[i,jj,kk], 5rm2[i,jj,kk])), dt12, u[i,j,k])
      end
    jj+=1
    end
  end
end

@inbounds @par function _tAdams_Bashforth3rdO!(u::AbstractArray{Float64,3}, rhs::AbstractArray, dt12::Real, rm1::AbstractArray, rm2::AbstractArray, s::@par(AbstractSimulation)) 
  @mthreads for kk in 1:length(Kzr)
    k = Kzr[kk]
    jj::Int = 1
    for y in Kyr, j in y
      @fastmath @msimd for i in 1:(2length(Kxr))
        #u[i] += dt12*(23*rhs[i] - 16rm1[i] + 5rm2[i])
        u[i,j,k] = muladd(muladd(23, rhs[i,j,k], muladd(-16, rm1[i,jj,kk], 5rm2[i,jj,kk])), dt12, u[i,j,k])
      end
    jj+=1
    end
  end
end
