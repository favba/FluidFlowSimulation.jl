(f::NoForcing)(dt,s) = nothing
initialize!(f::NoForcing,s) = nothing

@par function (F::RfForcing)(s::@par(AbstractSimulation))

  cPlaneNumber = 1
  dt = get_dt(s)
  Tf = getTf(F)
  ts = 10*dt
  omega = (2*π/(Tf*ts))
  alpha = getalpha(F)
  eps = Base.eps()
  _kf = getKf(F)
  maxdk = getmaxdk(F)
  avgWaveNumInShell2d = getavgk(F)
  nShells2d = length(avgWaveNumInShell2d)

  Ef = F.Ef
  R = F.R
  Em = F.Em
  Zf = getZf(F)
  factor = F.factor
  
  s1 = F.forcex
  s2 = F.forcey

  u1 = s.u.cx
  u2 = s.u.cy
  u3 = s.u.cz

  # // get current spectrum for k=0 plane
  calculate_u1u2_spectrum!(Ef,s.u, cPlaneNumber,s)

  for i=2:nShells2d
    if kx[i] <= _kf
      R[i] += dt*(-2*alpha*omega*R[i] - omega*omega*(Ef[i] - Em[i])) 
      R[i]=max(0.0, R[i])
      factor[i] = sqrt(R[i]/ifelse(Ef[i]>eps,Ef[i],eps))*Zf[i]*dt     
    end
  end

  umag = 0.0
#  # -------------------- Horizontal Forcing -------------------- 
  ff = 0.0
  fv = 0.0

  for j=1:Ny
    conjFactX=1.0
    for i=1:Nx
      K=sqrt(muladd(kx[i],kx[i],ky[j]^2))
      n = round(Int,K/maxdk) + 1
      if (1 < n < nShells2d)
        if avgWaveNumInShell2d[n] <= _kf
          s1[i,j,1] = u1[i,j,1]*factor[n]
          s2[i,j,1] = u2[i,j,1]*factor[n]
          ff += (abs2(s1[i,j,1]) + abs2(s2[i,j,1]))*conjFactX
          fv += (abs2(u1[i,j,1]) + abs2(u2[i,j,1]))*conjFactX
          conjFactX=2.0
        end
      end
    end
  end
  ff = ff*0.5/dt
  fv = fv*0.5/dt
  horizontalPower = ff+fv
  powerFraction = 0.99

  # # -------------------- Vertical Forcing -------------------- 
  # /*
  #   Force the points with kx=ky=0 and kz=1,2,3,4,5 so that the power
  #   added is (1-powerFraction) of that added to the horizontal */
  ff = 0.0
  fv = 0.0
  umag = 0.0
  #     // get the sum of the existing velocites
  for k = 2:6
    umag += sqrt(abs2(u1[1,1,k]) + abs2(u2[1,1,k])) + sqrt(abs2(u1[1,1,Nz-k+2]) + abs2(u2[1,1,Nz-k+2]))
  end

  f = dt*sqrt(2.0*horizontalPower*(1.0-powerFraction))/umag

  for k=2:6
    s1[1,1,k] = u1[1,1,k]*f
    s2[1,1,k] = u2[1,1,k]*f
    s1[1,1,Nz-k+2] = u1[1,1,Nz-k+2]*f
    s2[1,1,Nz-k+2] = u2[1,1,Nz-k+2]*f
    ff += abs2(s1[1,1,k]) + abs2(s2[1,1,k]) + abs2(s1[1,1,Nz-k+2]) + abs2(s2[1,1,Nz-k+2])
    fv += abs2(u1[1,1,k]) + abs2(u2[1,1,k]) + abs2(u1[1,1,Nz-k+2]) + abs2(u2[1,1,Nz-k+2])
  end
  ff = ff*0.5/dt
  fv = fv/dt
  verticalPower = ff+fv
  return 1/dt
end

function initialize!(f::RfForcing,s)
  if f.init
    calculate_u1u2_spectrum!(f.Em,s.u,1,s)
  end
  return nothing
end