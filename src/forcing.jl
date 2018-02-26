@par function (F::RfForcing)(dt::Real,s::@par(AbstractSimulation))

  cPlaneNumber = 1

  Tf = getTf(F)
  ts = 10*dt
  omega = (2*π/(Tf*ts))
  alpha = getalpha(F)
  eps = Base.eps()
  _kf = getKf(F)

  Ef = F.Ef
  R = F.R
  Em = F.Em
  Zf = F.Zf
  factor = F.factor

  # // get current spectrum for k=0 plane
  calculate_u1u2_spectrum!(Ef,s.u, cPlaneNumber,s)

  #      for (i=1; i<nShells2d; i++)
  #        if(avgWaveNumInShell2d[i]<=_kf)
  #          {
  #            dRdt[i]= -2*alpha*omega*R[i] - omega*omega*(Ef[i] - Em[i]);  
  #            R[i]= R[i] + dRdt[i]*dt;
  #            R[i]=max(0.0, R[i]);
  #            factor[i] = sqrt(R[i]/max(Ef[i],eps))*Zf[i]*dt;    
  #          }
  # 
  for i=2:nShells2d
    if kx[i] <= _kf
      R[i] += dt*(-2*alpha*omega*R[i] - omega*omega*(Ef[i] - Em[i])) 
      R[i]=max(0.0, R[i])
      factor[i] = sqrt(R[i]/ifelse(Ef[i]>eps,Ef[i],eps))*Zf[i]*dt     
    end
  end

#  Int ic,ic2,n;
#  Float horizontalPower;
#  Float verticalPower;
#  Float umag = 0.0; 
  umag = 0.0
#  # -------------------- Horizontal Forcing -------------------- 
#  Float conjFactX;
#  Float ff=0;
#  Float fv=0;
  ff = 0.0
  fv = 0.0

  # for (j=yl; j<=yh; j++) # What is yl and yh ?
  #   for (conjFactX=1.0, i=0; i < nkx; i++)
  #     {
  #       n = getn2d(i,j);
  #       if ((n > 0) && (n < nShells2d))
  #         if(avgWaveNumInShell2d[n]<=_kf)
  #           {
  #             ic  = indexc(i,j,0); 
  #             s1.c[ic].r= u1.c[ic].r*factor[n];
  #             s1.c[ic].i= u1.c[ic].i*factor[n];
  #             s2.c[ic].r= u2.c[ic].r*factor[n];
  #             s2.c[ic].i= u2.c[ic].i*factor[n];
              
  #             ff += (s1.c[ic].r*s1.c[ic].r + s1.c[ic].i*s1.c[ic].i +
  #                     s2.c[ic].r*s2.c[ic].r + s2.c[ic].i*s2.c[ic].i )*conjFactX;
  #             fv += (u1.c[ic].r*s1.c[ic].r + u1.c[ic].i*s1.c[ic].i +
  #                     u2.c[ic].r*s2.c[ic].r + u2.c[ic].i*s2.c[ic].i)*conjFactX;
  #             conjFactX=2.0;
  #           }
  #     }

  for j=1:Ny
    conjFactX=1.0
    for i=1:Nx
      K=sqrt(muladd(kx[i],kx[i],ky[j]^2))
      n = round(Int,K/maxdk) + 1
      if avgWaveNumInShell2d[n] <= _kf
        s1[i,j,1] = u1[i,j,1]*factor[n]
        s2[i,j,1] = u2[i,j,1]*factor[n]
        ff += (abs2(s1[i,j,1]) + abs2(s2[i,j,1]))*conjFactX
        fv += (abs2(u1[i,j,1]) + abs2(u2[i,j,1]))*conjFactX
        conjFactX=2.0
      end
    end
  end
  ff = ff*0.5/dt
  fv = fv*0.5/dt
  horizontalPower = ff+fv
  powerFraction = 0.99
  # ff = sumToAll(ff)*0.5/dt;
  # fv = sumToAll(fv)/dt;
  # horizontalPower = ff+fv;

  # # -------------------- Vertical Forcing -------------------- 
  # /*
  #   Force the points with kx=ky=0 and kz=1,2,3,4,5 so that the power
  #   added is (1-powerFraction) of that added to the horizontal */
  # ff=0;
  # fv=0;
  ff = 0.0
  fv = 0.0
  # if (_my_pe()==0)
  #   {
  #     umag=0;
  umag = 0.0
  #     // get the sum of the existing velocites
  #     for (k=1; k<=5; k++)
  # {
  #         ic=indexc(0,0,k);
  #         ic2=indexc(0,0,Nz-k);
  #         umag += (getUmag(u1, u2, ic)+getUmag(u1, u2, ic2));
  # }
  
  for k = 2:6
    umag += sqrt(abs2(u1[1,1,k]) + abs2(u2[1,1,k])) + sqrt(abs2(u1[1,1,Nz-k+1]) + abs2(u2[1,1,Nz-k+1]))
  end

  #     Float f = dt*sqrt(2.0*horizontalPower*(1.0-powerFraction))/umag;
  f = dt*sqrt(2.0*horizontalPower*(1.0-powerFraction))/umag

  #     for(k=1; k<=5; k++)
  for k=2:6
  # {
  #         ic=indexc(0,0,k);
  #         s1.c[ic].r = u1.c[ic].r*f;
  #         s1.c[ic].i = u1.c[ic].i*f;
  #         s2.c[ic].r = u2.c[ic].r*f;
  #         s2.c[ic].i = u2.c[ic].i*f;
    s1[1,1,k] = u1[1,1,k]*f
    s2[1,1,k] = u2[1,1,k]*f
  #         ic2=indexc(0,0,Nz-k);
  #         s1.c[ic2].r = u1.c[ic2].r*f;
  #         s1.c[ic2].i = u1.c[ic2].i*f;
  #         s2.c[ic2].r = u2.c[ic2].r*f;
  #         s2.c[ic2].i = u2.c[ic2].i*f;
    s1[1,1,Nz-k+1] = u1[1,1,Nz-k+1]*f
    s2[1,1,Nz-k+1] = u2[1,1,Nz-k+1]*f
  #         ff += (s1.c[ic].r*s1.c[ic].r + s1.c[ic].i*s1.c[ic].i + 
  #                 s2.c[ic].r*s2.c[ic].r + s2.c[ic].i*s2.c[ic].i +
  #                 s1.c[ic2].r*s1.c[ic2].r + s1.c[ic2].i*s1.c[ic2].i + 
  #                 s2.c[ic2].r*s2.c[ic2].r + s2.c[ic2].i*s2.c[ic2].i);
    ff += abs2(s1[1,1,k]) + abs2(s2[1,1,k]) + abs2(s1[1,1,Nz-k+1]) + abs2(s2[1,1,Nz-k+1])
  #         fv += (u1.c[ic].r*s1.c[ic].r + u1.c[ic].i*s1.c[ic].i +
  #                 u2.c[ic].r*s2.c[ic].r + u2.c[ic].i*s2.c[ic].i +
  #                 u1.c[ic2].r*s1.c[ic2].r + u1.c[ic2].i*s1.c[ic2].i +
  #                 u2.c[ic2].r*s2.c[ic2].r + u2.c[ic2].i*s2.c[ic2].i);
    fv += abs2(u1[1,1,k]) + abs2(u2[1,1,k]) + abs2(u1[1,1,Nz-k+1]) + abs2(u2[1,1,Nz-k+1])
  # }
  end
  #     ff = ff*0.5/dt;
  #     fv = fv/dt;
  #     verticalPower = ff+fv;
  #   }
  ff = ff*0.5/dt
  fv = fv/dt
  verticalPower = ff+fv
  # return 1.0/dt; 
  # }
  return 1/dt
end