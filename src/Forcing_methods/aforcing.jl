function (F::AForcing)(s::AbstractSimulation)

    cPlaneNumber = 1
    dt = get_dt(s)
    Tf = F.Tf
    ts = 10*dt
    omega = (2*π/(Tf*ts))
    alpha = F.α
    eps = Base.eps()
    _kf = F.Kf
    maxdk = F.maxDk
    avgWaveNumInShell2d = F.avgK
    nShells2d = length(avgWaveNumInShell2d)
    npts = F.npts

    R = F.R
    Em = F.Em
    Zf = F.Zf
    factor = F.factor
  
    s1 = fill!(F.forcex,0)
    s2 = fill!(F.forcey,0)

    u1 = s.u.c.x
    u2 = s.u.c.y
    u3 = s.u.c.z

    @inbounds for j=YRANGE
        for i=XRANGE
            k = fsqrt(muladd(KX[i],KX[i],KY[j]^2))
            n = round(Int,k/maxdk) + 1
            if (1 < n <= nShells2d)
                Ef = 0.5*(1 + (i>1))*(abs2(u1[i,j,1]) + abs2(u2[i,j,1]))/maxdk
                R[i,j] += dt*(-2*alpha*omega*R[i,j] - omega*omega*(Ef - Em[n]/npts[n])) 
                R[i,j] = max(0.0, R[i,j])
                factor[i,j] = fsqrt(R[i,j]/max(Ef,eps))*Zf[n]*dt
            end
        end
    end

    umag = 0.0
  #  # -------------------- Horizontal Forcing -------------------- 
    ff = 0.0
    fv = 0.0

    @inbounds for j=YRANGE
        conjFactX=1.0
        for i=XRANGE
            k = fsqrt(muladd(KX[i],KX[i],KY[j]^2))
            n = round(Int,k/maxdk) + 1
            if (1 < n <= nShells2d)
                if avgWaveNumInShell2d[n] <= _kf
                    s1[i,j,1] = u1[i,j,1]*factor[i,j]
                    s2[i,j,1] = u2[i,j,1]*factor[i,j]
                    ff += (abs2(s1[i,j,1]) + abs2(s2[i,j,1]))*conjFactX
                    fv += (proj(u1[i,j,1],s1[i,j,1]) + proj(u2[i,j,1],s2[i,j,1]))*conjFactX
                    conjFactX=2.0
                end
            end
        end
    end
    ff = ff*0.5/dt
    fv = fv/dt
    horizontalPower = ff+fv
    F.hp[] = horizontalPower
    powerFraction = 0.99

    # # -------------------- Vertical Forcing -------------------- 
    # /*
    #   Force the points with kx=ky=0 and kz=1,2,3,4,5 so that the power
    #   added is (1-powerFraction) of that added to the horizontal */
    ff = 0.0
    fv = 0.0
    umag = 0.0
    #     // get the sum of the existing velocites
    @inbounds for k = 2:6
        umag += fsqrt(abs2(u1[1,1,k]) + abs2(u2[1,1,k])) + fsqrt(abs2(u1[1,1,NZ-k+2]) + abs2(u2[1,1,NZ-k+2]))
    end

    f = dt*sqrt(2.0*horizontalPower*(1.0-powerFraction))/umag

    @inbounds for k=2:6
        s1[1,1,k] = u1[1,1,k]*f
        s2[1,1,k] = u2[1,1,k]*f
        s1[1,1,NZ-k+2] = u1[1,1,NZ-k+2]*f
        s2[1,1,NZ-k+2] = u2[1,1,NZ-k+2]*f
        ff += abs2(s1[1,1,k]) + abs2(s2[1,1,k]) + abs2(s1[1,1,NZ-k+2]) + abs2(s2[1,1,NZ-k+2])
        fv += proj(u1[1,1,k],s1[1,1,k]) + proj(u2[1,1,k],s2[1,1,k]) + proj(u1[1,1,NZ-k+2],s1[1,1,NZ-k+2]) + proj(u2[1,1,NZ-k+2],s2[1,1,NZ-k+2])
    end
    ff = ff*0.5/dt
    fv = fv/dt
    verticalPower = ff+fv
    F.vp[] = verticalPower
    return 1/dt
end
