@par function (F::QfForcing)(s::@par(AbstractSimulation))

    cPlaneNumber = 1
    dt = get_dt(s)

    _kf = getKf(F)
    maxdk = getmaxdk(F)
    avgWaveNumInShell2d = getavgk(F)
    nShells2d = length(avgWaveNumInShell2d)

    R = F.R
  
    s1 = fill!(F.forcex,0)
    s2 = fill!(F.forcey,0)

    u1 = s.u.c.x
    u2 = s.u.c.y

    Ef = F.Ef
    # // get current spectrum for k=1 plane
    calculate_u1u2_mag_spectrum!(Ef,s.u, cPlaneNumber)

  #  # -------------------- Horizontal Forcing -------------------- 
    ff = 0.0
    fv = 0.0

    powerFraction = 0.99
    @inbounds for j=YRANGE
        conjFactX=1.0
        for i=XRANGE
            k = fsqrt(muladd(KX[i],KX[i],KY[j]^2))
            n = round(Int,k/maxdk) + 1
            if (1 < n <= nShells2d)
                if avgWaveNumInShell2d[n] <= _kf
                    umag = fsqrt(abs2(u1[i,j,1]) + abs2(u2[i,j,1]))
                    Pp = R[n]
                    #fkn = (fsqrt(Ef[n]^2 + 2*dt*Pp*powerFraction) - Ef[n])
                    fkn = dt*Pp*powerFraction/Ef[n]
                    s1[i,j,1] = (u1[i,j,1]/umag)*fkn
                    s2[i,j,1] = (u2[i,j,1]/umag)*fkn
                    #ff += (abs2(s1[i,j,1]) + abs2(s2[i,j,1]))*conjFactX
                    fv += (proj(u1[i,j,1],s1[i,j,1]/dt) + proj(u2[i,j,1],s2[i,j,1]/dt))*conjFactX
                end
            end
            conjFactX=2.0
        end
    end
    ff = ff*0.5/dt
    fv = fv
    horizontalPower = ff+fv
    F.hp[] = horizontalPower

    # # -------------------- Vertical Forcing -------------------- 
    # /*
    #   Force the points with kx=ky=0 and kz=1,2,3,4,5 so that the power
    #   added is (1-powerFraction) of that added to the horizontal */
    ff = 0.0
    fv = 0.0
    umagsq = 0.0
    #     // get the sum of the existing velocites
    @inbounds for k = 4:6
        umagsq += abs2(u1[1,1,k]) + abs2(u2[1,1,k]) + abs2(u1[1,1,NZ-k+2]) + abs2(u2[1,1,NZ-k+2])
    end

    #f = dt*sqrt(2.0*horizontalPower*(1.0-powerFraction))/umag
    f = dt*horizontalPower*(1-powerFraction)/(umagsq*powerFraction)

    @inbounds for k=4:6
        s1[1,1,k] = u1[1,1,k]*f
        s2[1,1,k] = u2[1,1,k]*f
        s1[1,1,NZ-k+2] = u1[1,1,NZ-k+2]*f
        s2[1,1,NZ-k+2] = u2[1,1,NZ-k+2]*f
        #ff += abs2(s1[1,1,k]) + abs2(s2[1,1,k]) + abs2(s1[1,1,NZ-k+2]) + abs2(s2[1,1,NZ-k+2])
        fv += proj(u1[1,1,k],s1[1,1,k]/dt) + proj(u2[1,1,k],s2[1,1,k]/dt) + proj(u1[1,1,NZ-k+2],s1[1,1,NZ-k+2]/dt) + proj(u2[1,1,NZ-k+2],s2[1,1,NZ-k+2]/dt)
    end
    ff = ff*0.5/dt
    fv = fv
    verticalPower = ff+fv
    F.vp[] = verticalPower
    return 1/dt
end
